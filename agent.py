import os
import time
import json
import gc
import asyncio
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from telegram import Bot
from supabase import create_client, Client

load_dotenv()

CONFIG_FILE = "config.json"
last_alert_time = {}

# Initialize Supabase Client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

def load_config():
    default_config = {
        "account_balance": 1000.0,
        "risk_pct": 1.0,
        "proximity_threshold_pct": 2.0,
        "min_adx": 20.0,
        "min_atr_pct": 0.4,
        "atr_mult_sl": 1.5,
        "min_rr_ratio": 1.5,
        "scan_interval_minutes": 15,
        "alert_cooldown_hours": 4,
        "watchlist": [
            "ONDO/USDT", "PENDLE/USDT", "LINK/USDT", "TIA/USDT", "NEAR/USDT",
            "SOL/USDT", "AR/USDT", "FET/USDT", "RENDER/USDT", "TAO/USDT",
            "SYRUP/USDT", "SEI/USDT", "CFG/USDT", "HNT/USDT", "XRP/USDT", "DOGE/USDT"
        ],
        # Strategy Upgrades Configurations
        "use_mtf_tema_alignment": True,
        "htf_tema_period": 200,
        "htf_alignment_mode": "Price Level",  # "Price Level" or "TEMA Slope"
        "use_adaptive_atr_trail": True,
        "atr_trail_mult": 1.5,
        "use_equity_curve_filter": True,
        "eq_ma_period": 10,
        "reduced_risk_factor": 0.5,
        
        # Recommendation 2: Asset-Specific Parameter Overrides Map
        # Format: "SYMBOL": { override_key: override_value }
        "asset_overrides": {
            "GBP/JPY": {
                "min_adx": 25.0,
                "atr_mult_sl": 2.0,
                "proximity_threshold_pct": 2.5
            },
            "SOL/USDT": {
                "min_adx": 22.0,
                "atr_mult_sl": 1.8,
                "proximity_threshold_pct": 2.2
            }
        }
    }
    
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            existing_config = json.load(f)
            return {**default_config, **existing_config}
    except Exception as e:
        print(f"⚠️ Error loading {CONFIG_FILE}, falling back to defaults: {e}")
        return default_config

def get_symbol_config(symbol: str, global_config: dict) -> dict:
    """
    Combines global configurations with symbol-specific overrides.
    Allows tailored risk and parameters per asset while using a single codebase.
    """
    symbol_config = global_config.copy()
    overrides_map = global_config.get("asset_overrides", {})
    
    if symbol in overrides_map and isinstance(overrides_map[symbol], dict):
        symbol_config.update(overrides_map[symbol])
        
    return symbol_config

def get_exchange():
    try:
        print("Connecting to MEXC...")
        exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 20000})
        exchange.load_markets()
        print("Connected to MEXC successfully.")
        return exchange
    except Exception as e:
        print(f"MEXC failed ({e}), falling back to Gate.io...")
        exchange = ccxt.gateio({'enableRateLimit': True, 'timeout': 20000})
        return exchange

exchange = get_exchange()

def init_db():
    if not supabase:
        print("⚠️ Supabase credentials missing. Check SUPABASE_URL and SUPABASE_KEY.")
        return
    print("⚡ Connected to Supabase external database.")

def calculate_native_tema(series: pd.Series, length: int = 200) -> pd.Series:
    """Pure Pandas fallback calculation for TEMA 200."""
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    res = 3 * ema1 - 3 * ema2 + ema3
    del ema1, ema2, ema3
    return res

def fetch_market_data(symbol, timeframe='1h', limit=350, tema_length=200):
    """Fetches market data with float32 downcasting to optimize RAM."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

    if not ohlcv or len(ohlcv) < 10:
        return None

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    del ohlcv
    
    float_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 1. TEMA Calculation
    if len(df) >= tema_length:
        try:
            df['tema_200'] = ta.tema(df['close'], length=tema_length)
            if df['tema_200'].dropna().empty:
                df['tema_200'] = calculate_native_tema(df['close'], length=tema_length)
        except Exception:
            df['tema_200'] = calculate_native_tema(df['close'], length=tema_length)
            
        df['tema_200'] = pd.to_numeric(df['tema_200'], errors='coerce').astype(np.float32)

    # 2. ADX (14) Calculation
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None and not adx_df.empty:
            adx_cols = [c for c in adx_df.columns if c.startswith('ADX_')]
            df['adx'] = pd.to_numeric(adx_df[adx_cols[0]] if adx_cols else adx_df.iloc[:, 0], errors='coerce').astype(np.float32)
            del adx_df
        else:
            df['adx'] = np.float32(0.0)
    except Exception:
        df['adx'] = np.float32(0.0)

    # 3. ATR % Calculation
    try:
        raw_atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'] = pd.to_numeric(raw_atr, errors='coerce').astype(np.float32)
        df['atr_pct'] = ((df['atr'] / df['close']) * 100.0).astype(np.float32)
        del raw_atr
    except Exception:
        df['atr'] = np.float32(0.0)
        df['atr_pct'] = np.float32(0.0)

    return df

def calculate_volume_profile(df, num_bins=30):
    price_min = float(df['low'].min())
    price_max = float(df['high'].max())

    if price_min == price_max or pd.isna(price_min) or pd.isna(price_max):
        return None, []

    bins = np.linspace(price_min, price_max, num_bins)
    df['bin'] = pd.cut(df['close'], bins=bins)

    volume_profile = df.groupby('bin', observed=False)['volume'].sum().reset_index()
    volume_profile['price_mid'] = volume_profile['bin'].apply(lambda x: float(x.mid))

    if volume_profile['volume'].sum() == 0:
        return None, []

    poc_row = volume_profile.loc[volume_profile['volume'].idxmax()]
    poc_price = float(poc_row['price_mid'])

    hvn_nodes = volume_profile.sort_values(by='volume', ascending=False).head(3)
    hvn_list = [float(x) for x in hvn_nodes['price_mid'].tolist()]
    
    del volume_profile, hvn_nodes
    return poc_price, hvn_list

def calculate_effective_risk(base_risk_pct, config):
    """Calculates position risk factor based on equity curve drawdown filter."""
    if not config.get("use_equity_curve_filter", False) or not supabase:
        return base_risk_pct

    try:
        res = supabase.table("trade_journal").select("realized_pnl_usd").in_("status", ["CLOSED_TP2", "STOPPED_OUT"]).order("id", desc=True).limit(50).execute()
        closed_trades = res.data
        if len(closed_trades) >= config.get("eq_ma_period", 10):
            initial_balance = float(config.get("account_balance", 1000.0))
            pnl_list = [float(t.get("realized_pnl_usd", 0.0)) for t in reversed(closed_trades)]
            equity_series = np.cumsum([initial_balance] + pnl_list)
            
            eq_ma_period = int(config.get("eq_ma_period", 10))
            recent_ma = np.mean(equity_series[-eq_ma_period:])
            current_eq = equity_series[-1]
            
            if current_eq < recent_ma:
                reduced_factor = float(config.get("reduced_risk_factor", 0.5))
                effective_risk = base_risk_pct * reduced_factor
                print(f"📉 Equity Curve Filter Active: Current (${current_eq:.2f}) < MA (${recent_ma:.2f}). Risk reduced to {effective_risk:.2f}%")
                return effective_risk
    except Exception as e:
        print(f"Error checking equity curve filter: {e}")

    return base_risk_pct

def calculate_fixed_risk_position(account_balance, entry_price, stop_loss_price, risk_pct=1.0):
    risk_amount = float(account_balance) * (float(risk_pct) / 100.0)
    price_risk_per_unit = abs(float(entry_price) - float(stop_loss_price))

    if price_risk_per_unit == 0:
        return 0.0, 0.0, 0.0

    units = risk_amount / price_risk_per_unit
    position_size_usdt = units * float(entry_price)
    return float(units), float(position_size_usdt), float(risk_amount)

def log_trade_signal(symbol, reasons, entry, sl, tp1, tp2, position_usdt, risk_usd):
    if not supabase:
        print("⚠️ Cannot log trade: Supabase connection unavailable.")
        return

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reasons_str = " | ".join(reasons)

    data = {
        "timestamp": timestamp_str,
        "symbol": symbol,
        "trigger_reason": reasons_str,
        "entry_price": float(entry),
        "stop_loss": float(sl),
        "take_profit_1": float(tp1),
        "take_profit_2": float(tp2),
        "position_usdt": float(position_usdt),
        "max_risk_usd": float(risk_usd),
        "status": "OPEN"
    }

    try:
        supabase.table("trade_journal").insert(data).execute()
        print(f"💾 Trade signal logged to Supabase for {symbol}.")
    except Exception as e:
        print(f"Error inserting trade into Supabase: {e}")

def safe_format(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):.4f}"

# =====================================================================
# ACTIVE TRADE EVALUATOR ENGINE (WITH HARD BREAKEVEN PROTECTION)
# =====================================================================
async def evaluate_active_trades(bot, chat_id, global_config):
    if not supabase:
        return

    try:
        response = supabase.table("trade_journal").select("*").in_("status", ["OPEN", "TP1_HIT"]).execute()
        open_trades = response.data
    except Exception as e:
        print(f"Error fetching active trades from Supabase: {e}")
        return

    if not open_trades:
        return

    print(f"\n🔄 Evaluating {len(open_trades)} active signals in Supabase database...")

    for trade in open_trades:
        symbol = str(trade.get('symbol', 'UNKNOWN'))
        
        # Load asset-specific overrides for trade management
        config = get_symbol_config(symbol, global_config)
        
        try:
            trade_id = trade['id']
            entry = float(trade['entry_price'])
            sl = float(trade['stop_loss'])
            tp1 = float(trade['take_profit_1'])
            tp2 = float(trade['take_profit_2'])
            risk_usd = float(trade['max_risk_usd'])
            current_status = str(trade['status'])
            trade_time_str = str(trade['timestamp'])

            trade_dt = datetime.strptime(trade_time_str, "%Y-%m-%d %H:%M:%S")
            hours_elapsed = (datetime.now() - trade_dt).total_seconds() / 3600

            candles_needed = max(50, int(hours_elapsed * 4) + 10)
            candles_needed = min(candles_needed, 350)

            df_candles = fetch_market_data(symbol, timeframe='15m', limit=candles_needed)
            if df_candles is None or df_candles.empty:
                continue

            latest_close = float(df_candles['close'].iloc[-1])
            latest_low = float(df_candles['low'].min())
            latest_high = float(df_candles['high'].max())
            latest_atr = float(df_candles['atr'].iloc[-1]) if 'atr' in df_candles else 0.0
            close_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            is_long = sl < entry

            # Dynamic Adaptive ATR Trailing Stop Update (With Hard Breakeven Protection)
            if config.get("use_adaptive_atr_trail", True) and latest_atr > 0:
                trail_mult = float(config.get("atr_trail_mult", config.get("atr_trail_multiplier", 1.5)))
                if is_long:
                    new_trail_sl = latest_close - (latest_atr * trail_mult)
                    # Option A Clamping: Clamp stop loss so it never drops below entry if TP1 was reached
                    if current_status == "TP1_HIT":
                        new_trail_sl = max(new_trail_sl, entry)
                    
                    if new_trail_sl > sl:
                        sl = new_trail_sl
                        supabase.table("trade_journal").update({"stop_loss": float(sl)}).eq("id", trade_id).execute()
                        print(f"🛡️ Dynamic Trailing Stop updated for {symbol} (LONG) to ${sl:.4f}")
                else:
                    new_trail_sl = latest_close + (latest_atr * trail_mult)
                    # Option A Clamping: Clamp stop loss so it never rises above entry if TP1 was reached
                    if current_status == "TP1_HIT":
                        new_trail_sl = min(new_trail_sl, entry)
                        
                    if new_trail_sl < sl:
                        sl = new_trail_sl
                        supabase.table("trade_journal").update({"stop_loss": float(sl)}).eq("id", trade_id).execute()
                        print(f"🛡️ Dynamic Trailing Stop updated for {symbol} (SHORT) to ${sl:.4f}")

            # 1. Check Stop Loss
            sl_hit = (latest_low <= sl) if is_long else (latest_high >= sl)
            if sl_hit:
                # If stopped out after TP1 was hit, realized loss is capped at 0R / Breakeven
                if current_status == "TP1_HIT":
                    pnl_usd = 0.0
                    realized_r = 0.0
                else:
                    pnl_usd = -risk_usd
                    realized_r = -1.0

                update_data = {
                    "status": "STOPPED_OUT",
                    "exit_price": float(sl),
                    "closed_timestamp": str(close_time),
                    "realized_pnl_usd": float(pnl_usd),
                    "realized_r": float(realized_r)
                }
                supabase.table("trade_journal").update(update_data).eq("id", trade_id).execute()

                if bot and chat_id:
                    status_title = "🛡️ **TRADE CLOSED AT BREAKEVEN / TRAILING STOP: " if current_status == "TP1_HIT" else "🛑 **TRADE STOPPED OUT: "
                    msg = (
                        f"{status_title}{symbol}**\n\n"
                        f"• Exit Price: `${sl:.4f}`\n"
                        f"• Realized PnL: `${pnl_usd:.2f}` ({realized_r:.2f}R)\n"
                        f"• Timestamp: `{close_time}`"
                    )
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

            # 2. Check Take Profit 2
            tp2_hit = (latest_high >= tp2) if is_long else (latest_low <= tp2)
            if tp2_hit:
                r_multiple = abs(tp2 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
                pnl_usd = risk_usd * r_multiple

                update_data = {
                    "status": "CLOSED_TP2",
                    "exit_price": float(tp2),
                    "closed_timestamp": str(close_time),
                    "realized_pnl_usd": float(pnl_usd),
                    "realized_r": float(r_multiple)
                }
                supabase.table("trade_journal").update(update_data).eq("id", trade_id).execute()

                if bot and chat_id:
                    msg = (
                        f"🎯 **FULL TARGET HIT (TP2): {symbol}** 🎯\n\n"
                        f"• Target Price: `${tp2:.4f}`\n"
                        f"• Realized PnL: `+${pnl_usd:.2f}` (+{r_multiple:.2f}R)\n"
                        f"• Timestamp: `{close_time}`"
                    )
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

            # 3. Check Take Profit 1
            tp1_hit = (latest_high >= tp1) if is_long else (latest_low <= tp1)
            if tp1_hit and current_status == 'OPEN':
                r_multiple = abs(tp1 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0

                # Enforce Hard Breakeven Protection immediately on TP1 Hit
                new_sl = entry
                if is_long:
                    if sl < entry:
                        sl = new_sl
                else:
                    if sl > entry:
                        sl = new_sl

                supabase.table("trade_journal").update({
                    "status": "TP1_HIT",
                    "stop_loss": float(sl)
                }).eq("id", trade_id).execute()

                if bot and chat_id:
                    msg = (
                        f"✅ **FIRST TARGET REACHED (TP1): {symbol}** ✅\n\n"
                        f"• Milestone Price: `${tp1:.4f}`\n"
                        f"• Un-realized Gain: `+{r_multiple:.2f}R`\n"
                        f"• Hard Breakeven Protected: Stop Loss set to `${sl:.4f}`"
                    )
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

            del df_candles

        except Exception as e:
            print(f"Error evaluating trade ID {trade.get('id')} for {symbol}: {e}")

# =====================================================================
# ANALYSIS & SCANNER LOOP (ASSET-SPECIFIC OVERRIDES INTEGRATED)
# =====================================================================
async def analyze_symbol(symbol, bot, chat_id, global_config):
    current_time = time.time()

    # Load symbol parameters with individual overrides applied
    config = get_symbol_config(symbol, global_config)

    account_balance = float(config.get("account_balance", 1000.0))
    base_risk_pct = float(config.get("risk_pct", 1.0))
    min_rr_threshold = float(config.get("min_rr_ratio", 1.5))
    proximity_threshold = float(config.get("proximity_threshold_pct", 2.0))
    cooldown_hours = float(config.get("alert_cooldown_hours", 4))
    min_adx = float(config.get("min_adx", 20.0))
    min_atr_pct = float(config.get("min_atr_pct", 0.4))
    atr_mult_sl = float(config.get("atr_mult_sl", 1.5))

    if symbol in last_alert_time:
        elapsed_hours = (current_time - last_alert_time[symbol]) / 3600
        if elapsed_hours < cooldown_hours:
            print(f"[{symbol}] On cooldown ({elapsed_hours:.1f}h / {cooldown_hours}h elapsed). Skipping scan.")
            return

    df_1h = fetch_market_data(symbol, timeframe='1h', limit=350)
    
    htf_lookback = int(config.get("htf_tema_period", 200))
    df_4h = fetch_market_data(symbol, timeframe='4h', limit=350, tema_length=htf_lookback)

    if df_1h is None or df_4h is None:
        return

    current_price = float(df_1h['close'].iloc[-1])
    val_1h = float(df_1h['tema_200'].dropna().iloc[-1]) if 'tema_200' in df_1h and not df_1h['tema_200'].dropna().empty else None
    val_4h = float(df_4h['tema_200'].dropna().iloc[-1]) if 'tema_200' in df_4h and not df_4h['tema_200'].dropna().empty else None
    poc_price, hvn_prices = calculate_volume_profile(df_1h, num_bins=30)

    # Proximity Triggers
    triggered_reasons = []

    if val_1h is not None:
        dist_1h = abs(current_price - val_1h) / val_1h * 100
        if dist_1h <= proximity_threshold:
            triggered_reasons.append(f"Near 1H 200 TEMA ({dist_1h:.2f}% away)")

    if val_4h is not None:
        dist_4h = abs(current_price - val_4h) / val_4h * 100
        if dist_4h <= proximity_threshold:
            triggered_reasons.append(f"Near 4H HTF TEMA ({dist_4h:.2f}% away)")

    if poc_price is not None:
        dist_poc = abs(current_price - poc_price) / poc_price * 100
        if dist_poc <= proximity_threshold:
            triggered_reasons.append(f"Near Volume POC ({dist_poc:.2f}% away)")

    if not triggered_reasons:
        print(f"[{symbol}] Price (${current_price:.4f}) outside proximity threshold ({proximity_threshold}%). No trigger.")
        del df_1h, df_4h
        return

    # Multi-Timeframe TEMA Trend Alignment Filter
    if config.get("use_mtf_tema_alignment", config.get("use_mtf_alignment", True)) and val_1h is not None and val_4h is not None:
        alignment_mode = config.get("htf_alignment_mode", "Price Level")
        base_long = current_price >= val_1h
        base_short = current_price < val_1h

        if alignment_mode == "Price Level":
            htf_long = current_price >= val_4h
            htf_short = current_price < val_4h
        else:  # Slope Mode
            prev_val_4h = float(df_4h['tema_200'].dropna().iloc[-2]) if len(df_4h['tema_200'].dropna()) > 1 else val_4h
            htf_long = val_4h > prev_val_4h
            htf_short = val_4h < prev_val_4h

        if (base_long and not htf_long) or (base_short and not htf_short):
            print(f"[{symbol}] ❌ Filter Skipped: 1H and 4H HTF TEMA trend alignment failed ({alignment_mode}).")
            del df_1h, df_4h
            return
        else:
            triggered_reasons.append(f"Aligned with 4H HTF Trend ({alignment_mode})")

    # Hard Indicator Regime Filters
    current_adx = float(df_1h['adx'].dropna().iloc[-1]) if 'adx' in df_1h else 0.0
    current_atr_pct = float(df_1h['atr_pct'].dropna().iloc[-1]) if 'atr_pct' in df_1h else 0.0

    if current_adx < min_adx:
        print(f"[{symbol}] ❌ Filter Skipped: ADX ({current_adx:.2f}) below symbol threshold ({min_adx}).")
        del df_1h, df_4h
        return

    if current_atr_pct < min_atr_pct:
        print(f"[{symbol}] ❌ Filter Skipped: ATR% ({current_atr_pct:.2f}%) below symbol threshold ({min_atr_pct}%).")
        del df_1h, df_4h
        return

    print(f"🎯 TRIGGER & FILTERS MATCH for {symbol}: {', '.join(triggered_reasons)} (ADX: {current_adx:.2f}, ATR%: {current_atr_pct:.2f}%)")

    # Direction & Trade Sizing Calculations (Applying Symbol-Specific ATR Multiplier)
    tema_ref = val_1h if val_1h is not None else current_price
    direction = "LONG" if current_price >= tema_ref else "SHORT"
    entry_price = current_price

    atr_val = float(df_1h['atr'].dropna().iloc[-1]) if 'atr' in df_1h and not pd.isna(df_1h['atr'].iloc[-1]) else entry_price * 0.01

    if direction == "LONG":
        stop_loss = entry_price - (atr_mult_sl * atr_val) if atr_val > 0 else entry_price * 0.985
        valid_hvns = [h for h in hvn_prices if h > entry_price] if hvn_prices else []
        tp1 = valid_hvns[0] if valid_hvns else entry_price + (atr_mult_sl * atr_val * min_rr_threshold)
        tp2 = poc_price if (poc_price and poc_price > entry_price) else entry_price + (atr_mult_sl * atr_val * min_rr_threshold * 1.5)
    else:
        stop_loss = entry_price + (atr_mult_sl * atr_val) if atr_val > 0 else entry_price * 1.015
        valid_hvns = [h for h in hvn_prices if h < entry_price] if hvn_prices else []
        tp1 = valid_hvns[-1] if valid_hvns else entry_price - (atr_mult_sl * atr_val * min_rr_threshold)
        tp2 = poc_price if (poc_price and poc_price < entry_price) else entry_price - (atr_mult_sl * atr_val * min_rr_threshold * 1.5)

    risk_amount_per_unit = abs(entry_price - stop_loss)
    reward_amount_per_unit = abs(tp1 - entry_price)

    rr_ratio = reward_amount_per_unit / risk_amount_per_unit if risk_amount_per_unit > 0 else 0.0

    if rr_ratio < min_rr_threshold:
        print(f"[{symbol}] 🛑 SIGNAL REJECTED: Risk/Reward ({rr_ratio:.2f}R) is below threshold ({min_rr_threshold:.2f}R).")
        del df_1h, df_4h
        return

    # Apply Equity Curve Filter Adjustments
    effective_risk_pct = calculate_effective_risk(base_risk_pct, config)

    units, position_usdt, risk_usd = calculate_fixed_risk_position(
        account_balance, entry_price, stop_loss, risk_pct=effective_risk_pct
    )

    log_trade_signal(
        symbol, triggered_reasons, entry_price, 
        stop_loss, tp1, tp2, position_usdt, risk_usd
    )

    last_alert_time[symbol] = current_time

    reasons_text = "\n".join([f"• {r}" for r in triggered_reasons])
    message = (
        f"🎯 **PROXIMITY ALERT: {symbol}** 🎯\n\n"
        f"**Trigger Conditions Met:**\n{reasons_text}\n\n"
        f"**Current Price:** `${current_price:.4f}`\n\n"
        f"📈 **Technical Confluence:**\n"
        f"• 1H 200 TEMA: {safe_format(val_1h)}\n"
        f"• 4H HTF TEMA: {safe_format(val_4h)}\n"
        f"• Point of Control (POC): {safe_format(poc_price)}\n"
        f"• ADX (14): `{current_adx:.2f}` (Min: {min_adx})\n"
        f"• ATR %: `{current_atr_pct:.2f}%` (Min: {min_atr_pct}%)\n"
        f"• ATR Multiplier (SL): `{atr_mult_sl}x`\n\n"
        f"🎯 **Trade Parameters ({effective_risk_pct:.2f}% Risk Model):**\n"
        f"• Direction: `{direction}`\n"
        f"• Entry Zone: `${entry_price:.4f}`\n"
        f"• Stop Loss: `${stop_loss:.4f}` (Risk: `${risk_usd:.2f}`)\n"
        f"• Target 1 (HVN): `${tp1:.4f}`\n"
        f"• Target 2 (Macro): `${tp2:.4f}`\n\n"
        f"💰 **Position Sizing:**\n"
        f"• Position Value: `${position_usdt:.2f}` ({units:.2f} units)\n"
        f"• Risk/Reward Ratio: `{rr_ratio:.2f}R`\n\n"
        f"📄 Signal logged to Supabase Database"
    )

    if bot and chat_id:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")

    del df_1h, df_4h

async def run_scanner():
    global_config = load_config()
    init_db()

    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or global_config.get("telegram_bot_token")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID") or global_config.get("telegram_chat_id")

    bot = Bot(token=telegram_bot_token) if telegram_bot_token else None

    if bot and telegram_chat_id:
        startup_msg = "🟢 **Upgraded Structural Trading Agent Initialized & Active.** Hard Breakeven Protection active..."
        await bot.send_message(chat_id=telegram_chat_id, text=startup_msg, parse_mode="Markdown")

    while True:
        current_config = load_config()
        watchlist = current_config.get("watchlist", [])
        scan_interval = current_config.get("scan_interval_minutes", 15)

        # 1. Run Active Trade Evaluator against Supabase
        await evaluate_active_trades(bot, telegram_chat_id, current_config)

        # 2. Run Watchlist Proximity Scanner
        print(f"\n--- Starting Scan Cycle ({len(watchlist)} assets) ---")
        for symbol in watchlist:
            try:
                await analyze_symbol(symbol, bot, telegram_chat_id, current_config)
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")

        gc.collect()
        print(f"Cycle completed. Sleeping for {scan_interval} minutes...")
        await asyncio.sleep(scan_interval * 60)

if __name__ == "__main__":
    asyncio.run(run_scanner())