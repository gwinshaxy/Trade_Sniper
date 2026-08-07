import os
import time
import json
import gc
import asyncio
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from dotenv import load_dotenv
from telegram import Bot
from supabase import create_client, Client

load_dotenv()

# =====================================================================
# GLOBAL CONFIGURATION & SUPABASE SETUP
# =====================================================================
CONFIG_FILE = "config.json"
last_alert_time = {}

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

@st.cache_resource
def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

supabase = get_supabase_client()

@st.cache_resource
def get_telegram_bot():
    if TELEGRAM_BOT_TOKEN:
        try:
            return Bot(token=TELEGRAM_BOT_TOKEN)
        except Exception:
            return None
    return None

telegram_bot = get_telegram_bot()

AVAILABLE_PAIRS = [
    "ADA/USDT", "SOL/USDT", "XRP/USDT", "ONDO/USDT", "PENDLE/USDT", 
    "LINK/USDT", "TIA/USDT", "NEAR/USDT", "BTC/USDT", "ETH/USDT", 
    "AVAX/USDT", "SUI/USDT", "AR/USDT", "FET/USDT", "RENDER/USDT", 
    "TAO/USDT", "SYRUP/USDT", "AAVE/USDT", "UNI/USDT", "APT/USDT", 
    "INJ/USDT", "SEI/USDT"
]

def load_config():
    default_config = {
        "account_balance": 100.0,
        "risk_pct": 1.0,
        "long_risk_multiplier": 1.0,
        "short_risk_multiplier": 1.0,
        "proximity_threshold_pct": 2.0,
        "min_adx": 15.0,
        "min_atr_pct": 0.2,
        "atr_mult_sl": 1.5,
        "min_rr_ratio": 2.0,
        "scan_interval_minutes": 15,
        "alert_cooldown_hours": 4,
        "enable_live_trading": False,
        "watchlist": [
            "ADA/USDT",
            "SOL/USDT",
            "XRP/USDT"
        ],
        "use_mtf_tema_alignment": True,
        "htf_tema_period": 50,
        "htf_alignment_mode": "TEMA Slope",
        "use_adaptive_atr_trail": True,
        "atr_trail_mult": 2.0,
        "use_equity_curve_filter": True,
        "eq_ma_period": 10,
        "reduced_risk_factor": 0.3,
        "asset_overrides": {
            "SOL/USDT": {
                "atr_mult_sl": 2.0,
                "proximity_threshold_pct": 0.5,
                "atr_trail_mult": 2.2
            },
            "XRP/USDT": {
                "proximity_threshold_pct": 1.5,
                "atr_mult_sl": 2.0,
                "min_atr_pct": 0.4,
                "atr_trail_mult": 2.2
            }
        }
    }
    
    if not os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
        except Exception:
            pass
        return default_config

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            existing_config = json.load(f)
            merged = {**default_config, **existing_config}
            if "asset_overrides" in existing_config:
                merged["asset_overrides"] = {**default_config.get("asset_overrides", {}), **existing_config["asset_overrides"]}
            return merged
    except Exception:
        return default_config

def save_config(config_data):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        st.toast("Settings saved!", icon="💾")
    except Exception as e:
        st.error(f"Failed to save settings: {e}")

def get_symbol_config(symbol: str, global_config: dict) -> dict:
    symbol_config = global_config.copy()
    overrides_map = global_config.get("asset_overrides", {})
    if symbol in overrides_map and isinstance(overrides_map[symbol], dict):
        symbol_config.update(overrides_map[symbol])
    return symbol_config

# =====================================================================
# EXCHANGE ENGINE
# =====================================================================
@st.cache_resource
def get_exchange():
    try:
        exchange = ccxt.mexc({
            'enableRateLimit': True,
            'timeout': 20000,
            'apiKey': os.getenv("EXCHANGE_API_KEY", ""),
            'secret': os.getenv("EXCHANGE_API_SECRET", "")
        })
        exchange.load_markets()
        return exchange
    except Exception:
        exchange = ccxt.gateio({
            'enableRateLimit': True,
            'timeout': 20000,
            'apiKey': os.getenv("EXCHANGE_API_KEY", ""),
            'secret': os.getenv("EXCHANGE_API_SECRET", "")
        })
        return exchange

exchange = get_exchange()

def place_maker_limit_order(symbol, side, amount, price):
    params = {'postOnly': True}
    try:
        order = exchange.create_order(
            symbol=symbol,
            type='limit',
            side=side,
            amount=amount,
            price=price,
            params=params
        )
        return order
    except Exception as e:
        print(f"❌ Error placing Maker Limit Order for {symbol}: {e}")
        return None

# =====================================================================
# TECHNICAL INDICATORS & DATA FETCHING
# =====================================================================
def calculate_native_tema(series: pd.Series, length: int = 200) -> pd.Series:
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    res = 3 * ema1 - 3 * ema2 + ema3
    del ema1, ema2, ema3
    return res

@st.cache_data(ttl=120, max_entries=20)
def fetch_market_data(symbol: str, timeframe: str = '1h', limit: int = 350, tema_length: int = 200) -> pd.DataFrame:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        return pd.DataFrame()

    if not ohlcv or len(ohlcv) < 10:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    del ohlcv
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    df['time'] = (df['timestamp'] / 1000).astype(int)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 1. Base TEMA
    if len(df) >= tema_length:
        try:
            df['tema_200'] = ta.tema(df['close'], length=tema_length)
            if df['tema_200'].dropna().empty:
                df['tema_200'] = calculate_native_tema(df['close'], length=tema_length)
        except Exception:
            df['tema_200'] = calculate_native_tema(df['close'], length=tema_length)
        df['tema_200'] = pd.to_numeric(df['tema_200'], errors='coerce').astype(np.float32)

    # 2. Resample HTF TEMA
    try:
        df_indexed = df.set_index('datetime')
        df_4h = df_indexed.resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        
        try:
            df_4h['tema_htf'] = ta.tema(df_4h['close'], length=tema_length)
            if df_4h['tema_htf'].dropna().empty:
                df_4h['tema_htf'] = calculate_native_tema(df_4h['close'], length=tema_length)
        except Exception:
            df_4h['tema_htf'] = calculate_native_tema(df_4h['close'], length=tema_length)
            
        df = pd.merge_asof(
            df.sort_values('datetime'), 
            df_4h[['datetime', 'tema_htf']].sort_values('datetime'), 
            on='datetime', 
            direction='backward'
        )
        df['tema_htf'] = df['tema_htf'].ffill()
        df['tema_htf_slope'] = df['tema_htf'] - df['tema_htf'].shift(4)
    except Exception:
        df['tema_htf'] = df.get('tema_200', df['close'])
        df['tema_htf_slope'] = 0.0

    # 3. ADX Calculation
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

    # 4. ATR & ATR % Calculation
    try:
        raw_atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'] = pd.to_numeric(raw_atr, errors='coerce').astype(np.float32)
        df['atr_pct'] = ((df['atr'] / df['close']) * 100.0).astype(np.float32)
        del raw_atr
    except Exception:
        df['atr'] = np.float32(0.0)
        df['atr_pct'] = np.float32(0.0)

    # 5. VWAP POC Proxy
    try:
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        df['vwap_poc'] = (df['volume'] * tp).rolling(50).sum() / df['volume'].rolling(50).sum()
        df['vwap_poc'] = pd.to_numeric(df['vwap_poc'], errors='coerce').astype(np.float32)
    except Exception:
        df['vwap_poc'] = df['close']

    return df

def calculate_volume_profile(df, num_bins=30):
    if df.empty or 'volume' not in df.columns:
        return None, [], []

    price_min = float(df['low'].min())
    price_max = float(df['high'].max())

    if price_min == price_max or pd.isna(price_min) or pd.isna(price_max):
        return None, [], []

    bins = np.linspace(price_min, price_max, num_bins)
    df_temp = df.copy()
    df_temp['bin'] = pd.cut(df_temp['close'], bins=bins)

    volume_profile = df_temp.groupby('bin', observed=False)['volume'].sum().reset_index()
    volume_profile['price_mid'] = volume_profile['bin'].apply(lambda x: float(x.mid) if pd.notnull(x) else 0.0)

    if volume_profile['volume'].sum() == 0:
        return None, [], []

    max_vol = float(volume_profile['volume'].max())
    poc_row = volume_profile.loc[volume_profile['volume'].idxmax()]
    poc_price = float(poc_row['price_mid'])

    hvn_nodes = volume_profile.sort_values(by='volume', ascending=False).head(3)
    hvn_list = [float(x) for x in hvn_nodes['price_mid'].tolist()]
    
    vp_bins = []
    for _, r in volume_profile.iterrows():
        if r['bin'] is not None and pd.notnull(r['bin']):
            vp_bins.append({
                'price_low': float(r['bin'].left),
                'price_high': float(r['bin'].right),
                'price_mid': float(r['price_mid']),
                'volume': float(r['volume']),
                'vol_ratio': float(r['volume'] / max_vol) if max_vol > 0 else 0.0
            })
            
    del df_temp, volume_profile, hvn_nodes
    return poc_price, hvn_list, vp_bins

# =====================================================================
# RISK & POSITION MANAGEMENT ENGINE
# =====================================================================
def calculate_effective_risk(base_risk_pct, direction, config):
    dir_mult = float(config.get("long_risk_multiplier", 1.0)) if direction == "LONG" else float(config.get("short_risk_multiplier", 1.0))
    effective_risk = base_risk_pct * dir_mult

    if config.get("use_equity_curve_filter", False) and supabase:
        try:
            res = supabase.table("trade_journal").select("realized_pnl_usd").in_("status", ["CLOSED_TP2", "STOPPED_OUT"]).order("id", desc=True).limit(50).execute()
            closed_trades = res.data
            eq_period = int(config.get("eq_ma_period", 10))
            if closed_trades and len(closed_trades) >= eq_period:
                initial_balance = float(config.get("account_balance", 100.0))
                pnl_list = [float(t.get("realized_pnl_usd", 0.0)) for t in reversed(closed_trades)]
                equity_series = np.cumsum([initial_balance] + pnl_list)
                
                recent_ma = np.mean(equity_series[-eq_period:])
                current_eq = equity_series[-1]
                
                if current_eq < recent_ma:
                    reduced_factor = float(config.get("reduced_risk_factor", 0.3))
                    effective_risk *= reduced_factor
        except Exception:
            pass

    return effective_risk

def calculate_fixed_risk_position(account_balance, entry_price, stop_loss_price, risk_pct=1.0):
    risk_amount = float(account_balance) * (float(risk_pct) / 100.0)
    price_risk_per_unit = abs(float(entry_price) - float(stop_loss_price))

    if price_risk_per_unit == 0:
        return 0.0, 0.0, 0.0

    units = risk_amount / price_risk_per_unit
    position_size_usdt = units * float(entry_price)
    return float(units), float(position_size_usdt), float(risk_amount)

def log_trade_signal(symbol, reasons, entry, sl, tp1, tp2, position_usdt, risk_usd, order_id=None):
    if not supabase:
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
        "status": "OPEN",
        "order_id": str(order_id) if order_id else None
    }

    try:
        supabase.table("trade_journal").insert(data).execute()
    except Exception as e:
        print(f"Error logging trade signal: {e}")

# =====================================================================
# ACTIVE TRADE EVALUATION ENGINE
# =====================================================================
async def evaluate_active_trades(bot, chat_id, global_config):
    if not supabase:
        return

    try:
        response = supabase.table("trade_journal").select("*").in_("status", ["OPEN", "TP1_HIT"]).execute()
        open_trades = response.data
    except Exception:
        return

    if not open_trades:
        return

    for trade in open_trades:
        symbol = str(trade.get('symbol', 'UNKNOWN'))
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
            candles_needed = min(max(50, int(hours_elapsed * 4) + 10), 350)

            df_candles = fetch_market_data(symbol, timeframe='15m', limit=candles_needed)
            if df_candles is None or df_candles.empty:
                continue

            latest_close = float(df_candles['close'].iloc[-1])
            latest_low = float(df_candles['low'].min())
            latest_high = float(df_candles['high'].max())
            latest_atr = float(df_candles['atr'].iloc[-1]) if 'atr' in df_candles else 0.0
            close_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            is_long = sl < entry

            # Dynamic ATR Trailing Stop
            if config.get("use_adaptive_atr_trail", True) and latest_atr > 0:
                trail_mult = float(config.get("atr_trail_mult", 2.0))
                if is_long:
                    new_trail_sl = latest_close - (latest_atr * trail_mult)
                    if current_status == "TP1_HIT":
                        new_trail_sl = max(new_trail_sl, entry)
                    if new_trail_sl > sl:
                        sl = new_trail_sl
                        supabase.table("trade_journal").update({"stop_loss": float(sl)}).eq("id", trade_id).execute()
                else:
                    new_trail_sl = latest_close + (latest_atr * trail_mult)
                    if current_status == "TP1_HIT":
                        new_trail_sl = min(new_trail_sl, entry)
                    if new_trail_sl < sl:
                        sl = new_trail_sl
                        supabase.table("trade_journal").update({"stop_loss": float(sl)}).eq("id", trade_id).execute()

            # 1. Check Stop Loss
            sl_hit = (latest_low <= sl) if is_long else (latest_high >= sl)
            if sl_hit:
                pnl_usd = 0.0 if current_status == "TP1_HIT" else -risk_usd
                realized_r = 0.0 if current_status == "TP1_HIT" else -1.0

                update_data = {
                    "status": "STOPPED_OUT",
                    "exit_price": float(sl),
                    "closed_timestamp": str(close_time),
                    "realized_pnl_usd": float(pnl_usd),
                    "realized_r": float(realized_r)
                }
                supabase.table("trade_journal").update(update_data).eq("id", trade_id).execute()

                if bot and chat_id:
                    status_title = "🛡️ **TRADE CLOSED AT BREAKEVEN: " if current_status == "TP1_HIT" else "🛑 **TRADE STOPPED OUT: "
                    msg = (
                        f"{status_title}{symbol}**\n\n"
                        f"• Exit Price: `${sl:.4f}`\n"
                        f"• Realized PnL: `${pnl_usd:.2f}` ({realized_r:.2f}R)\n"
                        f"• Timestamp: `{close_time}`"
                    )
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

            # 2. Check TP2
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
                        f"🎯 **TARGET 2 HIT: {symbol}** 🎯\n\n"
                        f"• Target Price: `${tp2:.4f}`\n"
                        f"• Realized PnL: `+${pnl_usd:.2f}` (+{r_multiple:.2f}R)\n"
                        f"• Timestamp: `{close_time}`"
                    )
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

            # 3. Check TP1 (Hard Breakeven)
            tp1_hit = (latest_high >= tp1) if is_long else (latest_low <= tp1)
            if tp1_hit and current_status == 'OPEN':
                r_multiple = abs(tp1 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
                sl = entry

                supabase.table("trade_journal").update({
                    "status": "TP1_HIT",
                    "stop_loss": float(sl)
                }).eq("id", trade_id).execute()

                if bot and chat_id:
                    msg = (
                        f"✅ **TARGET 1 HIT: {symbol}** ✅\n\n"
                        f"• Milestone Price: `${tp1:.4f}`\n"
                        f"• Gain: `+{r_multiple:.2f}R`\n"
                        f"• Breakeven Set: SL moved to `${sl:.4f}`"
                    )
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

            del df_candles
        except Exception as e:
            print(f"Error evaluating trade ID {trade.get('id')}: {e}")

# =====================================================================
# LIVE MARKET SCANNER & SIGNAL GENERATOR
# =====================================================================
async def analyze_symbol(symbol, bot, chat_id, global_config):
    current_time = time.time()
    config = get_symbol_config(symbol, global_config)

    account_balance = float(config.get("account_balance", 100.0))
    base_risk_pct = float(config.get("risk_pct", 1.0))
    min_rr_threshold = float(config.get("min_rr_ratio", 2.0))
    proximity_threshold = float(config.get("proximity_threshold_pct", 2.0))
    cooldown_hours = float(config.get("alert_cooldown_hours", 4))
    min_adx = float(config.get("min_adx", 15.0))
    min_atr_pct = float(config.get("min_atr_pct", 0.2))
    atr_mult_sl = float(config.get("atr_mult_sl", 1.5))
    enable_live_trading = config.get("enable_live_trading", False)

    if symbol in last_alert_time:
        elapsed_hours = (current_time - last_alert_time[symbol]) / 3600
        if elapsed_hours < cooldown_hours:
            return

    df_1h = fetch_market_data(symbol, timeframe='1h', limit=350)
    htf_lookback = int(config.get("htf_tema_period", 50))
    df_4h = fetch_market_data(symbol, timeframe='4h', limit=350, tema_length=htf_lookback)

    if df_1h is None or df_4h is None or df_1h.empty or df_4h.empty:
        return

    current_price = float(df_1h['close'].iloc[-1])
    val_1h = float(df_1h['tema_200'].dropna().iloc[-1]) if 'tema_200' in df_1h and not df_1h['tema_200'].dropna().empty else None
    val_4h = float(df_4h['tema_htf'].dropna().iloc[-1]) if 'tema_htf' in df_4h and not df_4h['tema_htf'].dropna().empty else None
    vwap_poc_price = float(df_1h['vwap_poc'].dropna().iloc[-1]) if 'vwap_poc' in df_1h and not df_1h['vwap_poc'].dropna().empty else None
    poc_price, hvn_prices, _ = calculate_volume_profile(df_1h, num_bins=30)

    triggered_reasons = []

    if val_1h is not None:
        dist_1h = abs(current_price - val_1h) / val_1h * 100
        if dist_1h <= proximity_threshold:
            triggered_reasons.append(f"Near 1H 200 TEMA ({dist_1h:.2f}% away)")

    if val_4h is not None:
        dist_4h = abs(current_price - val_4h) / val_4h * 100
        if dist_4h <= proximity_threshold:
            triggered_reasons.append(f"Near 4H HTF TEMA ({dist_4h:.2f}% away)")

    if vwap_poc_price is not None:
        dist_vwap = abs(current_price - vwap_poc_price) / vwap_poc_price * 100
        if dist_vwap <= proximity_threshold:
            triggered_reasons.append(f"Near VWAP/POC Proxy ({dist_vwap:.2f}% away)")

    if poc_price is not None:
        dist_poc = abs(current_price - poc_price) / poc_price * 100
        if dist_poc <= proximity_threshold:
            triggered_reasons.append(f"Near Volume Profile POC ({dist_poc:.2f}% away)")

    if not triggered_reasons:
        del df_1h, df_4h
        return

    # Multi-Timeframe Alignment Filter
    if config.get("use_mtf_tema_alignment", True) and val_1h is not None and val_4h is not None:
        alignment_mode = config.get("htf_alignment_mode", "TEMA Slope")
        base_long = current_price >= val_1h
        base_short = current_price < val_1h

        if alignment_mode in ["Price Level", "Price Level (Price > HTF TEMA)"]:
            htf_long = current_price >= val_4h
            htf_short = current_price < val_4h
        else:
            prev_val_4h = float(df_4h['tema_htf'].dropna().iloc[-2]) if len(df_4h['tema_htf'].dropna()) > 1 else val_4h
            htf_long = val_4h > prev_val_4h
            htf_short = val_4h < prev_val_4h

        if (base_long and not htf_long) or (base_short and not htf_short):
            del df_1h, df_4h
            return
        else:
            triggered_reasons.append(f"Aligned with 4H Trend ({alignment_mode})")

    current_adx = float(df_1h['adx'].dropna().iloc[-1]) if 'adx' in df_1h else 0.0
    current_atr_pct = float(df_1h['atr_pct'].dropna().iloc[-1]) if 'atr_pct' in df_1h else 0.0

    if current_adx < min_adx or current_atr_pct < min_atr_pct:
        del df_1h, df_4h
        return

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

    risk_per_unit = abs(entry_price - stop_loss)
    reward_per_unit = abs(tp1 - entry_price)
    rr_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0

    if rr_ratio < min_rr_threshold:
        del df_1h, df_4h
        return

    effective_risk_pct = calculate_effective_risk(base_risk_pct, direction, config)
    units, position_usdt, risk_usd = calculate_fixed_risk_position(
        account_balance, entry_price, stop_loss, risk_pct=effective_risk_pct
    )

    order_id = None
    if enable_live_trading:
        order_side = "buy" if direction == "LONG" else "sell"
        order_res = place_maker_limit_order(symbol, side=order_side, amount=units, price=entry_price)
        if order_res:
            order_id = order_res.get("id")

    log_trade_signal(
        symbol, triggered_reasons, entry_price, 
        stop_loss, tp1, tp2, position_usdt, risk_usd, order_id=order_id
    )

    last_alert_time[symbol] = current_time

    reasons_text = "\n".join([f"• {r}" for r in triggered_reasons])
    message = (
        f"🎯 **SIGNAL DETECTED: {symbol}** 🎯\n\n"
        f"**Triggers:**\n{reasons_text}\n\n"
        f"**Price:** `${current_price:.4f}`\n\n"
        f"📈 **Technical Parameters:**\n"
        f"• 1H TEMA: `${val_1h:.4f}`\n"
        f"• 4H TEMA: `${val_4h:.4f}`\n"
        f"• ADX: `{current_adx:.2f}` | ATR%: `{current_atr_pct:.2f}%`\n\n"
        f"🎯 **Trade Execution ({effective_risk_pct:.2f}% Risk):**\n"
        f"• Direction: `{direction}`\n"
        f"• Entry: `${entry_price:.4f}`\n"
        f"• Stop Loss: `${stop_loss:.4f}` (${risk_usd:.2f})\n"
        f"• TP1: `${tp1:.4f}` | TP2: `${tp2:.4f}`\n"
        f"• R/R: `{rr_ratio:.2f}R`\n"
        f"• Order: `{'DISPATCHED (ID: ' + str(order_id) + ')' if order_id else 'SIMULATED'}`"
    )

    if bot and chat_id:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")

    del df_1h, df_4h

async def run_scanner_loop(config):
    bot = telegram_bot
    chat_id = TELEGRAM_CHAT_ID
    watchlist = config.get("watchlist", ["SOL/USDT"])
    
    await evaluate_active_trades(bot, chat_id, config)
    for symbol in watchlist:
        await analyze_symbol(symbol, bot, chat_id, config)

# =====================================================================
# HISTORICAL BACKTESTING ENGINE
# =====================================================================
def run_backtest_upgraded(df: pd.DataFrame, params: dict):
    if df.empty or len(df) < 200:
        return pd.DataFrame(), pd.Series(), {}

    initial_balance = params['initial_balance']
    base_risk_pct = params['risk_pct'] / 100.0
    target_rr = params['target_rr']
    min_adx = params['min_adx']
    min_atr_pct = params['min_atr_pct']
    proximity_pct = params['proximity_pct']
    atr_mult_sl = params.get('atr_mult_sl', 1.5)
    
    use_mtf = params.get('use_mtf', True)
    htf_mode = params.get('htf_mode', 'TEMA Slope')
    use_atr_trail = params.get('use_atr_trail', True)
    atr_trail_mult = params.get('atr_trail_mult', 2.0)
    use_equity_filter = params.get('use_equity_filter', True)
    eq_ma_period = params.get('eq_ma_period', 10)
    reduced_risk_factor = params.get('reduced_risk_factor', 0.3)

    balance = float(initial_balance)
    equity_curve = [balance]
    trades = []
    active_trade = None

    for i in range(200, len(df)):
        row = df.iloc[i]
        price = float(row['close'])
        high = float(row['high'])
        low = float(row['low'])
        tema_1h = float(row['tema_200']) if 'tema_200' in row and pd.notnull(row['tema_200']) else 0.0
        tema_htf = float(row['tema_htf']) if 'tema_htf' in row and pd.notnull(row['tema_htf']) else tema_1h
        htf_slope = float(row.get('tema_htf_slope', 0.0))
        adx = float(row.get('adx', 0.0))
        atr_pct = float(row.get('atr_pct', 0.0))
        atr_val = float(row.get('atr', 0.0))
        timestamp = datetime.fromtimestamp(int(row['time'])).strftime('%Y-%m-%d %H:%M')

        # 1. Active Trade Check
        if active_trade is not None:
            side = active_trade["side"]
            sl = active_trade["sl"]
            tp = active_trade["tp"]
            
            hit_sl = low <= sl if side == "LONG" else high >= sl
            hit_tp = high >= tp if side == "LONG" else low <= tp

            if hit_sl or hit_tp:
                exit_price = sl if hit_sl else tp
                pnl_r = -1.0 if hit_sl else target_rr
                pnl_usd = -active_trade["risk_usd"] if hit_sl else (active_trade["risk_usd"] * target_rr)
                balance += pnl_usd

                trades.append({
                    "Entry Time": active_trade["entry_time"],
                    "Exit Time": timestamp,
                    "Side": side,
                    "Entry ($)": round(active_trade["entry"], 4),
                    "Exit ($)": round(exit_price, 4),
                    "Result": "WIN" if pnl_usd > 0 else "LOSS",
                    "PnL ($)": round(pnl_usd, 2),
                    "Realized R": f"{pnl_r:.2f}R",
                    "Balance ($)": round(balance, 2)
                })
                active_trade = None

            elif use_atr_trail and atr_val > 0:
                if side == "LONG":
                    new_sl = price - (atr_val * atr_trail_mult)
                    active_trade["sl"] = max(active_trade["sl"], new_sl)
                else:
                    new_sl = price + (atr_val * atr_trail_mult)
                    active_trade["sl"] = min(active_trade["sl"], new_sl)

        # 2. Signal Check
        if active_trade is None and tema_1h > 0:
            if adx >= min_adx and atr_pct >= min_atr_pct:
                prox = abs(price - tema_1h) / tema_1h * 100
                if prox <= proximity_pct:
                    base_long = price >= tema_1h
                    base_short = price < tema_1h
                    direction = None

                    if use_mtf:
                        if htf_mode == "Price Level":
                            htf_long = price >= tema_htf
                            htf_short = price < tema_htf
                        else:
                            htf_long = htf_slope > 0
                            htf_short = htf_slope < 0

                        if base_long and htf_long:
                            direction = "LONG"
                        elif base_short and htf_short:
                            direction = "SHORT"
                    else:
                        direction = "LONG" if base_long else "SHORT"

                    if direction:
                        effective_risk = base_risk_pct
                        if use_equity_filter and len(equity_curve) >= eq_ma_period:
                            recent_eq_ma = np.mean(equity_curve[-eq_ma_period:])
                            if balance < recent_eq_ma:
                                effective_risk *= reduced_risk_factor

                        risk_usd = balance * effective_risk
                        
                        if direction == "LONG":
                            sl = price - (atr_mult_sl * atr_val)
                            tp = price + (atr_mult_sl * atr_val * target_rr)
                        else:
                            sl = price + (atr_mult_sl * atr_val)
                            tp = price - (atr_mult_sl * atr_val * target_rr)

                        active_trade = {
                            "entry_time": timestamp,
                            "side": direction,
                            "entry": price,
                            "sl": sl,
                            "tp": tp,
                            "risk_usd": risk_usd
                        }

        equity_curve.append(balance)

    trades_df = pd.DataFrame(trades)
    wins = len(trades_df[trades_df['Result'] == 'WIN']) if not trades_df.empty else 0
    total_trades = len(trades_df)
    
    metrics = {
        "Starting Balance": f"${initial_balance:.2f}",
        "Final Balance": f"${balance:.2f}",
        "Total Trades": total_trades,
        "Win Rate": f"{(wins / total_trades * 100):.1f}%" if total_trades > 0 else "0.0%",
        "Net Return": f"{((balance - initial_balance) / initial_balance * 100):.2f}%"
    }

    gc.collect()
    return trades_df, pd.Series(equity_curve), metrics

# =====================================================================
# TRADINGVIEW LIGHTWEIGHT CHARTS RENDERER
# =====================================================================
def render_tradingview_chart(df: pd.DataFrame, symbol: str, df_journal: pd.DataFrame):
    candles_records = df[['time', 'open', 'high', 'low', 'close']].to_dict(orient='records')
    
    tema_records = []
    if 'tema_200' in df.columns:
        valid_tema = df[['time', 'tema_200']].dropna().copy()
        for _, r in valid_tema.iterrows():
            val = float(r['tema_200'])
            if np.isfinite(val):
                tema_records.append({'time': int(r['time']), 'value': round(val, 6)})
        del valid_tema

    _, _, vp_bins = calculate_volume_profile(df, num_bins=24)
    max_vol = max([b['volume'] for b in vp_bins]) if vp_bins else 1.0
    avg_vol = np.mean([b['volume'] for b in vp_bins]) if vp_bins else 0.0

    price_lines_js = ""
    if df_journal is not None and not df_journal.empty and 'symbol' in df_journal.columns:
        symbol_trades = df_journal[
            (df_journal['symbol'] == symbol) & 
            (df_journal['status'].isin(['OPEN', 'TP1_HIT', 'PENDING']))
        ]
        
        entries, stop_losses, tp1_levels, tp2_levels = {}, {}, {}, {}

        for _, trade in symbol_trades.iterrows():
            t_id = str(trade.get('id', ''))
            
            p_entry = trade.get('entry_price')
            if pd.notnull(p_entry) and float(p_entry) > 0:
                entries.setdefault(round(float(p_entry), 2), []).append((float(p_entry), t_id))

            p_sl = trade.get('stop_loss')
            if pd.notnull(p_sl) and float(p_sl) > 0:
                stop_losses.setdefault(round(float(p_sl), 2), []).append((float(p_sl), t_id))

            p_tp1 = trade.get('take_profit_1')
            if pd.notnull(p_tp1) and float(p_tp1) > 0:
                tp1_levels.setdefault(round(float(p_tp1), 2), []).append((float(p_tp1), t_id))

            p_tp2 = trade.get('take_profit_2')
            if pd.notnull(p_tp2) and float(p_tp2) > 0:
                tp2_levels.setdefault(round(float(p_tp2), 2), []).append((float(p_tp2), t_id))

        def build_line(data_dict, color, style, prefix):
            js_out = ""
            for _, item_list in data_dict.items():
                avg_price = sum(x[0] for x in item_list) / len(item_list)
                count = len(item_list)
                label = f"{prefix} (#{item_list[0][1]})" if count == 1 else f"{prefix} ({count} Trades)"
                js_out += f"""
                candlestickSeries.createPriceLine({{
                    price: {avg_price:.4f},
                    color: '{color}',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.{style},
                    axisLabelVisible: true,
                    title: '{label}',
                }});
                """
            return js_out

        price_lines_js += build_line(entries, '#2962FF', 'Dashed', 'ENTRY')
        price_lines_js += build_line(stop_losses, '#FF5252', 'Solid', 'SL')
        price_lines_js += build_line(tp1_levels, '#00E676', 'Dotted', 'TP1')
        price_lines_js += build_line(tp2_levels, '#00B0FF', 'Dotted', 'TP2')

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background-color: #131722; font-family: monospace; overflow: hidden; }}
            #chart-container {{ position: relative; width: 100%; height: 550px; background-color: #131722; }}
            #chart-container.fullscreen {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 99999; }}
            #chart {{ width: 100%; height: 100%; }}
            #vp-canvas {{ position: absolute; top: 0; left: 0; pointer-events: none; z-index: 2; }}
            #fullscreen-btn {{
                position: absolute; top: 10px; right: 80px; z-index: 100;
                background-color: rgba(42, 46, 57, 0.85); color: #d1d4dc;
                border: 1px solid rgba(197, 203, 206, 0.4); border-radius: 4px;
                padding: 4px 10px; font-size: 11px; cursor: pointer; transition: all 0.2s;
            }}
            #fullscreen-btn:hover {{ background-color: #2962FF; color: #ffffff; border-color: #2962FF; }}
        </style>
    </head>
    <body>
        <div id="chart-container">
            <button id="fullscreen-btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
            <div id="chart"></div>
            <canvas id="vp-canvas"></canvas>
        </div>
        <script>
            try {{
                const chartContainer = document.getElementById('chart-container');
                const chartElement = document.getElementById('chart');
                const vpCanvas = document.getElementById('vp-canvas');
                const fsBtn = document.getElementById('fullscreen-btn');
                const ctx = vpCanvas.getContext('2d');

                const chart = LightweightCharts.createChart(chartElement, {{
                    width: chartContainer.clientWidth || 800,
                    height: chartContainer.clientHeight || 550,
                    layout: {{
                        background: {{ type: 'solid', color: '#131722' }},
                        textColor: '#d1d4dc',
                    }},
                    grid: {{
                        vertLines: {{ color: 'rgba(42, 46, 57, 0.5)' }},
                        horzLines: {{ color: 'rgba(42, 46, 57, 0.5)' }},
                    }},
                    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                    rightPriceScale: {{ borderColor: 'rgba(197, 203, 206, 0.8)' }},
                    timeScale: {{
                        borderColor: 'rgba(197, 203, 206, 0.8)',
                        timeVisible: true,
                        secondsVisible: false,
                    }},
                }});

                const candlestickSeries = chart.addCandlestickSeries({{
                    upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
                    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
                }});
                const candleData = {json.dumps(candles_records)};
                candlestickSeries.setData(candleData);

                const temaData = {json.dumps(tema_records)};
                if (temaData && temaData.length > 0) {{
                    const temaSeries = chart.addLineSeries({{
                        color: '#FF9800',
                        lineWidth: 2,
                        crosshairMarkerVisible: true,
                        priceLineVisible: false,
                        title: '200 TEMA',
                    }});
                    temaSeries.setData(temaData);
                }}

                {price_lines_js}

                chart.timeScale().fitContent();

                const vpBins = {json.dumps(vp_bins)};
                const maxVol = {max_vol};

                function drawVolumeProfile() {{
                    vpCanvas.width = chartContainer.clientWidth;
                    vpCanvas.height = chartContainer.clientHeight;
                    ctx.clearRect(0, 0, vpCanvas.width, vpCanvas.height);

                    if (!vpBins || vpBins.length === 0 || maxVol <= 0) return;

                    const chartWidth = vpCanvas.width - 65;
                    const maxBarWidth = chartWidth * 0.20;

                    ctx.fillStyle = 'rgba(41, 98, 255, 0.25)';
                    ctx.strokeStyle = 'rgba(41, 98, 255, 0.5)';

                    vpBins.forEach(bin => {{
                        const yTop = candlestickSeries.priceToCoordinate(bin.price_high);
                        const yBottom = candlestickSeries.priceToCoordinate(bin.price_low);
                        
                        if (yTop !== null && yBottom !== null && !isNaN(yTop) && !isNaN(yBottom)) {{
                            const barHeight = Math.max(Math.abs(yBottom - yTop) - 1, 1);
                            const barWidth = (bin.volume / maxVol) * maxBarWidth;
                            const yPos = Math.min(yTop, yBottom);

                            ctx.fillRect(0, yPos, barWidth, barHeight);
                            ctx.strokeRect(0, yPos, barWidth, barHeight);
                        }}
                    }});
                }}

                window.toggleFullscreen = function() {{
                    const isFullscreen = chartContainer.classList.toggle('fullscreen');
                    if (isFullscreen) {{
                        fsBtn.innerText = "✕ Exit Fullscreen";
                        chart.applyOptions({{ width: window.innerWidth, height: window.innerHeight }});
                    }} else {{
                        fsBtn.innerText = "⛶ Fullscreen";
                        chart.applyOptions({{ width: chartContainer.parentElement.clientWidth || 800, height: 550 }});
                    }}
                    drawVolumeProfile();
                }};

                chart.timeScale().subscribeVisibleLogicalRangeChange(drawVolumeProfile);
                chart.timeScale().subscribeVisibleTimeRangeChange(drawVolumeProfile);
                
                window.addEventListener('resize', () => {{
                    if (chartContainer.classList.contains('fullscreen')) {{
                        chart.applyOptions({{ width: window.innerWidth, height: window.innerHeight }});
                    }} else {{
                        chart.applyOptions({{ width: chartContainer.clientWidth, height: 550 }});
                    }}
                    drawVolumeProfile();
                }});

                setTimeout(drawVolumeProfile, 150);

            }} catch (err) {{
                console.error(err);
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=570)
    del candles_records, tema_records, vp_bins
    gc.collect()

# =====================================================================
# DATABASE MANAGEMENT HELPERS
# =====================================================================
def load_trade_journal() -> pd.DataFrame:
    if not supabase:
        return pd.DataFrame()
    try:
        response = supabase.table("trade_journal").select("*").order("id", desc=True).execute()
        data = response.data
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        numeric_cols = [
            'entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2', 
            'position_usdt', 'max_risk_usd', 'exit_price', 
            'realized_pnl_usd', 'realized_r'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
        return df
    except Exception:
        return pd.DataFrame()

def clear_remote_journal():
    if supabase:
        try:
            supabase.table("trade_journal").delete().gt("id", 0).execute()
            st.toast("🧹 Trade journal database cleared!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear database: {e}")

# =====================================================================
# STREAMLIT UI LAYOUT & DASHBOARD
# =====================================================================
st.set_page_config(
    page_title="Trade Sniper Dashboard Pro",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Trade Sniper Dashboard Pro")
st.caption("Automated Multi-Timeframe TEMA Alignment, Maker Post-Only Orders & Dynamic Risk Scaling")

config = load_config()
df_journal = load_trade_journal()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Bot Parameters")
    
    with st.form("config_form"):
        st.subheader("⚡ Execution Mode")
        trading_mode = st.radio(
            "Trading Mode",
            ["Paper Trading (Simulation)", "Live Execution (Maker Post-Only)"],
            index=1 if config.get("enable_live_trading", False) else 0
        )
        enable_live = (trading_mode == "Live Execution (Maker Post-Only)")

        st.markdown("---")
        account_balance = st.number_input(
            "Account Balance ($)", 
            value=float(config.get("account_balance", 100.0)), 
            step=50.0
        )
        risk_pct = st.slider(
            "Base Risk % per Trade", 
            0.1, 5.0, 
            float(config.get("risk_pct", 1.0)), 
            step=0.1
        )
        long_risk_mult = st.number_input("Long Risk Multiplier", value=float(config.get("long_risk_multiplier", 1.0)), step=0.1)
        short_risk_mult = st.number_input("Short Risk Multiplier", value=float(config.get("short_risk_multiplier", 1.0)), step=0.1)
        
        proximity_thresh = st.slider(
            "Proximity Threshold %", 
            0.5, 5.0, 
            float(config.get("proximity_threshold_pct", 2.0)), 
            step=0.1
        )
        min_adx = st.slider(
            "ADX Cutoff", 
            10.0, 50.0, 
            float(config.get("min_adx", 15.0)), 
            step=1.0
        )
        min_atr = st.number_input(
            "Min ATR %", 
            value=float(config.get("min_atr_pct", 0.2)), 
            step=0.05
        )
        atr_mult_sl = st.slider(
            "ATR Multiplier (SL)", 
            0.5, 4.0, 
            float(config.get("atr_mult_sl", 1.5)), 
            step=0.1
        )
        min_rr_ratio = st.slider(
            "Minimum R/R Ratio", 
            1.0, 5.0, 
            float(config.get("min_rr_ratio", 2.0)), 
            step=0.1
        )

        st.markdown("---")
        st.subheader("🚀 Upgrades & Execution Panel")
        use_mtf = st.checkbox("Enable 1H + HTF Trend Alignment", value=config.get("use_mtf_tema_alignment", True))
        htf_period = st.slider("HTF TEMA Period", 10, 200, int(config.get("htf_tema_period", 50)), step=10)
        htf_mode = st.radio("HTF Alignment Mode", ["Price Level", "TEMA Slope"], index=1 if config.get("htf_alignment_mode", "TEMA Slope") == "TEMA Slope" else 0)

        st.markdown("---")
        use_atr_trail = st.checkbox("Enable Dynamic ATR Trailing Stop", value=config.get("use_adaptive_atr_trail", True))
        atr_trail_mult = st.slider("ATR Trailing Multiplier", 1.0, 4.0, float(config.get("atr_trail_mult", 2.0)), step=0.1)

        st.markdown("---")
        use_equity_filter = st.checkbox("Enable Equity Curve Drawdown Filter", value=config.get("use_equity_curve_filter", True))
        eq_ma = st.number_input("Equity MA Period", value=int(config.get("eq_ma_period", 10)), step=1)
        reduced_risk = st.slider("Drawdown Risk Factor", 0.1, 0.9, float(config.get("reduced_risk_factor", 0.3)), step=0.05)

        st.markdown("---")
        alert_cooldown = st.number_input("Alert Cooldown (hours)", value=int(config.get("alert_cooldown_hours", 4)), min_value=1, max_value=48, step=1)
        scan_interval = st.number_input("Scan Interval (mins)", value=int(config.get("scan_interval_minutes", 15)), step=1)
        
        st.markdown("---")
        st.subheader("📊 Watchlist Management")
        new_asset = st.text_input("➕ Add Custom Symbol (e.g., BTC/USDT):").strip().upper()
        
        current_watchlist = config.get("watchlist", ["ADA/USDT", "SOL/USDT", "XRP/USDT"])
        all_options = list(dict.fromkeys(AVAILABLE_PAIRS + current_watchlist + ([new_asset] if new_asset else [])))
        
        default_selected = current_watchlist.copy()
        if new_asset and new_asset not in default_selected:
            default_selected.append(new_asset)
            
        selected_watchlist = st.multiselect("Active Watchlist Assets:", options=all_options, default=default_selected)
        
        submitted = st.form_submit_button("Save Configuration")
        if submitted:
            updated_config = {
                "account_balance": account_balance,
                "risk_pct": risk_pct,
                "long_risk_multiplier": long_risk_mult,
                "short_risk_multiplier": short_risk_mult,
                "proximity_threshold_pct": proximity_thresh,
                "min_adx": min_adx,
                "min_atr_pct": min_atr,
                "atr_mult_sl": atr_mult_sl,
                "min_rr_ratio": min_rr_ratio,
                "enable_live_trading": enable_live,
                "use_mtf_tema_alignment": use_mtf,
                "htf_tema_period": htf_period,
                "htf_alignment_mode": htf_mode,
                "use_adaptive_atr_trail": use_atr_trail,
                "atr_trail_mult": atr_trail_mult,
                "use_equity_curve_filter": use_equity_filter,
                "eq_ma_period": eq_ma,
                "reduced_risk_factor": reduced_risk,
                "alert_cooldown_hours": alert_cooldown,
                "scan_interval_minutes": scan_interval,
                "watchlist": selected_watchlist,
                "asset_overrides": config.get("asset_overrides", {})
            }
            save_config(updated_config)
            st.cache_data.clear()
            st.rerun()

    if st.button("⚡ Trigger Manual Scanner Run", use_container_width=True):
        with st.spinner("Scanning markets & evaluating active positions..."):
            asyncio.run(run_scanner_loop(config))
        st.success("Scan sequence complete!")
        st.rerun()

# --- TOP METRICS ROW ---
m1, m2, m3, m4, m5 = st.columns(5)
if not df_journal.empty:
    total_trades = len(df_journal)
    open_trades = len(df_journal[df_journal['status'].isin(['OPEN', 'TP1_HIT'])])
    closed_trades = df_journal[df_journal['status'].isin(['CLOSED_TP2', 'STOPPED_OUT'])]
    
    net_pnl = closed_trades['realized_pnl_usd'].sum() if 'realized_pnl_usd' in closed_trades.columns else 0.0
    total_r = closed_trades['realized_r'].sum() if 'realized_r' in closed_trades.columns else 0.0
    wins = len(closed_trades[closed_trades['realized_pnl_usd'] > 0]) if not closed_trades.empty else 0
    win_rate = (wins / len(closed_trades) * 100) if len(closed_trades) > 0 else 0.0

    m1.metric("Total Signals", total_trades)
    m2.metric("Active Trades", open_trades)
    m3.metric("Net Realized PnL", f"${net_pnl:.2f}", delta=f"{total_r:.2f}R")
    m4.metric("Win Rate", f"{win_rate:.1f}%")
    m5.metric("Execution Mode", "LIVE (MAKER)" if config.get("enable_live_trading") else "SIMULATION")
else:
    m1.metric("Total Signals", "0")
    m2.metric("Active Trades", "0")
    m3.metric("Net Realized PnL", "$0.00")
    m4.metric("Win Rate", "0.0%")
    m5.metric("Execution Mode", "LIVE (MAKER)" if config.get("enable_live_trading") else "SIMULATION")

st.divider()

# --- DASHBOARD TABS ---
tab_active, tab_history, tab_charts, tab_backtest, tab_database = st.tabs([
    "🔥 Active Trades", 
    "📜 Closed History", 
    "📈 TradingView Chart", 
    "🧪 Historical Backtest",
    "🛠️ Database Operations"
])

# TAB 1: ACTIVE TRADES
with tab_active:
    st.subheader("Currently Open Positions")
    if not df_journal.empty:
        active_df = df_journal[df_journal['status'].isin(['OPEN', 'TP1_HIT'])].copy()
        if not active_df.empty:
            display_cols = ['id', 'timestamp', 'symbol', 'status', 'entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2', 'position_usdt', 'max_risk_usd', 'order_id', 'trigger_reason']
            st.dataframe(active_df[[c for c in display_cols if c in active_df.columns]], use_container_width=True, hide_index=True)
        else:
            st.info("No active signals currently open.")
    else:
        st.info("No trade data found in database.")

# TAB 2: CLOSED HISTORY
with tab_history:
    st.subheader("Closed Performance History")
    if not df_journal.empty:
        closed_df = df_journal[df_journal['status'].isin(['CLOSED_TP2', 'STOPPED_OUT'])].copy()
        if not closed_df.empty:
            display_cols = ['id', 'timestamp', 'closed_timestamp', 'symbol', 'status', 'entry_price', 'exit_price', 'realized_pnl_usd', 'realized_r']
            st.dataframe(closed_df[[c for c in display_cols if c in closed_df.columns]], use_container_width=True, hide_index=True)
        else:
            st.info("No closed trades recorded yet.")
    else:
        st.info("No trade data found in database.")

# TAB 3: TRADINGVIEW CANVAS
with tab_charts:
    col_sym, col_tf = st.columns([2, 1])
    with col_sym:
        selected_symbol = st.selectbox("Select Asset", config.get("watchlist", ["SOL/USDT"]))
    with col_tf:
        selected_tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)
        
    df_chart = fetch_market_data(selected_symbol, timeframe=selected_tf, limit=350, tema_length=int(config.get("htf_tema_period", 50)))
    if not df_chart.empty:
        render_tradingview_chart(df_chart, selected_symbol, df_journal)
    else:
        st.warning("Failed to retrieve chart data.")

# TAB 4: HISTORICAL BACKTEST ENGINE
with tab_backtest:
    st.subheader("🧪 Upgraded Backtesting Engine")
    b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
    with b_col1:
        bt_symbol = st.selectbox("Backtest Asset", config.get("watchlist", ["SOL/USDT"]), key="bt_sym")
    with b_col2:
        bt_tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=1, key="bt_tf")
    with b_col3:
        bt_limit = st.select_slider("Bars Limit", options=[300, 350, 500, 1000], value=350)
        
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        df_bt = fetch_market_data(bt_symbol, timeframe=bt_tf, limit=bt_limit, tema_length=int(config.get("htf_tema_period", 50)))
        if not df_bt.empty:
            params = {
                'initial_balance': config.get("account_balance", 100.0),
                'risk_pct': config.get("risk_pct", 1.0),
                'target_rr': config.get("min_rr_ratio", 2.0),
                'min_adx': config.get("min_adx", 15.0),
                'min_atr_pct': config.get("min_atr_pct", 0.2),
                'proximity_pct': config.get("proximity_threshold_pct", 2.0),
                'atr_mult_sl': config.get("atr_mult_sl", 1.5),
                'use_mtf': config.get("use_mtf_tema_alignment", True),
                'htf_mode': config.get("htf_alignment_mode", "TEMA Slope"),
                'use_atr_trail': config.get("use_adaptive_atr_trail", True),
                'atr_trail_mult': config.get("atr_trail_mult", 2.0),
                'use_equity_filter': config.get("use_equity_curve_filter", True),
                'eq_ma_period': int(config.get("eq_ma_period", 10)),
                'reduced_risk_factor': config.get("reduced_risk_factor", 0.3)
            }
            sim_trades, equity_series, sim_metrics = run_backtest_upgraded(df_bt, params)
            st.json(sim_metrics)
            st.line_chart(equity_series)
            if not sim_trades.empty:
                st.dataframe(sim_trades, use_container_width=True)
        else:
            st.error("Failed to fetch historical market data.")

# TAB 5: DATABASE CONTROL
with tab_database:
    st.subheader("Supabase Management")
    if not df_journal.empty:
        st.dataframe(df_journal, use_container_width=True)
        confirm_clear = st.checkbox("Confirm clear remote database.")
        if st.button("Wipe Journal Database", type="primary", disabled=not confirm_clear):
            clear_remote_journal()

gc.collect()