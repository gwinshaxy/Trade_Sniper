import os
import csv
import time
import json
import asyncio
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

CONFIG_FILE = "config.json"
last_alert_time = {}

def load_config():
    default_config = {
        "account_balance": 1000.0,
        "risk_pct": 1.0,
        "proximity_threshold_pct": 2.0,
        "min_adx": 20.0,
        "min_atr_pct": 0.4,
        "scan_interval_minutes": 15,
        "alert_cooldown_hours": 4,
        "journal_file": "trade_journal.csv",
        "watchlist": [
            "ONDO/USDT",
            "PENDLE/USDT",
            "LINK/USDT",
            "TIA/USDT",
            "NEAR/USDT",
            "SYRUP/USDT"
        ]
    }
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            return {**default_config, **config}
    except Exception:
        return default_config

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

def init_journal_file(journal_file="trade_journal.csv"):
    headers = [
        "Timestamp", "Symbol", "Trigger_Reason", "Entry_Price", 
        "Stop_Loss", "Take_Profit_1", "Take_Profit_2", "Position_USDT", 
        "Max_Risk_USD", "Status", "Exit_Price", "Closed_Timestamp", 
        "Realized_PnL_USD", "Realized_R"
    ]
    
    if not os.path.exists(journal_file):
        with open(journal_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"Initialized new trade journal at '{journal_file}'.")
    else:
        df = pd.read_csv(journal_file)
        missing_cols = [col for col in headers if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                df[col] = None
            df.to_csv(journal_file, index=False)
            print(f"Upgraded journal columns in '{journal_file}'.")

def log_trade_signal(journal_file, symbol, reasons, entry, sl, tp1, tp2, position_usdt, risk_usd):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reasons_str = " | ".join(reasons)
    
    with open(journal_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp_str, symbol, reasons_str, f"{entry:.4f}",
            f"{sl:.4f}", f"{tp1:.4f}", f"{tp2:.4f}", f"{position_usdt:.2f}",
            f"{risk_usd:.2f}", "OPEN", "", "", "", ""
        ])
    print(f"💾 Trade signal logged to '{journal_file}' for {symbol}.")

def calculate_native_tema(df, length=200):
    ema1 = df['close'].ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

def fetch_market_data(symbol, timeframe='1h', limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

    if not ohlcv or len(ohlcv) < 10:
        return None

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    if len(df) >= 200:
        try:
            df['tema_200'] = ta.tema(df['close'], length=200)
            if df['tema_200'].dropna().empty:
                df['tema_200'] = calculate_native_tema(df, length=200)
        except Exception:
            df['tema_200'] = calculate_native_tema(df, length=200)

    # --- CALCULATE ADX (14) & ATR % ---
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None and not adx_df.empty:
            adx_cols = [c for c in adx_df.columns if c.startswith('ADX_')]
            df['adx_14'] = adx_df[adx_cols[0]] if adx_cols else adx_df.iloc[:, 0]
        else:
            df['adx_14'] = 0.0
    except Exception:
        df['adx_14'] = 0.0

    try:
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_pct'] = (df['atr_14'] / df['close']) * 100.0
    except Exception:
        df['atr_14'] = 0.0
        df['atr_pct'] = 0.0
        
    return df

def calculate_volume_profile(df, num_bins=30):
    price_min = df['low'].min()
    price_max = df['high'].max()
    
    bins = np.linspace(price_min, price_max, num_bins)
    df['bin'] = pd.cut(df['close'], bins=bins)
    
    volume_profile = df.groupby('bin', observed=False)['volume'].sum().reset_index()
    volume_profile['price_mid'] = volume_profile['bin'].apply(lambda x: x.mid)
    
    poc_row = volume_profile.loc[volume_profile['volume'].idxmax()]
    poc_price = poc_row['price_mid']
    
    hvn_nodes = volume_profile.sort_values(by='volume', ascending=False).head(3)
    return poc_price, hvn_nodes['price_mid'].tolist()

def calculate_fixed_risk_position(account_balance, entry_price, stop_loss_price, risk_pct=1.0):
    risk_amount = account_balance * (risk_pct / 100.0)
    price_risk_per_unit = abs(entry_price - stop_loss_price)
    
    if price_risk_per_unit == 0:
        return 0, 0, 0
    
    units = risk_amount / price_risk_per_unit
    position_size_usdt = units * entry_price
    return units, position_size_usdt, risk_amount

def safe_format(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:.4f}"

# =====================================================================
# ACTIVE TRADE EVALUATOR ENGINE (TYPE-SAFE & DYNAMIC LOOKBACK)
# =====================================================================
async def evaluate_active_trades(bot, chat_id, config):
    journal_file = config.get("journal_file", "trade_journal.csv")
    if not os.path.exists(journal_file):
        return

    try:
        df = pd.read_csv(journal_file)
    except Exception as e:
        print(f"Error reading journal for evaluation: {e}")
        return

    if 'Status' not in df.columns or df.empty:
        return

    string_cols = ['Status', 'Exit_Price', 'Closed_Timestamp', 'Realized_PnL_USD', 'Realized_R']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')

    open_mask = df['Status'].isin(['OPEN', 'TP1_HIT'])
    if not open_mask.any():
        return

    print(f"\n🔄 Evaluating active signals in trade journal...")
    updated = False

    for idx, row in df[open_mask].iterrows():
        try:
            symbol = str(row['Symbol'])
            entry = float(row['Entry_Price'])
            sl = float(row['Stop_Loss'])
            tp1 = float(row['Take_Profit_1'])
            tp2 = float(row['Take_Profit_2'])
            risk_usd = float(row['Max_Risk_USD'])
            current_status = str(row['Status'])
            trade_time_str = str(row['Timestamp'])
            
            trade_dt = datetime.strptime(trade_time_str, "%Y-%m-%d %H:%M:%S")
            hours_elapsed = (datetime.now() - trade_dt).total_seconds() / 3600
            
            candles_needed = max(50, int(hours_elapsed * 4) + 10)
            candles_needed = min(candles_needed, 1000)

            df_candles = fetch_market_data(symbol, timeframe='15m', limit=candles_needed)
            if df_candles is None or df_candles.empty:
                continue

            df_candles = df_candles[df_candles['timestamp'] >= trade_dt]
            if df_candles.empty:
                continue

            latest_low = df_candles['low'].min()
            latest_high = df_candles['high'].max()
            close_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 1. Check Stop Loss
            if latest_low <= sl:
                pnl_usd = -risk_usd
                realized_r = -1.0
                
                df.at[idx, 'Status'] = 'STOPPED_OUT'
                df.at[idx, 'Exit_Price'] = f"{sl:.4f}"
                df.at[idx, 'Closed_Timestamp'] = str(close_time)
                df.at[idx, 'Realized_PnL_USD'] = f"{pnl_usd:.2f}"
                df.at[idx, 'Realized_R'] = f"{realized_r:.2f}"
                updated = True

                if bot and chat_id:
                    msg = (
                        f"🛑 **TRADE STOPPED OUT: {symbol}** 🛑\n\n"
                        f"• Exit Price: `${sl:.4f}`\n"
                        f"• Realized PnL: `${pnl_usd:.2f}` (-1.0R)\n"
                        f"• Timestamp: `{close_time}`"
                    )
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

            # 2. Check Take Profit 2
            elif latest_high >= tp2:
                r_multiple = abs(tp2 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
                pnl_usd = risk_usd * r_multiple

                df.at[idx, 'Status'] = 'CLOSED_TP2'
                df.at[idx, 'Exit_Price'] = f"{tp2:.4f}"
                df.at[idx, 'Closed_Timestamp'] = str(close_time)
                df.at[idx, 'Realized_PnL_USD'] = f"{pnl_usd:.2f}"
                df.at[idx, 'Realized_R'] = f"{r_multiple:.2f}"
                updated = True

                if bot and chat_id:
                    msg = (
                        f"🎯 **FULL TARGET HIT (TP2): {symbol}** 🎯\n\n"
                        f"• Target Price: `${tp2:.4f}`\n"
                        f"• Realized PnL: `+${pnl_usd:.2f}` (+{r_multiple:.2f}R)\n"
                        f"• Timestamp: `{close_time}`"
                    )
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

            # 3. Check Take Profit 1
            elif latest_high >= tp1 and current_status == 'OPEN':
                r_multiple = abs(tp1 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
                
                df.at[idx, 'Status'] = 'TP1_HIT'
                updated = True

                if bot and chat_id:
                    msg = (
                        f"✅ **FIRST TARGET REACHED (TP1): {symbol}** ✅\n\n"
                        f"• Milestone Price: `${tp1:.4f}`\n"
                        f"• Un-realized Gain: `+{r_multiple:.2f}R`\n"
                        f"• Status: Stop Loss moved to Breakeven (${entry:.4f})"
                    )
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

        except Exception as e:
            print(f"Error evaluating row {idx} for {symbol}: {e}")

    if updated:
        df.to_csv(journal_file, index=False)
        print("💾 Updated active trade statuses in CSV journal.")

# =====================================================================
# ANALYSIS & SCANNER LOOP (FILTERED)
# =====================================================================
async def analyze_symbol(symbol, bot, chat_id, config):
    current_time = time.time()
    
    account_balance = config.get("account_balance", 1000.0)
    risk_pct = config.get("risk_pct", 1.0)
    proximity_threshold = config.get("proximity_threshold_pct", 2.0)
    cooldown_hours = config.get("alert_cooldown_hours", 4)
    journal_file = config.get("journal_file", "trade_journal.csv")
    min_adx = config.get("min_adx", 20.0)
    min_atr_pct = config.get("min_atr_pct", 0.4)
    
    if symbol in last_alert_time:
        elapsed_hours = (current_time - last_alert_time[symbol]) / 3600
        if elapsed_hours < cooldown_hours:
            print(f"[{symbol}] On cooldown ({elapsed_hours:.1f}h / {cooldown_hours}h elapsed). Skipping scan.")
            return

    df_1h = fetch_market_data(symbol, timeframe='1h', limit=500)
    df_4h = fetch_market_data(symbol, timeframe='4h', limit=500)
    
    if df_1h is None or df_4h is None:
        return

    current_price = df_1h['close'].iloc[-1]
    val_1h = df_1h['tema_200'].dropna().iloc[-1] if not df_1h['tema_200'].dropna().empty else None
    val_4h = df_4h['tema_200'].dropna().iloc[-1] if not df_4h['tema_200'].dropna().empty else None
    poc_price, hvn_prices = calculate_volume_profile(df_1h, num_bins=30)
    
    # 1. Proximity Triggers
    triggered_reasons = []
    
    if val_1h is not None:
        dist_1h = abs(current_price - val_1h) / val_1h * 100
        if dist_1h <= proximity_threshold:
            triggered_reasons.append(f"Near 1H 200 TEMA ({dist_1h:.2f}% away)")
            
    if val_4h is not None:
        dist_4h = abs(current_price - val_4h) / val_4h * 100
        if dist_4h <= proximity_threshold:
            triggered_reasons.append(f"Near 4H 200 TEMA ({dist_4h:.2f}% away)")
            
    if poc_price is not None:
        dist_poc = abs(current_price - poc_price) / poc_price * 100
        if dist_poc <= proximity_threshold:
            triggered_reasons.append(f"Near Volume POC ({dist_poc:.2f}% away)")
            
    if not triggered_reasons:
        print(f"[{symbol}] Price (${current_price:.4f}) outside threshold. No trigger.")
        return

    # 2. Hard Indicator Regime Filters (ADX & ATR%)
    current_adx = df_1h['adx_14'].dropna().iloc[-1] if 'adx_14' in df_1h else 0.0
    current_atr_pct = df_1h['atr_pct'].dropna().iloc[-1] if 'atr_pct' in df_1h else 0.0

    if current_adx < min_adx:
        print(f"[{symbol}] ❌ Filter Skipped: ADX ({current_adx:.2f}) below min threshold ({min_adx}).")
        return

    if current_atr_pct < min_atr_pct:
        print(f"[{symbol}] ❌ Filter Skipped: ATR% ({current_atr_pct:.2f}%) below min threshold ({min_atr_pct}%).")
        return

    print(f"🎯 TRIGGER & FILTERS MATCH for {symbol}: {', '.join(triggered_reasons)} (ADX: {current_adx:.2f}, ATR%: {current_atr_pct:.2f}%)")

    # 3. Execution & Sizing Parameters
    direction = "LONG"
    entry_price = current_price
    
    atr_val = df_1h['atr_14'].dropna().iloc[-1] if 'atr_14' in df_1h and not pd.isna(df_1h['atr_14'].iloc[-1]) else entry_price * 0.01
    stop_loss = entry_price - (1.5 * atr_val) if atr_val > 0 else entry_price * 0.985
    
    tp1 = hvn_prices[0] if hvn_prices and hvn_prices[0] > entry_price else entry_price * 1.03
    tp2 = poc_price if poc_price > entry_price else entry_price * 1.05

    units, position_usdt, risk_usd = calculate_fixed_risk_position(
        account_balance, entry_price, stop_loss, risk_pct=risk_pct
    )
    
    rr_ratio = abs(tp1 - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0

    # Log to CSV
    log_trade_signal(
        journal_file, symbol, triggered_reasons, entry_price, 
        stop_loss, tp1, tp2, position_usdt, risk_usd
    )

    last_alert_time[symbol] = current_time

    # Telegram Alert Formatting
    reasons_text = "\n".join([f"• {r}" for r in triggered_reasons])
    message = (
        f"🎯 **PROXIMITY ALERT: {symbol}** 🎯\n\n"
        f"**Trigger Conditions Met:**\n{reasons_text}\n\n"
        f"**Current Price:** `${current_price:.4f}`\n\n"
        f"📈 **Technical Confluence:**\n"
        f"• 1H 200 TEMA: {safe_format(val_1h)}\n"
        f"• 4H 200 TEMA: {safe_format(val_4h)}\n"
        f"• Point of Control (POC): {safe_format(poc_price)}\n"
        f"• ADX (14): `{current_adx:.2f}` (Min: {min_adx})\n"
        f"• ATR %: `{current_atr_pct:.2f}%` (Min: {min_atr_pct}%)\n\n"
        f"🎯 **Trade Parameters ({risk_pct}% Risk Model):**\n"
        f"• Direction: `{direction}`\n"
        f"• Entry Zone: `${entry_price:.4f}`\n"
        f"• Stop Loss: `${stop_loss:.4f}` (Risk: `${risk_usd:.2f}`)\n"
        f"• Target 1 (HVN): `${tp1:.4f}`\n"
        f"• Target 2 (Macro): `${tp2:.4f}`\n\n"
        f"💰 **Position Sizing:**\n"
        f"• Position Value: `${position_usdt:.2f}` ({units:.2f} units)\n"
        f"• Risk/Reward Ratio: `{rr_ratio:.2f}R`\n\n"
        f"📄 Signal logged to `trade_journal.csv`"
    )
    
    if bot and chat_id:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")

async def run_scanner():
    config = load_config()
    init_journal_file(config.get("journal_file", "trade_journal.csv"))

    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or config.get("telegram_bot_token")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID") or config.get("telegram_chat_id")

    bot = Bot(token=telegram_bot_token) if telegram_bot_token else None
    
    if bot and telegram_chat_id:
        startup_msg = "🟢 **Structural Trading Agent Initialized & Active.** Scanning loop starting..."
        await bot.send_message(chat_id=telegram_chat_id, text=startup_msg, parse_mode="Markdown")
    
    while True:
        current_config = load_config()
        watchlist = current_config.get("watchlist", [])
        scan_interval = current_config.get("scan_interval_minutes", 15)
        
        # 1. Run Active Trade Evaluator
        await evaluate_active_trades(bot, telegram_chat_id, current_config)

        # 2. Run Watchlist Proximity Scanner
        print(f"\n--- Starting Scan Cycle ({len(watchlist)} assets) ---")
        for symbol in watchlist:
            try:
                await analyze_symbol(symbol, bot, telegram_chat_id, current_config)
                await asyncio.sleep(2)
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                
        print(f"Cycle completed. Sleeping for {scan_interval} minutes...")
        await asyncio.sleep(scan_interval * 60)

if __name__ == "__main__":
    asyncio.run(run_scanner())