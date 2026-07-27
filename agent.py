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
    """Reads settings dynamically from config.json."""
    if not os.path.exists(CONFIG_FILE):
        default_config = {
            "account_balance": 1000.0,
            "risk_pct": 1.0,
            "proximity_threshold_pct": 0.75,
            "scan_interval_minutes": 15,
            "alert_cooldown_hours": 4,
            "journal_file": "trade_journal.csv",
            "watchlist": [
                "ONDO/USDT",
                "PENDLE/USDT",
                "LINK/USDT",
                "TIA/USDT",
                "NEAR/USDT"
            ]
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_exchange():
    """Tries MEXC first, then Gate.io as fallbacks."""
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
    """Ensures trade_journal.csv exists with valid header columns."""
    if not os.path.exists(journal_file):
        with open(journal_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", 
                "Symbol", 
                "Trigger_Reason", 
                "Entry_Price", 
                "Stop_Loss", 
                "Take_Profit_1", 
                "Take_Profit_2", 
                "Position_USDT", 
                "Max_Risk_USD", 
                "Status"
            ])
        print(f"Initialized new trade journal at '{journal_file}'.")

def log_trade_signal(journal_file, symbol, reasons, entry, sl, tp1, tp2, position_usdt, risk_usd):
    """Appends a new trade signal entry into the CSV journal."""
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reasons_str = " | ".join(reasons)
    
    with open(journal_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp_str,
            symbol,
            reasons_str,
            f"{entry:.4f}",
            f"{sl:.4f}",
            f"{tp1:.4f}",
            f"{tp2:.4f}",
            f"{position_usdt:.2f}",
            f"{risk_usd:.2f}",
            "OPEN"
        ])
    print(f"💾 Trade signal logged to '{journal_file}' for {symbol}.")

def calculate_native_tema(df, length=200):
    """Fallback native pandas 200 TEMA calculation."""
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

    if not ohlcv or len(ohlcv) < 200:
        print(f"Insufficient candle history for {symbol}.")
        return None

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    try:
        df['tema_200'] = ta.tema(df['close'], length=200)
        if df['tema_200'].dropna().empty:
            df['tema_200'] = calculate_native_tema(df, length=200)
    except Exception:
        df['tema_200'] = calculate_native_tema(df, length=200)
        
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
        return 0, 0
    
    units = risk_amount / price_risk_per_unit
    position_size_usdt = units * entry_price
    return units, position_size_usdt

def safe_format(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:.4f}"

async def analyze_symbol(symbol, bot, chat_id, config):
    current_time = time.time()
    
    account_balance = config.get("account_balance", 1000.0)
    risk_pct = config.get("risk_pct", 1.0)
    proximity_threshold = config.get("proximity_threshold_pct", 0.75)
    cooldown_hours = config.get("alert_cooldown_hours", 4)
    journal_file = config.get("journal_file", "trade_journal.csv")
    
    # 1. Check Cooldown
    if symbol in last_alert_time:
        elapsed_hours = (current_time - last_alert_time[symbol]) / 3600
        if elapsed_hours < cooldown_hours:
            print(f"[{symbol}] On cooldown ({elapsed_hours:.1f}h / {cooldown_hours}h elapsed). Skipping scan.")
            return

    # 2. Fetch Data
    df_1h = fetch_market_data(symbol, timeframe='1h', limit=500)
    df_4h = fetch_market_data(symbol, timeframe='4h', limit=500)
    
    if df_1h is None or df_4h is None:
        return

    current_price = df_1h['close'].iloc[-1]
    val_1h = df_1h['tema_200'].dropna().iloc[-1] if not df_1h['tema_200'].dropna().empty else None
    val_4h = df_4h['tema_200'].dropna().iloc[-1] if not df_4h['tema_200'].dropna().empty else None
    poc_price, hvn_prices = calculate_volume_profile(df_1h, num_bins=30)
    
    # 3. Check Proximity Triggers
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

    print(f"🎯 TRIGGER MATCH for {symbol}: {', '.join(triggered_reasons)}")

    # 4. Calculate Risk & Position
    entry_price = current_price
    stop_loss = entry_price * 0.965  
    tp1 = hvn_prices[0] if hvn_prices[0] > entry_price else entry_price * 1.05
    tp2 = entry_price * 1.12
    risk_usd = account_balance * (risk_pct / 100.0)
    
    units, position_usdt = calculate_fixed_risk_position(
        account_balance, entry_price, stop_loss, risk_pct=risk_pct
    )
    
    # 5. Log & Dispatch
    log_trade_signal(
        journal_file, symbol, triggered_reasons, entry_price, 
        stop_loss, tp1, tp2, position_usdt, risk_usd
    )

    last_alert_time[symbol] = current_time

    reasons_text = "\n".join([f"• {r}" for r in triggered_reasons])
    message = (
        f"🎯 PROXIMITY ALERT: {symbol} 🎯\n\n"
        f"Trigger Conditions Met:\n{reasons_text}\n\n"
        f"Current Price: ${current_price:.4f}\n\n"
        f"📈 Technical Confluence:\n"
        f"• 1H 200 TEMA: {safe_format(val_1h)}\n"
        f"• 4H 200 TEMA: {safe_format(val_4h)}\n"
        f"• Point of Control (POC): ${poc_price:.4f}\n\n"
        f"🎯 Trade Parameters ({risk_pct}% Risk Model):\n"
        f"• Entry Zone: ${entry_price:.4f}\n"
        f"• Stop Loss: ${stop_loss:.4f} (Risk: ${risk_usd:.2f})\n"
        f"• Target 1 (HVN): ${tp1:.4f}\n"
        f"• Target 2 (Macro): ${tp2:.4f}\n\n"
        f"💰 Position Sizing:\n"
        f"• Position Value: {position_usdt:.2f} USDT ({units:.2f} units)\n"
        f"• Risk/Reward Ratio: {abs(tp1 - entry_price) / abs(entry_price - stop_loss):.2f}R\n\n"
        f"📁 Signal logged to {journal_file}"
    )
    
    await bot.send_message(chat_id=chat_id, text=message)

async def run_scanner():
    config = load_config()
    init_journal_file(config.get("journal_file", "trade_journal.csv"))

    # Safely load Telegram credentials from Env or Config
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or config.get("telegram_bot_token")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID") or config.get("telegram_chat_id")

    if not telegram_bot_token or not telegram_chat_id:
        print("❌ Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing!")
        return

    bot = Bot(token=telegram_bot_token)
    
    startup_msg = (
        "🔍 Filtered Watchlist Scanner & Journal Active\n\n"
        f"• Assets Monitored: {', '.join(config['watchlist'])}\n"
        f"• Proximity Threshold: Within {config['proximity_threshold_pct']}% of key levels\n"
        f"• Alert Cooldown: {config['alert_cooldown_hours']} Hours\n"
        f"• Trade Logging: Enabled ({config.get('journal_file', 'trade_journal.csv')})"
    )
    await bot.send_message(chat_id=telegram_chat_id, text=startup_msg)
    
    while True:
        current_config = load_config()
        watchlist = current_config.get("watchlist", [])
        scan_interval = current_config.get("scan_interval_minutes", 15)
        
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