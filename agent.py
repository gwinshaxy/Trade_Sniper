import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import pandas_ta as ta
import ccxt

# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "account_balance": 1000.0,
    "risk_pct": 1.0,
    "stop_loss_pct": 2.5,
    "proximity_threshold_pct": 1.0,
    "min_adx": 20.0,
    "min_atr_pct": 0.5,
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

# Memory store for alert cooldowns: { "SYMBOL": datetime_of_last_alert }
alert_cooldowns = {}

# Global exchange instance (MEXC) to avoid geo-blocking and connection overhead
exchange = ccxt.mexc({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot'
    }
})

# =====================================================================
# CONFIGURATION & UTILS
# =====================================================================
def load_config() -> dict:
    """Loads configuration from config.json or returns default settings if missing/invalid."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                logging.info("Successfully loaded config.json")
                return {**DEFAULT_CONFIG, **config}
        except Exception as e:
            logging.error(f"Error reading config.json ({e}). Falling back to defaults.")
            return DEFAULT_CONFIG
    else:
        logging.warning("config.json not found. Using default internal settings.")
        return DEFAULT_CONFIG

def send_telegram_alert(message: str) -> bool:
    """Sends a formatted alert notification to Telegram."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logging.warning("Telegram credentials missing (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID). Alert skipped.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logging.info("Telegram alert sent successfully.")
            return True
        else:
            logging.error(f"Telegram API Error ({response.status_code}): {response.text}")
            return False
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")
        return False

def log_trade_to_journal(journal_file: str, trade_data: dict):
    """Appends validated trade alert details into a local CSV journal file."""
    file_exists = os.path.isfile(journal_file)
    df_new = pd.DataFrame([trade_data])
    
    try:
        df_new.to_csv(journal_file, mode='a', header=not file_exists, index=False)
        logging.info(f"Logged trade to {journal_file}")
    except Exception as e:
        logging.error(f"Failed to write to journal CSV: {e}")

# =====================================================================
# MARKET DATA & INDICATORS
# =====================================================================
def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 300) -> pd.DataFrame:
    """Fetches candle history from the exchange."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) == 0:
            logging.warning(f"[{symbol}] Exchange returned empty OHLCV payload.")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df
    except Exception as e:
        logging.error(f"[{symbol}] Failed to fetch OHLCV data: {e}")
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates 200 TEMA, 14 ADX (Trend Filter), and 14 ATR % (Volatility Filter)."""
    if df.empty or len(df) < 200:
        return pd.DataFrame()

    # 1. 200-period TEMA
    df["TEMA_200"] = ta.tema(df["close"], length=200)

    # 2. Volatility Filter: 14-period ATR relative to current close (%)
    df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ATR_PCT"] = (df["ATR_14"] / df["close"]) * 100.0

    # 3. Trend Filter: 14-period ADX
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is not None and not adx_df.empty:
        # Locate the ADX column regardless of specific naming variations
        adx_cols = [c for c in adx_df.columns if c.startswith("ADX_")]
        if adx_cols:
            df["ADX_14"] = adx_df[adx_cols[0]]
        else:
            df["ADX_14"] = 0.0
    else:
        df["ADX_14"] = 0.0

    return df

def check_trade_signal(
    df: pd.DataFrame, 
    proximity_threshold: float = 1.0, 
    min_adx: float = 20.0, 
    min_atr_pct: float = 0.5
) -> dict:
    """Evaluates proximity to 200 TEMA alongside Trend (ADX) and Volatility (ATR %) filters safely."""
    if df.empty or "TEMA_200" not in df.columns:
        return {"valid": False, "rejected_reason": "Insufficient OHLCV candle depth (< 200)"}

    # Grab latest completed row
    latest = df.iloc[-1]

    # Guard against NaN/None values before executing float casting
    if pd.isna(latest.get("TEMA_200")) or pd.isna(latest.get("close")):
        return {"valid": False, "rejected_reason": "TEMA_200 or Close price contains NaN/None"}

    close_price = float(latest["close"])
    tema_val = float(latest["TEMA_200"])

    # Safely parse numeric values for ADX and ATR
    adx_raw = latest.get("ADX_14", 0.0)
    adx_val = float(adx_raw) if pd.notna(adx_raw) and adx_raw is not None else 0.0

    atr_raw = latest.get("ATR_PCT", 0.0)
    atr_pct_val = float(atr_raw) if pd.notna(atr_raw) and atr_raw is not None else 0.0

    # Distance to 200 TEMA (%)
    distance_pct = abs(close_price - tema_val) / tema_val * 100.0
    is_in_proximity = distance_pct <= proximity_threshold

    # Regime Filters
    is_trending = adx_val >= min_adx
    has_volatility = atr_pct_val >= min_atr_pct

    valid_signal = is_in_proximity and is_trending and has_volatility

    rejected_reason = None
    if not is_in_proximity:
        rejected_reason = f"Out of Proximity ({distance_pct:.2f}% vs {proximity_threshold}%)"
    elif not is_trending:
        rejected_reason = f"Choppy Market (ADX: {adx_val:.1f} < {min_adx})"
    elif not has_volatility:
        rejected_reason = f"Low Volatility (ATR%: {atr_pct_val:.2f}% < {min_atr_pct}%)"

    return {
        "valid": valid_signal,
        "close": close_price,
        "tema": tema_val,
        "distance_pct": distance_pct,
        "adx": adx_val,
        "atr_pct": atr_pct_val,
        "rejected_reason": rejected_reason
    }

# =====================================================================
# MAIN SCAN LOOP
# =====================================================================
def run_scan_cycle():
    """Executes a single scanning cycle across all watchlist symbols."""
    config = load_config()
    watchlist = config.get("watchlist", [])
    account_balance = config.get("account_balance", 1000.0)
    risk_pct = config.get("risk_pct", 1.0)
    stop_loss_pct = config.get("stop_loss_pct", 2.5)
    proximity_threshold = config.get("proximity_threshold_pct", 1.0)
    min_adx = config.get("min_adx", 20.0)
    min_atr_pct = config.get("min_atr_pct", 0.5)
    cooldown_hours = config.get("alert_cooldown_hours", 4)
    journal_file = config.get("journal_file", "trade_journal.csv")

    logging.info(f"--- Starting Scan Cycle across {len(watchlist)} pairs ---")

    now_utc = datetime.now(timezone.utc)

    for symbol in watchlist:
        # Check Cooldown
        if symbol in alert_cooldowns:
            time_since_last = now_utc - alert_cooldowns[symbol]
            if time_since_last < timedelta(hours=cooldown_hours):
                remaining_mins = int((timedelta(hours=cooldown_hours) - time_since_last).total_seconds() / 60)
                logging.info(f"[{symbol}] In cooldown mode ({remaining_mins} mins remaining). Skipped.")
                continue

        # Fetch Data & Add Indicators
        df = fetch_ohlcv(symbol, timeframe="1h", limit=300)
        df = calculate_indicators(df)

        # Evaluate Signal
        result = check_trade_signal(
            df=df,
            proximity_threshold=proximity_threshold,
            min_adx=min_adx,
            min_atr_pct=min_atr_pct
        )

        if result["valid"]:
            close_price = result["close"]
            tema_val = result["tema"]
            
            # Position Sizing Calculation
            risk_amount = account_balance * (risk_pct / 100.0)
            sl_distance_price = close_price * (stop_loss_pct / 100.0)
            
            # Determine Long or Short structure relative to 200 TEMA
            direction = "LONG" if close_price >= tema_val else "SHORT"
            sl_price = close_price - sl_distance_price if direction == "LONG" else close_price + sl_distance_price
            position_units = risk_amount / sl_distance_price if sl_distance_price > 0 else 0
            position_value_usdt = position_units * close_price

            # Format Telegram Alert Message
            message = (
                f"🚨 *PROXIMITY ALERT: {symbol}* 🚨\n\n"
                f"• *Direction:* `{direction}`\n"
                f"• *Current Price:* `${close_price:,.4f}`\n"
                f"• *200 TEMA:* `${tema_val:,.4f}` (Dist: `{result['distance_pct']:.2f}%`)\n"
                f"• *Calculated SL ({stop_loss_pct}%):* `${sl_price:,.4f}`\n"
                f"• *Risk Amount ({risk_pct}%):* `${risk_amount:,.2f}`\n"
                f"• *Suggested Position Size:* `${position_value_usdt:,.2f}` (`{position_units:.2f}` units)\n\n"
                f"📊 *Regime Metrics:*\n"
                f"• *ADX (14):* `{result['adx']:.1f}` (Min: `{min_adx}`)\n"
                f"• *ATR %:* `{result['atr_pct']:.2f}%` (Min: `{min_atr_pct}%`)\n\n"
                f"⏰ *Timestamp:* `{now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}`"
            )

            # Send Alert & Update Cooldown
            if send_telegram_alert(message):
                alert_cooldowns[symbol] = now_utc
                
                # Log Trade Entry to Journal
                log_trade_to_journal(journal_file, {
                    "timestamp": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": close_price,
                    "tema_200": tema_val,
                    "stop_loss_price": sl_price,
                    "risk_usd": risk_amount,
                    "position_value_usd": position_value_usdt,
                    "adx": result['adx'],
                    "atr_pct": result['atr_pct']
                })
        else:
            logging.info(f"[{symbol}] No Signal -> {result['rejected_reason']}")

def main():
    """Main process loop with continuous scanning interval."""
    logging.info("Initializing Trading Agent...")
    
    # Startup Heartbeat
    send_telegram_alert("🟢 *Trading Agent Initialized & Active.* Scanning loop starting...")

    while True:
        try:
            config = load_config()
            interval_mins = config.get("scan_interval_minutes", 15)
            
            run_scan_cycle()
            
            logging.info(f"Scan complete. Sleeping for {interval_mins} minutes...")
            time.sleep(interval_mins * 60)
            
        except KeyboardInterrupt:
            logging.info("Agent stopped manually.")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
            time.sleep(60) # Pause briefly before attempting next cycle

if __name__ == "__main__":
    main()