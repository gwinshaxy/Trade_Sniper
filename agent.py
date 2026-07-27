import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
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
    "proximity_threshold_pct": 1.5,
    "min_adx": 18.0,
    "min_atr_pct": 0.4,
    "scan_interval_minutes": 15,
    "alert_cooldown_hours": 4,
    "journal_file": "trade_journal.csv",
    "watchlist": ["ONDO/USDT", "PENDLE/USDT", "LINK/USDT", "TIA/USDT", "NEAR/USDT"]
}

alert_cooldowns = {}
exchange = ccxt.mexc({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})

# =====================================================================
# HELPER & UTILITY FUNCTIONS
# =====================================================================
def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return {**DEFAULT_CONFIG, **config}
        except Exception:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

def send_telegram_alert(message: str) -> bool:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        res = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
        return res.status_code == 200
    except Exception as e:
        logging.error(f"Telegram error: {e}")
        return False

def fetch_paginated_ohlcv(symbol: str, timeframe: str = "1h", total_candles: int = 1000) -> pd.DataFrame:
    """Fetches enough historical data via pagination to warm up TEMA 200 and Volume Profile."""
    try:
        all_ohlcv = []
        limit_per_call = 500
        
        # Calculate start time
        tf_ms = exchange.parse_timeframe(timeframe) * 1000
        since = exchange.milliseconds() - (total_candles * tf_ms)
        
        while len(all_ohlcv) < total_candles:
            fetch_limit = min(limit_per_call, total_candles - len(all_ohlcv))
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=fetch_limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(0.1) # Rate limit protection

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        return df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"[{symbol}] Failed fetching {timeframe} candles: {e}")
        return pd.DataFrame()

def calculate_volume_profile(df: pd.DataFrame, bins: int = 30):
    """Calculates Point of Control (POC) and High Volume Nodes (HVN)."""
    if df.empty or len(df) < 50:
        return None, None, None

    price_min = df["low"].min()
    price_max = df["high"].max()
    counts, bin_edges = np.histogram(df["close"], bins=bins, weights=df["volume"], range=(price_min, price_max))
    
    poc_idx = np.argmax(counts)
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0
    
    # Sort bins by volume to find HVNs
    sorted_indices = np.argsort(counts)[::-1]
    hvn_prices = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in sorted_indices[:3]]
    
    return poc_price, hvn_prices, counts

# =====================================================================
# TECHNICAL ANALYSIS ENGINE
# =====================================================================
def process_market_data(symbol: str):
    # Fetch 1H & 4H Data with sufficient warmup depth
    df_1h = fetch_paginated_ohlcv(symbol, timeframe="1h", total_candles=1000)
    df_4h = fetch_paginated_ohlcv(symbol, timeframe="4h", total_candles=600)

    if df_1h.empty or len(df_1h) < 400 or df_4h.empty or len(df_4h) < 250:
        return None

    # Calculate 1H Indicators
    df_1h["TEMA_200"] = ta.tema(df_1h["close"], length=200)
    df_1h["ATR_14"] = ta.atr(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
    df_1h["ATR_PCT"] = (df_1h["ATR_14"] / df_1h["close"]) * 100.0
    
    adx_df = ta.adx(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
    df_1h["ADX_14"] = adx_df[[c for c in adx_df.columns if c.startswith("ADX_")][0]] if adx_df is not None else 0.0

    # Calculate 4H TEMA 200
    df_4h["TEMA_200_4H"] = ta.tema(df_4h["close"], length=200)

    latest_1h = df_1h.iloc[-1]
    latest_4h = df_4h.iloc[-1]

    # Validate Non-NaN
    if pd.isna(latest_1h["TEMA_200"]) or pd.isna(latest_4h["TEMA_200_4H"]):
        return None

    # Compute Volume Profile on 1H lookback
    poc_price, hvn_prices, _ = calculate_volume_profile(df_1h.tail(300))

    return {
        "df_1h": df_1h,
        "close": float(latest_1h["close"]),
        "tema_1h": float(latest_1h["TEMA_200"]),
        "tema_4h": float(latest_4h["TEMA_200_4H"]),
        "atr_14": float(latest_1h["ATR_14"]),
        "atr_pct": float(latest_1h["ATR_PCT"]),
        "adx_14": float(latest_1h["ADX_14"]),
        "poc": poc_price,
        "hvns": hvn_prices
    }

def evaluate_signal(symbol: str, data: dict, config: dict):
    close = data["close"]
    tema_1h = data["tema_1h"]
    tema_4h = data["tema_4h"]
    poc = data["poc"]
    hvns = data["hvns"]
    
    dist_1h = abs(close - tema_1h) / tema_1h * 100.0
    dist_4h = abs(close - tema_4h) / tema_4h * 100.0
    
    is_near_1h = dist_1h <= config["proximity_threshold_pct"]
    is_near_4h = dist_4h <= config["proximity_threshold_pct"]
    
    if not (is_near_1h or is_near_4h):
        return {"valid": False, "reason": f"Out of proximity (1H: {dist_1h:.2f}%, 4H: {dist_4h:.2f}%)"}

    if data["adx_14"] < config["min_adx"]:
        return {"valid": False, "reason": f"Choppy Market (ADX: {data['adx_14']:.1f} < {config['min_adx']})"}

    # Determine Direction based on TEMA baseline
    direction = "LONG" if close >= tema_1h else "SHORT"

    # Volume Profile Structural Stop Loss & Targets
    if direction == "LONG":
        # SL dynamic: 1.5x ATR below entry or lowest recent swing low
        sl_price = close - max(1.5 * data["atr_14"], close * 0.015)
        # Target 1: Nearest HVN above entry; Target 2: Macro POC or 2x R:R
        t1_candidates = [h for h in hvns if h > close] if hvns else []
        tp1 = min(t1_candidates) if t1_candidates else close + (close - sl_price) * 1.5
        tp2 = poc if (poc and poc > tp1) else close + (close - sl_price) * 2.5
    else:
        sl_price = close + max(1.5 * data["atr_14"], close * 0.015)
        t1_candidates = [h for h in hvns if h < close] if hvns else []
        tp1 = max(t1_candidates) if t1_candidates else close - (sl_price - close) * 1.5
        tp2 = poc if (poc and poc < tp1) else close - (sl_price - close) * 2.5

    risk_per_unit = abs(close - sl_price)
    reward_per_unit = abs(tp1 - close)
    rr_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0

    return {
        "valid": True,
        "direction": direction,
        "close": close,
        "dist_1h": dist_1h,
        "tema_1h": tema_1h,
        "tema_4h": tema_4h,
        "poc": poc,
        "sl_price": sl_price,
        "tp1": tp1,
        "tp2": tp2,
        "rr_ratio": rr_ratio,
        "adx": data["adx_14"],
        "atr_pct": data["atr_pct"]
    }

# =====================================================================
# JOURNALING & TELEGRAM ALERTS
# =====================================================================
def log_to_journal(journal_file: str, trade_data: dict):
    """Logs clean structured trade entries matching dashboard UI columns."""
    columns = ["Timestamp", "Symbol", "Trigger_Reason", "Entry_Price", "Stop_Loss", "Take_Profit_1", "Take_Profit_2", "Position_USDT", "Max_Risk_USD", "Status"]
    
    file_exists = os.path.isfile(journal_file)
    df_row = pd.DataFrame([trade_data])
    
    try:
        df_row.to_csv(journal_file, mode='a', header=not file_exists, index=False)
        logging.info(f"Successfully logged structural trade to {journal_file}")
    except Exception as e:
        logging.error(f"Journal writing error: {e}")

def run_scan_cycle():
    config = load_config()
    watchlist = config.get("watchlist", [])
    journal_file = config.get("journal_file", "trade_journal.csv")
    cooldown_hours = config.get("alert_cooldown_hours", 4)
    now_utc = datetime.now(timezone.utc)

    logging.info(f"--- Starting Structural Scan Cycle across {len(watchlist)} pairs ---")

    for symbol in watchlist:
        if symbol in alert_cooldowns:
            if now_utc - alert_cooldowns[symbol] < timedelta(hours=cooldown_hours):
                continue

        market_data = process_market_data(symbol)
        if not market_data:
            logging.warning(f"[{symbol}] Incomplete market data or warming up indicators.")
            continue

        sig = evaluate_signal(symbol, market_data, config)
        if sig["valid"]:
            acc_bal = config.get("account_balance", 1000.0)
            risk_pct = config.get("risk_pct", 1.0)
            risk_usd = acc_bal * (risk_pct / 100.0)
            
            sl_dist = abs(sig["close"] - sig["sl_price"])
            units = risk_usd / sl_dist if sl_dist > 0 else 0
            position_usdt = units * sig["close"]

            # Telegram Alert Body (Matching Structural Format)
            message = (
                f"🎯 *PROXIMITY ALERT: {symbol}* 🎯\n\n"
                f"*Trigger Conditions Met:*\n"
                f"• Near 1H 200 TEMA ({sig['dist_1h']:.2f}% away)\n\n"
                f"*Current Price:* `${sig['close']:,.4f}`\n\n"
                f"📈 *Technical Confluence:*\n"
                f"• *1H 200 TEMA:* `${sig['tema_1h']:,.4f}`\n"
                f"• *4H 200 TEMA:* `${sig['tema_4h']:,.4f}`\n"
                f"• *Point of Control (POC):* `${sig['poc']:,.4f}`\n\n"
                f"🎯 *Trade Parameters ({risk_pct}% Risk Model):*\n"
                f"• *Direction:* `{sig['direction']}`\n"
                f"• *Entry Zone:* `${sig['close']:,.4f}`\n"
                f"• *Stop Loss:* `${sig['sl_price']:,.4f}` (Risk: `${risk_usd:,.2f}`)\n"
                f"• *Target 1 (HVN):* `${sig['tp1']:,.4f}`\n"
                f"• *Target 2 (Macro):* `${sig['tp2']:,.4f}`\n\n"
                f"💰 *Position Sizing:*\n"
                f"• *Position Value:* `${position_usdt:,.2f}` (`{units:.2f}` units)\n"
                f"• *Risk/Reward Ratio:* `{sig['rr_ratio']:.2f}R`\n\n"
                f"📄 *Signal logged to trade_journal.csv*"
            )

            if send_telegram_alert(message):
                alert_cooldowns[symbol] = now_utc
                log_to_journal(journal_file, {
                    "Timestamp": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "Symbol": symbol,
                    "Trigger_Reason": f"Near 1H 200 TEMA ({sig['dist_1h']:.2f}% away)",
                    "Entry_Price": sig["close"],
                    "Stop_Loss": sig["sl_price"],
                    "Take_Profit_1": sig["tp1"],
                    "Take_Profit_2": sig["tp2"],
                    "Position_USDT": round(position_usdt, 2),
                    "Max_Risk_USD": round(risk_usd, 2),
                    "Status": "OPEN"
                })
        else:
            logging.info(f"[{symbol}] No Signal -> {sig['reason']}")

def main():
    logging.info("Initializing Structural Trading Agent...")
    send_telegram_alert("🟢 *Structural Trading Agent Initialized & Active.* Scanning loop starting...")
    while True:
        try:
            config = load_config()
            run_scan_cycle()
            time.sleep(config.get("scan_interval_minutes", 15) * 60)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Unexpected loop error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()