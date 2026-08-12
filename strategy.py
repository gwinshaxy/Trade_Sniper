import json
import logging
import urllib.request
import os
import pandas as pd
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

def normalize_symbol(symbol: str) -> str:
    if not symbol:
        return ""
    s = str(symbol).replace('"', '').replace("'", "").strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/{s[-4:]}"
    return s

def load_symbol_config(symbol: str) -> dict:
    clean_symbol = normalize_symbol(symbol).replace("/", "").upper()
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        db_url = db_url.strip('"').strip("'")
    
    if PSYCOPG2_AVAILABLE and db_url:
        try:
            conn = psycopg2.connect(db_url)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT tema_period, rsi_period, rsi_thresh, zone_tolerance, min_sentiment, risk_pct, min_rr, vp_detection_pct, use_rsi_filter, use_candlestick_confirm FROM strategy_parameters WHERE REPLACE(REPLACE(symbol, '\"', ''), '''', '') = %s;", 
                    (clean_symbol,)
                )
                row = cur.fetchone()
                conn.close()
                if row:
                    return dict(row)
        except Exception:
            pass

    return {
        "tema_period": 200, "rsi_period": 14, "rsi_thresh": 42.0,
        "zone_tolerance": 0.0075, "min_sentiment": 0.0, "risk_pct": 1.0,
        "min_rr": 2.0, "vp_detection_pct": 0.07, "use_rsi_filter": True, "use_candlestick_confirm": True
    }

def get_ai_sentiment_score(text: str) -> float:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return 0.5
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"Analyze the financial sentiment of the following text and return ONLY a single float number between 0.0 (bearish) and 1.0 (bullish):\n\n'{text}'"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        score = float(response.text.strip().replace("`", ""))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5

def fetch_klines(symbol: str = "BNB/USDT", interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    norm_sym = normalize_symbol(symbol)
    binance_symbol = norm_sym.replace("/", "")
    binance_url = f"https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval={interval}&limit={min(limit, 1000)}"
    
    try:
        req = urllib.request.Request(binance_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        if data and isinstance(data, list):
            df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'not', 'tbba', 'tbqa', 'ignore'])
            df['time'] = pd.to_datetime(df['open_time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df[['time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception:
        pass
    return pd.DataFrame()

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_tema(series: pd.Series, period: int = 200) -> pd.Series:
    ema1 = calc_ema(series, period)
    ema2 = calc_ema(ema1, period)
    ema3 = calc_ema(ema2, period)
    return (3 * ema1) - (3 * ema2) + ema3

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_volume_profile_gaps(df: pd.DataFrame, num_bins: int = 100, detection_pct: float = 0.07) -> list:
    if df.empty or 'volume' not in df.columns:
        return []
    pLST, pHST = df['low'].min(), df['high'].max()
    if pLST == pHST:
        return []
    pSTP = (pHST - pLST) / num_bins
    vD_vt = np.zeros(num_bins)
    for _, row in df.iterrows():
        sSI = max(int(np.floor((row['low'] - pLST) / pSTP)), 0)
        eSI = min(int(np.floor((row['high'] - pLST) / pSTP)), num_bins - 1)
        for pLI in range(sSI, eSI + 1):
            vD_vt[pLI] += row['volume'] / max(row['high'] - row['low'], 1e-10) * pSTP
    return [float(round(pLST + (i + 0.5) * pSTP, 2)) for i in range(len(vD_vt)) if vD_vt[i] == 0]

def compute_volume_profile(df: pd.DataFrame, num_bins: int = 70):
    min_p, max_p = df['low'].min(), df['high'].max()
    bin_size = (max_p - min_p) / num_bins
    if bin_size <= 0:
        return 0, 0, 0
    bins = np.zeros(num_bins)
    for _, row in df.iterrows():
        idx = min(int(((row['high'] + row['low'] + row['close']) / 3.0 - min_p) / bin_size), num_bins - 1)
        bins[max(idx, 0)] += row['volume']
    poc_idx = np.argmax(bins)
    return min_p + (poc_idx + 0.5) * bin_size, min_p + (poc_idx + 1) * bin_size, min_p + poc_idx * bin_size

def evaluate_signals(df: pd.DataFrame, symbol: str = "ETH/USDT", account_balance: float = 10000.0, **kwargs) -> dict:
    config = load_symbol_config(symbol)
    tema_period = int(config.get("tema_period", 200))
    
    if df.empty or len(df) < tema_period:
        return {"action": "HOLD", "reason": "Insufficient historical data for indicator convergence"}

    # Compute Indicators
    df['tema'] = calc_tema(df['close'], period=tema_period)
    df['rsi'] = calc_rsi(df['close'], period=int(config.get("rsi_period", 14)))
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    current_close = float(latest['close'])
    current_tema = float(latest['tema'])
    current_rsi = float(latest['rsi'])
    
    rsi_thresh = float(config.get("rsi_thresh", 42.0))
    use_rsi = bool(config.get("use_rsi_filter", True))
    use_candlestick = bool(config.get("use_candlestick_confirm", True))
    zone_tolerance = float(config.get("zone_tolerance", 0.0075))
    min_sentiment = float(config.get("min_sentiment", 0.0))
    min_rr = float(config.get("min_rr", 2.0))
    risk_pct = float(config.get("risk_pct", 1.0))

    # Optional Sentiment Check via kwargs or default
    sentiment_score = kwargs.get("sentiment_score", 0.5)
    if sentiment_score < min_sentiment:
        return {"action": "HOLD", "reason": "Sentiment score below minimum threshold"}

    # Zone Proximity & Trend Rules
    upper_zone = current_tema * (1.0 + zone_tolerance)
    lower_zone = current_tema * (1.0 - zone_tolerance)

    bullish_candlestick = (latest['close'] > latest['open']) if use_candlestick else True
    bearish_candlestick = (latest['close'] < latest['open']) if use_candlestick else True

    rsi_long_ok = (current_rsi >= rsi_thresh) if use_rsi else True
    rsi_short_ok = (current_rsi <= (100 - rsi_thresh)) if use_rsi else True

    # Long Setup
    if current_close >= lower_zone and current_close <= upper_zone and rsi_long_ok and bullish_candlestick:
        stop_loss = round(current_tema * (1.0 - (zone_tolerance * 1.5)), 5)
        risk_distance = current_close - stop_loss
        if risk_distance <= 0:
            return {"action": "HOLD", "reason": "Invalid risk distance computed"}
        take_profit = round(current_close + (risk_distance * min_rr), 5)
        risk_amt = account_balance * (risk_pct / 100.0)
        position_size = round(risk_amt / risk_distance, 4)

        return {
            "action": "BUY",
            "symbol": symbol,
            "direction": "LONG",
            "entry_price": current_close,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": min_rr,
            "position_size": position_size
        }

    # Short Setup
    elif current_close <= upper_zone and current_close >= lower_zone and rsi_short_ok and bearish_candlestick:
        stop_loss = round(current_tema * (1.0 + (zone_tolerance * 1.5)), 5)
        risk_distance = stop_loss - current_close
        if risk_distance <= 0:
            return {"action": "HOLD", "reason": "Invalid risk distance computed"}
        take_profit = round(current_close - (risk_distance * min_rr), 5)
        risk_amt = account_balance * (risk_pct / 100.0)
        position_size = round(risk_amt / risk_distance, 4)

        return {
            "action": "SELL",
            "symbol": symbol,
            "direction": "SHORT",
            "entry_price": current_close,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": min_rr,
            "position_size": position_size
        }

    return {"action": "HOLD", "reason": "Price action outside target TEMA zones or filters unmet"}