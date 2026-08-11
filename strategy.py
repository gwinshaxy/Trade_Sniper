import json
import logging
import urllib.request
import urllib.error
import os
import pandas as pd
import numpy as np
from google import genai

from dotenv import load_dotenv
load_dotenv()

# ==========================================
# PROXY & NETWORK CONFIGURATION ROUTING
# ==========================================
HTTP_PROXY = os.getenv("HTTP_PROXY") or os.getenv("PROXY_URL")
HTTPS_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("PROXY_URL")

if HTTP_PROXY or HTTPS_PROXY:
    os.environ["HTTP_PROXY"] = HTTP_PROXY or HTTPS_PROXY
    os.environ["HTTPS_PROXY"] = HTTPS_PROXY or HTTP_PROXY
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,.supabase.co"

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logging.warning("psycopg2 is not installed. Database configuration fetching will be unavailable.")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "strategy_config.json")

def normalize_symbol(symbol: str) -> str:
    """Ensures uniform slash-separated formatting (e.g., 'ETH/USDT')."""
    s = symbol.strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/{s[-4:]}"
    return s

def load_optimized_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not read {CONFIG_FILE}, using defaults: {e}")
    return {
        "tema_period": 200,
        "rsi_period": 14,
        "zone_tolerance": 0.015,
        "min_sentiment": 0.0,
        "risk_pct": 1.0,
        "min_rr": 2.0,
        "vp_detection_pct": 0.07
    }

def load_symbol_config(symbol: str) -> dict:
    """
    Fetches asset-specific optimized parameters dynamically from Supabase database
    using slash-free uppercase symbols (e.g. 'BNBUSDT') to match database constraints.
    """
    clean_symbol = symbol.replace("/", "").replace("-", "").upper()
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        db_url = db_url.strip('"').strip("'")
    
    if PSYCOPG2_AVAILABLE and db_url:
        try:
            conn = psycopg2.connect(db_url)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT tema_period, rsi_period, zone_tolerance, min_sentiment, risk_pct, min_rr, vp_detection_pct 
                    FROM strategy_parameters 
                    WHERE symbol = %s;
                    """, 
                    (clean_symbol,)
                )
                row = cur.fetchone()
                conn.close()
                if row:
                    return dict(row)
        except Exception as e:
            logging.warning(f"Could not load Supabase config for {clean_symbol}: {e}")

    return load_optimized_config()

def get_ai_sentiment_score(text: str) -> float:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return 0.5
    
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"Analyze the financial/market sentiment of the following text and return ONLY a single float number between 0.0 (extremely bearish) and 1.0 (extremely bullish):\n\n'{text}'"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        score = float(response.text.strip().replace("`", ""))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5

def fetch_fundamental_sentiment(news_headlines: list) -> float:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not news_headlines:
        return 0.0

    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        Analyze the following market news headlines and economic data for forex/crypto. 
        Provide a single sentiment score between -1.0 (very negative) and 1.0 (very positive).
        Return ONLY a floating-point number.
        
        Headlines:
        {news_headlines}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return float(response.text.strip().replace("`", ""))
    except Exception:
        return 0.0

def fetch_klines(symbol: str = "BNB/USDT", interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    norm_sym = normalize_symbol(symbol)
    clean_base = norm_sym.split("/")[0] if "/" in norm_sym else norm_sym.replace("USDT", "")
    
    symbol_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "ADA": "cardano",
        "SOL": "solana",
        "XRP": "ripple",
        "DOGE": "dogecoin",
        "DOT": "polkadot",
        "AVAX": "avalanche-2"
    }
    
    coin_id = symbol_map.get(clean_base, "bitcoin")
    days = max(1, min(365, int(limit / 24))) if interval in ["1h", "4h", "1d"] else 30
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close'])
        df['time'] = pd.to_datetime(df['time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
        df['volume'] = 1000.0
        
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        return df[['time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logging.error(f"Public Market Data API error for {symbol}: {e}")
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
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_volume_profile_gaps(df: pd.DataFrame, num_bins: int = 100, detection_pct: float = 0.07) -> list:
    if df.empty or 'volume' not in df.columns:
        return []

    pLST = df['low'].min()
    pHST = df['high'].max()
    
    if pLST == pHST:
        return []

    pSTP = (pHST - pLST) / num_bins
    vD_vt = np.zeros(num_bins)
    
    for _, row in df.iterrows():
        lH, lL, lV = row['high'], row['low'], row['volume']
        lR = max(lH - lL, 1e-10)
        
        sSI = max(int(np.floor((lL - pLST) / pSTP)), 0)
        eSI = min(int(np.floor((lH - pLST) / pSTP)), num_bins - 1)
        
        for pLI in range(sSI, eSI + 1):
            pL = pLST + pLI * pSTP
            
            if lL >= pL and lH > pL + pSTP:
                vPOR = (pL + pSTP - lL) / lR
            elif lH <= pL + pSTP and lL < pL:
                vPOR = (lH - pL) / lR
            elif lL >= pL and lH <= pL + pSTP:
                vPOR = 1.0
            else:
                vPOR = pSTP / lR

            vD_vt[pLI] += lV * vPOR

    noN = int(num_bins * detection_pct)
    if noN < 1:
        noN = 1

    max_vol = float(vD_vt.max()) if len(vD_vt) > 0 else 0.0
    if max_vol == 0.0:
        return []

    tVT = np.concatenate([np.full(noN, max_vol), vD_vt, np.full(noN, max_vol)])
    detected_gap_prices = []

    for vn in range(2 * noN, num_bins + 2 * noN):
        current_val = tVT[vn - noN]
        
        uNth = True
        for cVN in range(vn - 2 * noN, vn - noN):
            if current_val >= tVT[cVN]:
                uNth = False
                break
                
        lNth = True
        for cVN in range(vn - noN + 1, vn + 1):
            if current_val >= tVT[cVN]:
                lNth = False
                break

        if uNth and lNth:
            bin_idx = vn - 2 * noN
            gap_price = pLST + (bin_idx + 0.5) * pSTP
            detected_gap_prices.append(float(round(gap_price, 2)))

    return detected_gap_prices

def compute_volume_profile(df: pd.DataFrame, num_bins: int = 70):
    min_p = df['low'].min()
    max_p = df['high'].max()
    bin_size = (max_p - min_p) / num_bins
    
    if bin_size <= 0:
        return 0, 0, 0

    bins = np.zeros(num_bins)
    for _, row in df.iterrows():
        avg_price = (row['high'] + row['low'] + row['close']) / 3.0
        idx = int((avg_price - min_p) / bin_size)
        if idx >= num_bins:
            idx = num_bins - 1
        if idx < 0:
            idx = 0
        bins[idx] += row['volume']

    poc_idx = np.argmax(bins)
    poc_price = min_p + (poc_idx + 0.5) * bin_size

    total_vol = np.sum(bins)
    target_vol = total_vol * 0.70
    
    curr_vol = bins[poc_idx]
    val_idx = poc_idx
    vah_idx = poc_idx

    while curr_vol < target_vol and (val_idx > 0 or vah_idx < num_bins - 1):
        prev_v = bins[val_idx - 1] if val_idx > 0 else 0
        next_v = bins[vah_idx + 1] if vah_idx < num_bins - 1 else 0

        if next_v >= prev_v and vah_idx < num_bins - 1:
            vah_idx += 1
            curr_vol += bins[vah_idx]
        elif val_idx > 0:
            val_idx -= 1
            curr_vol += bins[val_idx]
        elif vah_idx < num_bins - 1:
            vah_idx += 1
            curr_vol += bins[vah_idx]
        else:
            break

    vah_price = min_p + (vah_idx + 1) * bin_size
    val_price = min_p + val_idx * bin_size

    return poc_price, vah_price, val_price

def is_bullish_pinbar(candle: pd.Series) -> bool:
    total_range = candle['high'] - candle['low']
    if total_range == 0:
        return False
    body_size = abs(candle['close'] - candle['open'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    return (lower_wick >= 1.5 * body_size) and ((candle['close'] - candle['low']) >= 0.50 * total_range)

def is_bearish_pinbar(candle: pd.Series) -> bool:
    total_range = candle['high'] - candle['low']
    if total_range == 0:
        return False
    body_size = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    return (upper_wick >= 1.5 * body_size) and ((candle['high'] - candle['close']) >= 0.50 * total_range)

def is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    prev_bearish = prev['close'] < prev['open']
    curr_bullish = curr['close'] > curr['open']
    engulfs = (curr['close'] >= prev['open']) and (curr['open'] <= prev['close'])
    return curr_bullish and engulfs

def is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    prev_bullish = prev['close'] > prev['open']
    curr_bearish = curr['close'] < curr['open']
    engulfs = (curr['close'] <= prev['open']) and (curr['open'] >= prev['close'])
    return curr_bearish and engulfs

def evaluate_signals(df: pd.DataFrame, symbol: str = "ETH/USDT", account_balance: float = 10000.0, 
                     tema_period: int = None, rsi_period: int = None, 
                     zone_tolerance: float = None, risk_pct: float = None, 
                     min_rr: float = None,  
                     use_rsi_filter: bool = True, use_candlestick_confirm: bool = True,
                     use_sentiment_filter: bool = False, min_sentiment: float = None,
                     vp_detection_pct: float = None) -> dict:
    
    normalized_symbol = normalize_symbol(symbol)
    dynamic_cfg = load_symbol_config(normalized_symbol)
    
    tema_period = int(tema_period) if tema_period is not None else int(dynamic_cfg.get("tema_period", 200))
    rsi_period = int(rsi_period) if rsi_period is not None else int(dynamic_cfg.get("rsi_period", 14))
    zone_tolerance = float(zone_tolerance) if zone_tolerance is not None else float(dynamic_cfg.get("zone_tolerance", 0.015))
    min_sentiment = float(min_sentiment) if min_sentiment is not None else float(dynamic_cfg.get("min_sentiment", 0.0))
    risk_pct = float(risk_pct) if risk_pct is not None else float(dynamic_cfg.get("risk_pct", 1.0))
    min_rr = float(min_rr) if min_rr is not None else float(dynamic_cfg.get("min_rr", 2.0))
    vp_detection_pct = float(vp_detection_pct) if vp_detection_pct is not None else float(dynamic_cfg.get("vp_detection_pct", 0.07))

    df['200_TEMA'] = calc_tema(df['close'], tema_period)
    df['RSI'] = calc_rsi(df['close'], rsi_period)
    df['RSI_SMA'] = df['RSI'].rolling(5).mean()
    df['TEMA_slope'] = df['200_TEMA'].diff(3)

    poc, vah, val = compute_volume_profile(df, num_bins=70)
    gap_levels = calculate_volume_profile_gaps(df, num_bins=100, detection_pct=vp_detection_pct)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest['close']
    tema = latest['200_TEMA']

    sentiment_score = 0.0
    if use_sentiment_filter and 'sentiment_score' in latest:
        sentiment_score = latest['sentiment_score']

    in_gap_zone = False
    if gap_levels:
        min_gap_dist = min([abs(price - gap) / gap for gap in gap_levels])
        in_gap_zone = min_gap_dist <= zone_tolerance

    above_gaps = [g for g in gap_levels if g > price]
    below_gaps = [g for g in gap_levels if g < price]

    is_uptrend = (price >= tema) and (latest['TEMA_slope'] > 0)
    in_poc_zone_long = abs(price - poc) / poc <= zone_tolerance if poc > 0 else True
    in_val_zone_long = abs(price - val) / val <= zone_tolerance if val > 0 else True

    in_structural_support = in_poc_zone_long or in_val_zone_long or in_gap_zone
    rsi_turning_long = (latest['RSI'] > latest['RSI_SMA']) if use_rsi_filter else True
    
    has_bullish_confirm = (
        is_bullish_pinbar(latest) or 
        is_bullish_engulfing(prev, latest)
    ) if use_candlestick_confirm else True
    
    sentiment_pass_long = (sentiment_score >= min_sentiment) if use_sentiment_filter else True

    if is_uptrend and in_structural_support and rsi_turning_long and has_bullish_confirm and sentiment_pass_long:
        tp_price = min(above_gaps) if above_gaps else (vah if vah > price else price * 1.02)
        nearest_support_gap = max(below_gaps) if below_gaps else val
        sl_price = min(latest['low'], nearest_support_gap) * 0.9950
        
        risk = price - sl_price
        if risk > 0:
            rr_ratio = (tp_price - price) / risk
            if rr_ratio >= min_rr:
                risk_usd = account_balance * (risk_pct / 100.0)
                position_size = risk_usd / risk
                
                return {
                    "action": "BUY",
                    "pair": normalized_symbol,
                    "direction": "LONG",
                    "entry": float(price),
                    "sl": float(round(sl_price, 2)),
                    "tp": float(round(tp_price, 2)),
                    "rr": float(round(rr_ratio, 2)),
                    "risk_pct": float(risk_pct),
                    "size": float(round(position_size, 4)),
                    "detected_gaps": gap_levels
                }

    is_downtrend = (price <= tema) and (latest['TEMA_slope'] < 0)
    in_poc_zone_short = abs(price - poc) / poc <= zone_tolerance if poc > 0 else True
    in_vah_zone_short = abs(price - vah) / vah <= zone_tolerance if vah > 0 else True

    in_structural_resistance = in_poc_zone_short or in_vah_zone_short or in_gap_zone
    rsi_turning_short = (latest['RSI'] < latest['RSI_SMA']) if use_rsi_filter else True
    
    has_bearish_confirm = (
        is_bearish_pinbar(latest) or 
        is_bearish_engulfing(prev, latest)
    ) if use_candlestick_confirm else True
    
    sentiment_pass_short = (sentiment_score <= -min_sentiment) if use_sentiment_filter else True

    if is_downtrend and in_structural_resistance and rsi_turning_short and has_bearish_confirm and sentiment_pass_short:
        tp_price = max(below_gaps) if below_gaps else (val if val < price else price * 0.98)
        nearest_resistance_gap = min(above_gaps) if above_gaps else vah
        sl_price = max(latest['high'], nearest_resistance_gap) * 1.0050
        
        risk = sl_price - price
        if risk > 0:
            rr_ratio = (price - tp_price) / risk
            if rr_ratio >= min_rr:
                risk_usd = account_balance * (risk_pct / 100.0)
                position_size = risk_usd / risk
                
                return {
                    "action": "SELL",
                    "pair": normalized_symbol,
                    "direction": "SHORT",
                    "entry": float(price),
                    "sl": float(round(sl_price, 2)),
                    "tp": float(round(tp_price, 2)),
                    "rr": float(round(rr_ratio, 2)),
                    "risk_pct": float(risk_pct),
                    "size": float(round(position_size, 4)),
                    "detected_gaps": gap_levels
                }

    return {"action": "HOLD"}