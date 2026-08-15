import json
import logging
import os
import re
import urllib.request
import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


def normalize_symbol(symbol: str) -> str:
    """Normalizes exchange symbol formats (e.g. BTCUSDT to BTC/USDT)."""
    if not symbol:
        return ""
    s = str(symbol).replace('"', '').replace("'", "").strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/{s[-4:]}"
    return s


def load_symbol_config(symbol: str) -> dict:
    """Loads dynamic strategy parameters from PostgreSQL database with safe fallback defaults[cite: 25]."""
    clean_symbol = normalize_symbol(symbol).replace("/", "").upper()
    db_url = os.getenv("DATABASE_URL")
    
    if db_url:
        db_url = db_url.strip('"').strip("'").strip()
    
    if PSYCOPG2_AVAILABLE and db_url:
        conn = None
        try:
            conn = psycopg2.connect(db_url)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * 
                    FROM strategy_parameters 
                    WHERE UPPER(TRIM(REPLACE(REPLACE(symbol, '"', ''), '''', ''))) = %s
                    ORDER BY updated_at DESC LIMIT 1;
                    """, 
                    (clean_symbol,)
                )
                row = cur.fetchone()
                if row:
                    config = dict(row)
                    config["rsi_thresh"] = float(config.get("rsi_thresh") or 42.0)
                    config["use_rsi_filter"] = bool(config.get("use_rsi_filter", True))
                    config["use_candlestick_confirm"] = bool(config.get("use_candlestick_confirm", True))
                    config["use_adx_filter"] = bool(config.get("use_adx_filter", True))
                    config["adx_period"] = int(config.get("adx_period") or 14)
                    config["adx_threshold"] = float(config.get("adx_threshold") or 20.0)
                    config["max_sl_pct"] = float(config.get("max_sl_pct") or 0.02)
                    return config
        except Exception as e:
            logging.error(f"[load_symbol_config] Database query failed for {clean_symbol}: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    return {
        "tema_period": 200, "rsi_period": 14, "rsi_thresh": 42.0,
        "adx_period": 14, "adx_threshold": 20.0, "use_adx_filter": True,
        "zone_tolerance": 0.0075, "min_sentiment": 0.0, "risk_pct": 1.0,
        "min_rr": 2.0, "vp_detection_pct": 0.07, "use_rsi_filter": True,
        "use_candlestick_confirm": True, "max_sl_pct": 0.02
    }


def get_ai_sentiment_score(text: str) -> float:
    """Evaluates text market sentiment using Gemini 2.5 Flash[cite: 25]."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not GENAI_AVAILABLE:
        return 0.5
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"Analyze the financial sentiment of the following text and return ONLY a single float number between 0.0 (bearish) and 1.0 (bullish):\n\n'{text}'"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        
        match = re.search(r"0\.\d+|1\.0|0|1", response.text.strip())
        if match:
            return max(0.0, min(1.0, float(match.group(0))))
        return 0.5
    except Exception:
        return 0.5


def fetch_binance_and_ccxt_klines(symbol: str = "ETH/USDT", interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """Fetches market data using Binance Public REST API as primary, with CCXT library as fallback."""
    clean = str(symbol).replace('/', '').replace('"', '').replace("'", "").strip().upper()
    
    # 1. Primary: Binance Public REST API
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={clean}&interval={interval}&limit={min(limit, 1000)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw_data = json.loads(resp.read().decode('utf-8'))

        if isinstance(raw_data, list) and len(raw_data) > 0:
            df = pd.DataFrame(raw_data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df[['time', 'open', 'high', 'low', 'close', 'volume']].sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    except Exception as e:
        logging.warning(f"[Binance Public API Error]: {e}. Falling back to CCXT...")

    # 2. Backup/Fallback: CCXT Library
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        formatted_symbol = normalize_symbol(symbol)
        ohlcv = exchange.fetch_ohlcv(formatted_symbol, timeframe=interval, limit=limit)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df[['time', 'open', 'high', 'low', 'close', 'volume']].sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    except Exception as e2:
        logging.error(f"[CCXT Fallback Error]: {e2}")

    return pd.DataFrame()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average[cite: 25]."""
    return series.ewm(span=period, adjust=False).mean()


def calc_tema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Triple Exponential Moving Average (TEMA)[cite: 25]."""
    ema1 = calc_ema(series, period)
    ema2 = calc_ema(ema1, period)
    ema3 = calc_ema(ema2, period)
    return (3 * ema1) - (3 * ema2) + ema3


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)[cite: 25]."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average Directional Index (ADX)[cite: 25]."""
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0.0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx.fillna(0.0)


def compute_volume_profile(df: pd.DataFrame, num_bins: int = 100, lookback_bars: int = 500, va_pct: float = 0.70):
    """Computes POC, VAH, and VAL over lookback window[cite: 25]."""
    if df.empty or len(df) < 10:
        return None, None, None

    df_sub = df.tail(lookback_bars)
    price_min = df_sub['low'].min()
    price_max = df_sub['high'].max()

    if price_min == price_max:
        return None, None, None

    bins = np.linspace(price_min, price_max, num_bins + 1)
    vol_profile = np.zeros(num_bins)

    for _, row in df_sub.iterrows():
        b_idx = np.digitize([row['close']], bins) - 1
        b_idx = np.clip(b_idx, 0, num_bins - 1)[0]
        vol_profile[b_idx] += row['volume']

    poc_idx = np.argmax(vol_profile)
    poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0

    total_vol = vol_profile.sum()
    target_vol = total_vol * va_pct

    current_vol = vol_profile[poc_idx]
    left = poc_idx
    right = poc_idx

    while current_vol < target_vol and (left > 0 or right < num_bins - 1):
        v_left = vol_profile[left - 1] if left > 0 else -1
        v_right = vol_profile[right + 1] if right < num_bins - 1 else -1

        if v_left >= v_right and left > 0:
            left -= 1
            current_vol += vol_profile[left]
        elif right < num_bins - 1:
            right += 1
            current_vol += vol_profile[right]

    val = bins[left]
    vah = bins[right + 1]
    return float(poc), float(vah), float(val)


def calculate_volume_profile_gaps(df: pd.DataFrame, num_bins: int = 100, lookback_bars: int = 500, detection_pct: float = 0.07):
    """Identifies low-volume gaps across the price distribution[cite: 25]."""
    if df.empty or len(df) < 10:
        return []

    df_sub = df.tail(lookback_bars)
    price_min = df_sub['low'].min()
    price_max = df_sub['high'].max()

    if price_min == price_max:
        return []

    bins = np.linspace(price_min, price_max, num_bins + 1)
    vol_profile = np.zeros(num_bins)

    for _, row in df_sub.iterrows():
        b_idx = np.digitize([row['close']], bins) - 1
        b_idx = np.clip(b_idx, 0, num_bins - 1)[0]
        vol_profile[b_idx] += row['volume']

    avg_vol = np.mean(vol_profile)
    threshold = avg_vol * detection_pct
    gap_prices = []

    for i in range(1, num_bins - 1):
        if vol_profile[i] < threshold:
            mid_p = (bins[i] + bins[i + 1]) / 2.0
            gap_prices.append(float(mid_p))

    return gap_prices


def evaluate_signals(
    df: pd.DataFrame,
    symbol: str = "ETH/USDT",
    account_balance: float = 10000.0,
    sentiment_score: float = 0.5,
    **kwargs
) -> dict:
    """Evaluates entry signals based on TEMA, RSI, ADX, and Volume Profile[cite: 25]."""
    if df.empty or len(df) < 30:
        return {"action": "HOLD", "reason": "Insufficient market data"}

    config = load_symbol_config(symbol)
    config.update(kwargs)

    tema_p = int(config.get("tema_period", 200))
    rsi_p = int(config.get("rsi_period", 14))
    rsi_thresh = float(config.get("rsi_thresh", 42.0))
    adx_p = int(config.get("adx_period", 14))
    adx_thresh = float(config.get("adx_threshold", 20.0))
    use_adx = bool(config.get("use_adx_filter", True))
    use_rsi = bool(config.get("use_rsi_filter", True))
    use_candlestick = bool(config.get("use_candlestick_confirm", True))
    risk_pct = float(config.get("risk_pct", 1.0))
    min_rr = float(config.get("min_rr", 2.0))
    max_sl_pct = float(config.get("max_sl_pct", 0.02))

    df_calc = df.copy()
    df_calc['tema'] = calc_tema(df_calc['close'], tema_p) if 'tema' not in df_calc else df_calc['tema']
    df_calc['rsi'] = calc_rsi(df_calc['close'], rsi_p) if 'rsi' not in df_calc else df_calc['rsi']
    df_calc['adx'] = calc_adx(df_calc, adx_p) if 'adx' not in df_calc else df_calc['adx']

    curr = df_calc.iloc[-1]
    prev = df_calc.iloc[-2]

    curr_close = float(curr['close'])
    curr_tema = float(curr['tema'])
    curr_rsi = float(curr['rsi'])
    curr_adx = float(curr['adx'])

    # Guardrail ADX Filter
    if use_adx and curr_adx < adx_thresh:
        return {"action": "HOLD", "reason": f"ADX below threshold ({curr_adx:.2f} < {adx_thresh})"}

    # Bullish Setup
    if curr_close > curr_tema:
        if use_rsi and curr_rsi < rsi_thresh:
            return {"action": "HOLD", "reason": f"RSI momentum low ({curr_rsi:.2f} < {rsi_thresh})"}
        
        if use_candlestick and not (curr['close'] > curr['open'] and prev['close'] <= prev['open']):
            return {"action": "HOLD", "reason": "Awaiting bullish confirmation candle"}

        stop_loss = curr_close * (1.0 - max_sl_pct)
        take_profit = curr_close + ((curr_close - stop_loss) * min_rr)
        risk_amount = account_balance * (risk_pct / 100.0)
        pos_size = risk_amount / (curr_close - stop_loss) if (curr_close - stop_loss) > 0 else 1.0

        return {
            "action": "BUY",
            "direction": "LONG",
            "entry_price": curr_close,
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "position_size": round(pos_size, 4),
            "risk_reward_ratio": min_rr
        }

    # Bearish Setup
    elif curr_close < curr_tema:
        if use_rsi and curr_rsi > (100.0 - rsi_thresh):
            return {"action": "HOLD", "reason": f"RSI momentum high for short ({curr_rsi:.2f})"}

        if use_candlestick and not (curr['close'] < curr['open'] and prev['close'] >= prev['open']):
            return {"action": "HOLD", "reason": "Awaiting bearish confirmation candle"}

        stop_loss = curr_close * (1.0 + max_sl_pct)
        take_profit = curr_close - ((stop_loss - curr_close) * min_rr)
        risk_amount = account_balance * (risk_pct / 100.0)
        pos_size = risk_amount / (stop_loss - curr_close) if (stop_loss - curr_close) > 0 else 1.0

        return {
            "action": "SELL",
            "direction": "SHORT",
            "entry_price": curr_close,
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "position_size": round(pos_size, 4),
            "risk_reward_ratio": min_rr
        }

    return {"action": "HOLD", "reason": "No entry criteria met"}


# Export functions explicitly
fetch_klines = fetch_binance_and_ccxt_klines