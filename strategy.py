import json
import logging
import os
import re
import urllib.request
import functools
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

# Reduce logging noisiness for root loggers
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("ccxt").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

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
    if s.endswith("USD") and len(s) > 3:
        return f"{s[:-3]}/{s[-3:]}"
    return s


@functools.lru_cache(maxsize=32)
def load_symbol_config(symbol: str) -> dict:
    """Cached config fetcher to prevent high-frequency DB queries on every candle tick."""
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
        except Exception:
            pass
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
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not GENAI_AVAILABLE:
        return 0.5
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"Analyze financial sentiment. Return ONLY a single float between 0.0 and 1.0:\n\n'{text}'"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        match = re.search(r"0\.\d+|1\.0|0|1", response.text.strip())
        return max(0.0, min(1.0, float(match.group(0)))) if match else 0.5
    except Exception:
        return 0.5


def fetch_binance_and_ccxt_klines(symbol: str = "ETH/USDT", interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Fetches market data with quiet failover handling."""
    norm_symbol = normalize_symbol(symbol)
    clean = norm_symbol.replace('/', '').replace('"', '').replace("'", "").strip().upper()

    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={clean}&interval={interval}&limit={min(limit, 1000)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        
        with urllib.request.urlopen(req, timeout=3) as resp:
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
    except Exception:
        pass  # Quiet fallback to Kraken / Coinbase

    exchanges_to_try = [
        ('kraken', ccxt.kraken({'enableRateLimit': True})),
        ('binanceus', ccxt.binanceus({'enableRateLimit': True})),
        ('coinbase', ccxt.coinbase({'enableRateLimit': True}))
    ]

    for ex_name, exchange in exchanges_to_try:
        try:
            ohlcv = exchange.fetch_ohlcv(norm_symbol, timeframe=interval, limit=limit)
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                return df[['time', 'open', 'high', 'low', 'close', 'volume']].sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
        except Exception:
            continue

    return pd.DataFrame()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_tema(series: pd.Series, period: int) -> pd.Series:
    ema1 = calc_ema(series, period)
    ema2 = calc_ema(ema1, period)
    ema3 = calc_ema(ema2, period)
    return (3 * ema1) - (3 * ema2) + ema3


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = np.where(high.diff() > (-low.diff()), high.diff().clip(lower=0), 0.0)
    minus_dm = np.where((-low.diff()) > high.diff(), (-low.diff()).clip(lower=0), 0.0)

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean().fillna(0.0)


def compute_volume_profile(df: pd.DataFrame, num_bins: int = 100, lookback_bars: int = 500, va_pct: float = 0.70):
    if df.empty or len(df) < 10:
        return None, None, None

    df_sub = df.tail(lookback_bars)
    price_min, price_max = df_sub['low'].min(), df_sub['high'].max()

    if price_min == price_max:
        return None, None, None

    bins = np.linspace(price_min, price_max, num_bins + 1)
    vol_profile = np.zeros(num_bins)

    for _, row in df_sub.iterrows():
        b_idx = np.clip(np.digitize([row['close']], bins) - 1, 0, num_bins - 1)[0]
        vol_profile[b_idx] += row['volume']

    poc_idx = np.argmax(vol_profile)
    poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0
    target_vol = vol_profile.sum() * va_pct

    current_vol = vol_profile[poc_idx]
    left = right = poc_idx

    while current_vol < target_vol and (left > 0 or right < num_bins - 1):
        v_left = vol_profile[left - 1] if left > 0 else -1
        v_right = vol_profile[right + 1] if right < num_bins - 1 else -1

        if v_left >= v_right and left > 0:
            left -= 1
            current_vol += vol_profile[left]
        elif right < num_bins - 1:
            right += 1
            current_vol += vol_profile[right]

    return float(poc), float(bins[right + 1]), float(bins[left])


def calculate_volume_profile_gaps(df: pd.DataFrame, num_bins: int = 100, lookback_bars: int = 500, detection_pct: float = 0.07):
    if df.empty or len(df) < 10:
        return []

    df_sub = df.tail(lookback_bars)
    price_min, price_max = df_sub['low'].min(), df_sub['high'].max()
    if price_min == price_max:
        return []

    bins = np.linspace(price_min, price_max, num_bins + 1)
    vol_profile = np.zeros(num_bins)

    for _, row in df_sub.iterrows():
        b_idx = np.clip(np.digitize([row['close']], bins) - 1, 0, num_bins - 1)[0]
        vol_profile[b_idx] += row['volume']

    threshold = np.mean(vol_profile) * detection_pct
    return [float((bins[i] + bins[i + 1]) / 2.0) for i in range(1, num_bins - 1) if vol_profile[i] < threshold]


def evaluate_signals(df: pd.DataFrame, symbol: str = "ETH/USDT", account_balance: float = 10000.0, sentiment_score: float = 0.5, **kwargs) -> dict:
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
    df_calc['tema'] = calc_tema(df_calc['close'], tema_p)
    df_calc['rsi'] = calc_rsi(df_calc['close'], rsi_p)
    df_calc['adx'] = calc_adx(df_calc, adx_p)

    curr, prev = df_calc.iloc[-1], df_calc.iloc[-2]
    curr_close, curr_tema, curr_rsi, curr_adx = float(curr['close']), float(curr['tema']), float(curr['rsi']), float(curr['adx'])

    if use_adx and curr_adx < adx_thresh:
        return {"action": "HOLD", "reason": f"ADX below threshold ({curr_adx:.2f} < {adx_thresh})"}

    if curr_close > curr_tema:
        if use_rsi and curr_rsi < rsi_thresh:
            return {"action": "HOLD", "reason": "RSI momentum low"}
        if use_candlestick and not (curr['close'] > curr['open'] and prev['close'] <= prev['open']):
            return {"action": "HOLD", "reason": "Awaiting confirmation candle"}

        stop_loss = curr_close * (1.0 - max_sl_pct)
        take_profit = curr_close + ((curr_close - stop_loss) * min_rr)
        pos_size = (account_balance * (risk_pct / 100.0)) / (curr_close - stop_loss) if (curr_close - stop_loss) > 0 else 1.0

        return {
            "action": "BUY", "direction": "LONG", "symbol": symbol, "entry_price": curr_close,
            "stop_loss": round(stop_loss, 4), "take_profit": round(take_profit, 4),
            "position_size": round(pos_size, 4), "risk_reward_ratio": min_rr
        }

    elif curr_close < curr_tema:
        if use_rsi and curr_rsi > (100.0 - rsi_thresh):
            return {"action": "HOLD", "reason": "RSI momentum high"}
        if use_candlestick and not (curr['close'] < curr['open'] and prev['close'] >= prev['open']):
            return {"action": "HOLD", "reason": "Awaiting confirmation candle"}

        stop_loss = curr_close * (1.0 + max_sl_pct)
        take_profit = curr_close - ((stop_loss - curr_close) * min_rr)
        pos_size = (account_balance * (risk_pct / 100.0)) / (stop_loss - curr_close) if (stop_loss - curr_close) > 0 else 1.0

        return {
            "action": "SELL", "direction": "SHORT", "symbol": symbol, "entry_price": curr_close,
            "stop_loss": round(stop_loss, 4), "take_profit": round(take_profit, 4),
            "position_size": round(pos_size, 4), "risk_reward_ratio": min_rr
        }

    return {"action": "HOLD", "reason": "No entry criteria met"}


fetch_klines = fetch_binance_and_ccxt_klines