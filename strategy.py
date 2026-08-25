import json
import logging
import os
import urllib.request
import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from common import get_db_connection, release_db_connection

load_dotenv()

logger = logging.getLogger("trading_agent")

# Global CCXT Exchange Instances
mexc_exchange = ccxt.mexc({"enableRateLimit": True, "timeout": 10000})
bybit_exchange = ccxt.bybit({"enableRateLimit": True, "timeout": 10000})

def safe_float(val, default: float = 0.0) -> float:
    """Safe conversion helper to prevent float(None) errors across modules."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def normalize_symbol(symbol: str) -> str:
    if not symbol:
        return ""
    s = str(symbol).replace('"', "").replace("'", "").strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/{s[-4:]}"
    return s

def is_empty_data(data) -> bool:
    """Safely checks if data (DataFrame, Series, dict, list, or None) is empty."""
    if data is None:
        return True
    if isinstance(data, (dict, list)):
        return not bool(data)
    if hasattr(data, "empty"):
        return data.empty
    return not bool(data)

def load_symbol_config(symbol: str):
    """Loads strategy configuration parameters for a given symbol from DB with default fallbacks."""
    default_config = {
        "tema_period": 200,
        "rsi_period": 14,
        "rsi_thresh": 42.0,
        "adx_period": 14,
        "adx_threshold": 20.0,
        "use_adx_filter": True,
        "max_sl_pct": 0.02,
        "zone_tolerance": 0.0075,
        "min_sentiment": 0.0,
        "risk_pct": 1.0,
        "min_rr": 2.0,
        "vp_detection_pct": 0.07,
        "use_rsi_filter": True,
        "use_candlestick_confirm": True
    }
    
    clean_symbol = str(symbol).replace("/", "").strip().upper() if symbol else ""
    
    try:
        conn = get_db_connection()
        if not conn:
            return default_config
            
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tema_period, rsi_period, rsi_thresh, adx_period, adx_threshold,
                   use_adx_filter, max_sl_pct, zone_tolerance, min_sentiment,
                   risk_pct, min_rr, vp_detection_pct, use_rsi_filter, use_candlestick_confirm
            FROM strategy_parameters
            WHERE symbol = %s;
        """, (clean_symbol,))
        row = cursor.fetchone()
        cursor.close()
        release_db_connection(conn)

        if row:
            keys = [
                "tema_period", "rsi_period", "rsi_thresh", "adx_period", "adx_threshold",
                "use_adx_filter", "max_sl_pct", "zone_tolerance", "min_sentiment",
                "risk_pct", "min_rr", "vp_detection_pct", "use_rsi_filter", "use_candlestick_confirm"
            ]
            return {keys[i]: row[i] for i in range(len(keys))}
    except Exception as e:
        logger.error(f"Error fetching strategy params for {symbol}: {e}")
        
    return default_config

def fetch_cryptocompare_klines(symbol: str = "XRP/USDT", limit: int = 1000) -> pd.DataFrame:
    try:
        clean = symbol.replace("/", "").upper()
        fsym = clean[:-4] if clean.endswith("USDT") else clean[:-3]
        tsym = "USDT" if clean.endswith("USDT") else "USD"

        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym={tsym}&limit={min(limit, 2000)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("Response") == "Success" and "Data" in data and "Data" in data["Data"]:
            candles = data["Data"]["Data"]
            df = pd.DataFrame(candles)
            df.rename(columns={"time": "open_time", "volumeto": "volume"}, inplace=True)
            df["time"] = pd.to_datetime(df["open_time"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")

            df = df[["time", "open", "high", "low", "close", "volume"]].copy()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(np.float32)

            return df
    except Exception as e:
        logger.debug(f"[CryptoCompare Fallback Error]: {e}")

    return pd.DataFrame()

def fetch_klines(symbol: str, timeframe: str = "1h", limit: int = 600, interval: str = None) -> pd.DataFrame:
    tf = interval if interval is not None else timeframe
    norm_sym = normalize_symbol(symbol)

    conn = get_db_connection()
    if conn:
        try:
            query = """
                SELECT open_time AS timestamp, open, high, low, close, volume 
                FROM candles 
                WHERE symbol = %s 
                ORDER BY open_time DESC 
                LIMIT %s;
            """
            cursor = conn.cursor()
            cursor.execute(query, (symbol, limit))
            rows = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description] if cursor.description else []
            cursor.close()

            if rows:
                df = pd.DataFrame(rows, columns=colnames)
                df = df.sort_values("timestamp").reset_index(drop=True)
                df["time"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                
                df = df[["time", "open", "high", "low", "close", "volume"]].copy()
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(np.float32)
                return df
        except Exception as e:
            logger.warning(f"DB fetch failed for {symbol}: {e}. Initiating exchange fallback chain.")
        finally:
            release_db_connection(conn)

    clean_sym = norm_sym.replace("-", "/").replace("_", "")

    try:
        ohlcv = mexc_exchange.fetch_ohlcv(clean_sym, timeframe=tf, limit=limit)
        if ohlcv and isinstance(ohlcv, list) and len(ohlcv) > 0:
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")
            df = df[["time", "open", "high", "low", "close", "volume"]].copy()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(np.float32)
            return df
    except Exception as e:
        logger.warning(f"[MEXC Fallback Error] for {symbol}: {e}. Trying Bybit...")

    try:
        ohlcv = bybit_exchange.fetch_ohlcv(clean_sym, timeframe=tf, limit=limit)
        if ohlcv and isinstance(ohlcv, list) and len(ohlcv) > 0:
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")
            df = df[["time", "open", "high", "low", "close", "volume"]].copy()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(np.float32)
            return df
    except Exception as e:
        logger.warning(f"[Bybit Fallback Error] for {symbol}: {e}. Trying CryptoCompare...")

    df_fallback = fetch_cryptocompare_klines(symbol=norm_sym, limit=limit)
    if not df_fallback.empty:
        return df_fallback

    logger.error(f"[fetch_klines] All endpoints failed across DB, MEXC, Bybit, and CryptoCompare for symbol: {symbol}")
    return pd.DataFrame()

def calculate_tema(series: pd.Series, length: int = 20, period: int = None, **kwargs) -> pd.Series:
    span_val = period if period is not None else length
    ema1 = series.ewm(span=span_val, adjust=False).mean()
    ema2 = ema1.ewm(span=span_val, adjust=False).mean()
    ema3 = ema2.ewm(span=span_val, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

def calculate_rsi(series: pd.Series, period: int = 14, **kwargs) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan) + 1e-10)
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_atr(df: pd.DataFrame, period: int = 14, **kwargs) -> pd.Series:
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    if is_empty_data(df) or not isinstance(df, pd.DataFrame):
        return pd.Series(dtype=float)
        
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_adx(df: pd.DataFrame, period: int = 14, **kwargs) -> pd.Series:
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    if is_empty_data(df) or not isinstance(df, pd.DataFrame) or len(df) < period + 1:
        return pd.Series(0.0, index=getattr(df, 'index', [0]))

    up_move = df["high"].diff()
    down_move = df["low"].diff().abs()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / (tr.replace(0, np.nan) + 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (tr.replace(0, np.nan) + 1e-10))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    if is_empty_data(df) or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
        
    df = df.copy()
    df["tema"] = calculate_tema(df["close"], length=200)
    df["rsi"] = calculate_rsi(df["close"], period=14)
    df["atr"] = calculate_atr(df, period=14)
    df["adx"] = calculate_adx(df, period=14)
    return df

def check_htf_bias(df_4h: pd.DataFrame) -> str:
    if isinstance(df_4h, dict):
        df_4h = pd.DataFrame(df_4h)
        
    if is_empty_data(df_4h) or not isinstance(df_4h, pd.DataFrame) or len(df_4h) < 20:
        return "NEUTRAL"

    if "tema" not in df_4h.columns:
        df_4h = calculate_indicators(df_4h)

    if is_empty_data(df_4h) or "tema" not in df_4h.columns:
        return "NEUTRAL"

    clean_4h = df_4h.dropna(subset=["tema"])
    if is_empty_data(clean_4h):
        return "NEUTRAL"

    latest_4h = clean_4h.iloc[-1]
    c_4h = safe_float(latest_4h["close"])
    tema_4h = safe_float(latest_4h["tema"])

    if c_4h > tema_4h:
        return "BULLISH"
    elif c_4h < tema_4h:
        return "BEARISH"
    return "NEUTRAL"

def evaluate_signals(
    df: pd.DataFrame,
    df_4h: pd.DataFrame = None,
    symbol: str = "XRP/USDT",
    account_balance: float = 100.0,
    risk_pct: float = 1.0,
    **kwargs
) -> tuple:
    config = load_symbol_config(symbol)
    
    if is_empty_data(df) or not isinstance(df, pd.DataFrame) or len(df) < 30:
        return "HOLD", "Insufficient data length"

    tema_period = int(kwargs.get("tema_period", config.get("tema_period", 200)))
    rsi_period = int(kwargs.get("rsi_period", config.get("rsi_period", 14)))
    rsi_thresh = float(kwargs.get("rsi_thresh", config.get("rsi_thresh", 42.0)))
    adx_period = int(kwargs.get("adx_period", config.get("adx_period", 14)))
    adx_threshold = float(kwargs.get("adx_threshold", config.get("adx_threshold", 20.0)))
    use_adx = bool(kwargs.get("use_adx_filter", config.get("use_adx_filter", True)))

    df_ind = calculate_indicators(df)
    clean_df = df_ind.dropna(subset=["tema", "rsi", "adx", "atr"])
    if is_empty_data(clean_df):
        return "HOLD", "Indicators returned NaN"

    latest = clean_df.iloc[-1]
    c_price = safe_float(latest["close"])
    tema_val = safe_float(latest["tema"])
    rsi_val = safe_float(latest["rsi"])
    adx_val = safe_float(latest["adx"])
    atr_val = safe_float(latest["atr"], default=c_price * 0.01)

    htf_bias = check_htf_bias(df_4h) if df_4h is not None else "BULLISH"

    # Strategy Entry Signal Conditions
    long_cond = (c_price > tema_val) and (rsi_val > rsi_thresh) and (htf_bias in ["BULLISH", "NEUTRAL"])
    if use_adx:
        long_cond = long_cond and (adx_val >= adx_threshold)

    if long_cond:
        stop_loss = c_price - (1.5 * atr_val)
        take_profit = c_price + (3.0 * atr_val)
        risk_usd = account_balance * (risk_pct / 100.0)
        stop_dist = abs(c_price - stop_loss)
        pos_size = (risk_usd / stop_dist) if stop_dist > 0 else 0.0

        return "BUY", {
            "entry_price": c_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": pos_size,
            "position_size_usd": pos_size * c_price,
            "account_balance": account_balance
        }

    return "HOLD", "No signal threshold triggered"

calc_tema = calculate_tema
calc_rsi = calculate_rsi
calc_atr = calculate_atr
calc_adx = calculate_adx