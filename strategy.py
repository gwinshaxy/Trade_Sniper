import json
import logging
import os
import re
import urllib.request
import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from common import get_db_connection, release_db_connection

load_dotenv()

logger = logging.getLogger("trading_agent")

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# Global CCXT Exchange Instances (Reused across calls for rate limiting)
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
    
    # Strip slashes to ensure "SOL/USDT" matches "SOLUSDT" in the database
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


def fetch_cryptocompare_klines(
    symbol: str = "XRP/USDT", limit: int = 1000
) -> pd.DataFrame:
    """Tertiary fallback fetching hourly OHLCV data directly from CryptoCompare REST API."""
    try:
        clean = symbol.replace("/", "").upper()
        fsym = clean[:-4] if clean.endswith("USDT") else clean[:-3]
        tsym = "USDT" if clean.endswith("USDT") else "USD"

        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym={tsym}&limit={min(limit, 2000)}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if (
            data.get("Response") == "Success"
            and "Data" in data
            and "Data" in data["Data"]
        ):
            candles = data["Data"]["Data"]
            df = pd.DataFrame(candles)
            df.rename(
                columns={"time": "open_time", "volumeto": "volume"},
                inplace=True,
            )
            df["time"] = pd.to_datetime(df["open_time"], unit="s").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.debug(f"[CryptoCompare Fallback Error]: {e}")

    return pd.DataFrame()


def fetch_klines(
    symbol: str, timeframe: str = "1h", limit: int = 600, interval: str = None
) -> pd.DataFrame:
    """Fetches candlestick data using DB -> MEXC -> Bybit -> CryptoCompare fallback chain."""
    tf = interval if interval is not None else timeframe
    norm_sym = normalize_symbol(symbol)

    # 1. PRIMARY: Database Candle Cache
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
            colnames = (
                [desc[0] for desc in cursor.description]
                if cursor.description
                else []
            )
            cursor.close()

            if rows:
                df = pd.DataFrame(rows, columns=colnames)
                df = df.sort_values("timestamp").reset_index(drop=True)
                df["time"] = pd.to_datetime(df["timestamp"]).dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].apply(lambda x: safe_float(x))
                return df
        except Exception as e:
            logger.warning(
                f"DB fetch failed for {symbol}: {e}. Initiating exchange fallback chain."
            )
        finally:
            release_db_connection(conn)

    clean_sym = norm_sym.replace("-", "/").replace("_", "")

    # 2. SECONDARY: MEXC (CCXT)
    try:
        ohlcv = mexc_exchange.fetch_ohlcv(clean_sym, timeframe=tf, limit=limit)
        if ohlcv and isinstance(ohlcv, list) and len(ohlcv) > 0:
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].apply(lambda x: safe_float(x))
            return df
    except Exception as e:
        logger.warning(
            f"[MEXC Fallback Error] for {symbol}: {e}. Trying Bybit..."
        )

    # 3. TERTIARY: Bybit (CCXT)
    try:
        ohlcv = bybit_exchange.fetch_ohlcv(
            clean_sym, timeframe=tf, limit=limit
        )
        if ohlcv and isinstance(ohlcv, list) and len(ohlcv) > 0:
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].apply(lambda x: safe_float(x))
            return df
    except Exception as e:
        logger.warning(
            f"[Bybit Fallback Error] for {symbol}: {e}. Trying CryptoCompare..."
        )

    # 4. QUATERNARY: CryptoCompare REST API
    df_fallback = fetch_cryptocompare_klines(symbol=norm_sym, limit=limit)
    if not df_fallback.empty:
        return df_fallback

    logger.error(
        f"[fetch_klines] All endpoints failed across DB, MEXC, Bybit, and CryptoCompare for symbol: {symbol}"
    )
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
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def calculate_atr(df, period: int = 14, **kwargs) -> pd.Series:
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    if is_empty_data(df) or not isinstance(df, pd.DataFrame):
        return pd.Series(dtype=float)
        
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(df, period: int = 14, **kwargs) -> pd.Series:
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    if is_empty_data(df) or not isinstance(df, pd.DataFrame) or len(df) < period + 1:
        return pd.Series(0.0, index=getattr(df, 'index', [0]))

    up_move = df["high"].diff()
    down_move = df["low"].diff().abs()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def compute_volume_profile(df, num_bins: int = 100, lookback_bars: int = 600, va_pct: float = 0.70):
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    if is_empty_data(df) or not isinstance(df, pd.DataFrame) or "volume" not in df.columns or "high" not in df.columns or "low" not in df.columns:
        return np.nan, np.nan, np.nan

    df_range = df.tail(lookback_bars).copy()
    if is_empty_data(df_range):
        return np.nan, np.nan, np.nan

    pLST, pHST = safe_float(df_range["low"].min()), safe_float(df_range["high"].max())
    if pLST >= pHST or np.isnan(pLST) or np.isnan(pHST):
        return np.nan, np.nan, np.nan

    pSTP = (pHST - pLST) / num_bins
    if pSTP <= 0:
        return np.nan, np.nan, np.nan

    vD_vt = np.zeros(num_bins)

    for _, row in df_range.iterrows():
        lL, lH, lV = safe_float(row["low"]), safe_float(row["high"]), safe_float(row["volume"])
        lR = max(lH - lL, 1e-8)

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

            vD_vt[pLI] += lV * max(vPOR, 0.0)

    pcL = int(np.argmax(vD_vt))
    poc = round(pLST + (pcL + 0.5) * pSTP, 4)

    ttV = max(np.sum(vD_vt), 1e-10) * va_pct
    va = vD_vt[pcL]
    laP, lbP = pcL, pcL

    max_iter = num_bins * 2
    iter_count = 0

    while va < ttV and iter_count < max_iter:
        iter_count += 1
        if lbP == 0 and laP == num_bins - 1:
            break

        vaP = vD_vt[laP + 1] if laP < num_bins - 1 else 0.0
        vbP = vD_vt[lbP - 1] if lbP > 0 else 0.0

        if vaP >= vbP:
            va += vaP
            laP += 1
        else:
            va += vbP
            lbP -= 1

    vaH = round(pLST + (laP + 1.0) * pSTP, 4)
    vaL = round(pLST + (lbP + 0.0) * pSTP, 4)

    return float(poc), float(vaH), float(vaL)


def calculate_volume_profile_gaps(df, num_bins: int = 100, lookback_bars: int = 600, detection_pct: float = 0.07) -> list:
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    if is_empty_data(df) or not isinstance(df, pd.DataFrame) or "volume" not in df.columns or "high" not in df.columns or "low" not in df.columns:
        return []

    df_range = df.tail(lookback_bars).copy()
    if is_empty_data(df_range):
        return []

    pLST, pHST = safe_float(df_range["low"].min()), safe_float(df_range["high"].max())
    if pLST >= pHST or np.isnan(pLST) or np.isnan(pHST):
        return []

    pSTP = (pHST - pLST) / num_bins
    if pSTP <= 0:
        return []

    vD_vt = np.zeros(num_bins)

    for _, row in df_range.iterrows():
        lL, lH, lV = safe_float(row["low"]), safe_float(row["high"]), safe_float(row["volume"])
        lR = max(lH - lL, 1e-8)

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

            vD_vt[pLI] += lV * max(vPOR, 0.0)

    noN = max(int(num_bins * detection_pct), 1)
    max_val = np.max(vD_vt)
    tVT = list(vD_vt)

    for _ in range(noN):
        tVT.insert(0, max_val)
        tVT.append(max_val)

    gap_prices = []
    for vn in range(2 * noN, num_bins + 2 * noN):
        uNth = True
        for cVN in range(vn - 2 * noN, vn - noN):
            if tVT[vn - noN] >= tVT[cVN]:
                uNth = False
                break

        lNth = True
        for cVN in range(vn - noN + 1, vn + 1):
            if tVT[vn - noN] >= tVT[cVN]:
                lNth = False
                break

        if uNth and lNth:
            bin_idx = vn - 2 * noN
            gap_price = round(pLST + (bin_idx + 0.5) * pSTP, 4)
            if pLST <= gap_price <= pHST:
                gap_prices.append(gap_price)

    return list(dict.fromkeys(gap_prices))


def calculate_indicators(df) -> pd.DataFrame:
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    if is_empty_data(df) or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
        
    df = df.copy()
    df["tema"] = calculate_tema(df["close"], length=20)
    df["rsi"] = calculate_rsi(df["close"], period=14)
    df["atr"] = calculate_atr(df, period=14)
    df["adx"] = calculate_adx(df, period=14)
    return df


def check_htf_bias(df_4h) -> str:
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
    df,
    df_4h=None,
    symbol: str = "XRP/USDT",
    account_balance: float = 10000.0,
    **kwargs,
) -> tuple:
    config = load_symbol_config(symbol)
    
    # Safely convert inputs to DataFrames
    if isinstance(df, dict):
        df = pd.DataFrame(df) if df else pd.DataFrame()
    if isinstance(df_4h, dict):
        df_4h = pd.DataFrame(df_4h) if df_4h else pd.DataFrame()

    if is_empty_data(df) or not isinstance(df, pd.DataFrame):
        return "HOLD", "Invalid or empty market DataFrame provided"

    tema_period = int(kwargs.get("tema_period", config.get("tema_period", 20)))
    rsi_period = int(kwargs.get("rsi_period", config.get("rsi_period", 14)))
    adx_period = int(kwargs.get("adx_period", config.get("adx_period", 14)))

    if len(df) < max(tema_period, adx_period + 1):
        return "HOLD", "Insufficient historical data for indicator convergence"

    df = df.copy()
    df["tema"] = calculate_tema(df["close"], period=tema_period)
    df["rsi"] = calculate_rsi(df["close"], period=rsi_period)
    df["adx"] = calculate_adx(df, period=adx_period)

    clean_df = df.dropna(subset=["tema", "rsi", "adx"])
    if is_empty_data(clean_df):
        return "HOLD", "Indicators contained NaN values"

    latest = clean_df.iloc[-1]
    current_close = safe_float(latest["close"])
    current_tema = safe_float(latest["tema"])
    current_rsi = safe_float(latest["rsi"])
    current_adx = safe_float(latest["adx"])

    use_adx_filter = bool(kwargs.get("use_adx_filter", config.get("use_adx_filter", True)))
    adx_threshold = safe_float(kwargs.get("adx_threshold", config.get("adx_threshold", 20.0)))

    if use_adx_filter and current_adx < adx_threshold:
        return "HOLD", f"ADX too low ({current_adx:.1f} < {adx_threshold:.1f})"

    # Safely evaluate 4H HTF Bias
    if not is_empty_data(df_4h) and isinstance(df_4h, pd.DataFrame):
        htf_bias = check_htf_bias(df_4h)
    else:
        try:
            df_4h_fetched = fetch_klines(symbol=symbol, timeframe="4h", limit=max(200, tema_period + 10))
            htf_bias = check_htf_bias(df_4h_fetched)
        except Exception:
            htf_bias = "NEUTRAL"

    poc, vah, val = compute_volume_profile(clean_df, lookback_bars=int(config.get("lookback_bars", 600)))
    vp_gaps = calculate_volume_profile_gaps(clean_df, detection_pct=safe_float(config.get("vp_detection_pct", 0.07)))

    rsi_thresh = safe_float(kwargs.get("rsi_thresh", config.get("rsi_thresh", 42.0)))
    use_rsi = bool(kwargs.get("use_rsi_filter", config.get("use_rsi_filter", True)))
    use_candlestick = bool(kwargs.get("use_candlestick_confirm", config.get("use_candlestick_confirm", True)))
    zone_tolerance = safe_float(kwargs.get("zone_tolerance", config.get("zone_tolerance", 0.0075)))
    min_sentiment = safe_float(kwargs.get("min_sentiment", config.get("min_sentiment", 0.0)))
    min_rr = safe_float(kwargs.get("min_rr", config.get("min_rr", 2.0)))
    risk_pct = safe_float(kwargs.get("risk_pct", config.get("risk_pct", 1.0)))
    max_sl_pct = safe_float(kwargs.get("max_sl_pct", config.get("max_sl_pct", 0.02)))

    sentiment_score = safe_float(kwargs.get("sentiment_score", 0.5))
    if sentiment_score < min_sentiment:
        return "HOLD", "Sentiment score below minimum threshold"

    upper_tema_zone = current_tema * (1.0 + zone_tolerance)
    lower_tema_zone = current_tema * (1.0 - zone_tolerance)

    bullish_candlestick = (latest["close"] > latest["open"]) if use_candlestick else True
    bearish_candlestick = (latest["close"] < latest["open"]) if use_candlestick else True

    rsi_long_ok = (current_rsi >= rsi_thresh) if use_rsi else True
    rsi_short_ok = (current_rsi <= (100 - rsi_thresh)) if use_rsi else True

    overhead_gaps = sorted([g for g in vp_gaps if g > current_close])
    underneath_gaps = sorted([g for g in vp_gaps if g < current_close], reverse=True)

    near_tema = lower_tema_zone <= current_close <= upper_tema_zone
    near_val = not np.isnan(val) and abs(current_close - val) / current_close <= zone_tolerance
    near_gap_support = len(underneath_gaps) > 0 and (current_close - underneath_gaps[0]) / current_close <= zone_tolerance

    if (near_tema or near_val or near_gap_support) and rsi_long_ok and bullish_candlestick and htf_bias != "BEARISH":
        sl_anchor = val if (not np.isnan(val) and val < current_close) else current_tema
        raw_stop_loss = min(sl_anchor * (1.0 - zone_tolerance), current_close * 0.99)
        tightest_allowed_sl = current_close * (1.0 - max_sl_pct)
        stop_loss = round(max(raw_stop_loss, tightest_allowed_sl), 5)

        risk_distance = current_close - stop_loss
        if risk_distance <= 0:
            return "HOLD", "Invalid risk distance computed for LONG"

        tp_candidates = overhead_gaps + [x for x in [poc, vah] if not np.isnan(x) and x > current_close]
        min_tp = current_close + (risk_distance * min_rr)
        take_profit = round(max(min(tp_candidates), min_tp) if tp_candidates else min_tp, 5)

        computed_rr = round((take_profit - current_close) / risk_distance, 2)
        risk_amt = account_balance * (risk_pct / 100.0)
        position_size = round(risk_amt / risk_distance, 4)

        return "BUY", {
            "symbol": symbol,
            "direction": "LONG",
            "entry_price": current_close,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": computed_rr,
            "position_size": position_size,
            "reason": "Long signal confirmed with TEMA, ADX trend strength, 4H HTF bias, and Volume Profile support confluence"
        }

    near_vah = not np.isnan(vah) and abs(current_close - vah) / current_close <= zone_tolerance
    near_gap_resistance = len(overhead_gaps) > 0 and (overhead_gaps[0] - current_close) / current_close <= zone_tolerance

    if (near_tema or near_vah or near_gap_resistance) and rsi_short_ok and bearish_candlestick and htf_bias != "BULLISH":
        sl_anchor = vah if (not np.isnan(vah) and vah > current_close) else current_tema
        raw_stop_loss = max(sl_anchor * (1.0 + zone_tolerance), current_close * 1.01)
        tightest_allowed_sl = current_close * (1.0 + max_sl_pct)
        stop_loss = round(min(raw_stop_loss, tightest_allowed_sl), 5)

        risk_distance = stop_loss - current_close
        if risk_distance <= 0:
            return "HOLD", "Invalid risk distance computed for SHORT"

        tp_candidates = underneath_gaps + [x for x in [poc, val] if not np.isnan(x) and x < current_close]
        min_tp = current_close - (risk_distance * min_rr)
        take_profit = round(min(max(tp_candidates), min_tp) if tp_candidates else min_tp, 5)

        computed_rr = round((current_close - take_profit) / risk_distance, 2)
        risk_amt = account_balance * (risk_pct / 100.0)
        position_size = round(risk_amt / risk_distance, 4)

        return "SELL", {
            "symbol": symbol,
            "direction": "SHORT",
            "entry_price": current_close,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": computed_rr,
            "position_size": position_size,
            "reason": "Short signal confirmed with TEMA, ADX trend strength, 4H HTF bias, and Volume Profile resistance confluence"
        }

    return "HOLD", "Conditions not fully aligned"


# Module-level aliases to support external script imports (e.g. ws_monitor.py)
calc_tema = calculate_tema
calc_rsi = calculate_rsi
calc_atr = calculate_atr
calc_adx = calculate_adx