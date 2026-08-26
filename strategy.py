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
mexc_exchange = ccxt.mexc({"enableRateLimit": True, "timeout": 10000, "options": {"defaultType": "spot"}})
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
    """Normalizes exchange trading pair string format (e.g., XRPUSDT or XRP-USDT -> XRP/USDT)."""
    if not symbol:
        return ""
    s = str(symbol).replace('"', "").replace("'", "").replace("-", "").replace("_", "").strip().upper()
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


def load_symbol_config(symbol: str) -> dict:
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
    
    clean_symbol = str(symbol).replace("/", "").replace("-", "").strip().upper() if symbol else ""
    
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
        return default_config
    except Exception as e:
        logger.error(f"Error loading symbol config for {symbol}: {e}")
        return default_config


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 500, timeframe: str = None) -> pd.DataFrame:
    """Fetches real-time candlestick kline data from MEXC Spot endpoint with fallback handling."""
    active_tf = timeframe or interval
    clean_symbol = normalize_symbol(symbol)

    try:
        raw_klines = mexc_exchange.fetch_ohlcv(clean_symbol, timeframe=active_tf, limit=limit)
        if not raw_klines:
            return pd.DataFrame()

        df = pd.DataFrame(raw_klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch klines from MEXC Spot for {clean_symbol}: {e}")
        return pd.DataFrame()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False).mean()


def calc_tema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Triple Exponential Moving Average (TEMA)."""
    ema1 = calc_ema(series, period)
    ema2 = calc_ema(ema1, period)
    ema3 = calc_ema(ema2, period)
    return (3 * ema1) - (3 * ema2) + ema3


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average Directional Index (ADX) to gauge trend strength."""
    df = df.copy()
    df["up"] = df["high"].diff()
    df["down"] = -df["low"].diff()

    df["+dm"] = np.where((df["up"] > df["down"]) & (df["up"] > 0), df["up"], 0.0)
    df["-dm"] = np.where((df["down"] > df["up"]) & (df["down"] > 0), df["down"], 0.0)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_smooth = df["tr"].ewm(alpha=1/period, adjust=False).mean()
    pos_dm_smooth = df["+dm"].ewm(alpha=1/period, adjust=False).mean()
    neg_dm_smooth = df["-dm"].ewm(alpha=1/period, adjust=False).mean()

    pos_di = 100 * (pos_dm_smooth / tr_smooth.replace(0, np.nan))
    neg_di = 100 * (neg_dm_smooth / tr_smooth.replace(0, np.nan))

    dx = (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.fillna(0.0)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR) for volatility and stop placement."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_volume_profile(df: pd.DataFrame, num_bins: int = 100, lookback_bars: int = 600, va_pct: float = 0.70) -> tuple:
    """Computes Volume Profile metrics matching TradingView DGT implementation."""
    if is_empty_data(df) or len(df) < 10:
        return None, None, None

    slice_df = df.tail(lookback_bars).copy()
    min_p = float(slice_df["low"].min())
    max_p = float(slice_df["high"].max())

    if min_p >= max_p or np.isnan(min_p) or np.isnan(max_p):
        return None, None, None

    pstp = (max_p - min_p) / num_bins
    vol_profile = np.zeros(num_bins)

    for _, row in slice_df.iterrows():
        c_low, c_high, c_vol = float(row["low"]), float(row["high"]), float(row["volume"])
        c_range = max(c_high - c_low, 1e-8)

        s_idx = max(int(np.floor((c_low - min_p) / pstp)), 0)
        e_idx = min(int(np.floor((c_high - min_p) / pstp)), num_bins - 1)

        for i in range(s_idx, e_idx + 1):
            p_level = min_p + i * pstp
            if c_low >= p_level and c_high > p_level + pstp:
                vpor = (p_level + pstp - c_low) / c_range
            elif c_high <= p_level + pstp and c_low < p_level:
                vpor = (c_high - p_level) / c_range
            elif c_low >= p_level and c_high <= p_level + pstp:
                vpor = 1.0
            else:
                vpor = pstp / c_range

            vol_profile[i] += c_vol * vpor

    poc_idx = int(np.argmax(vol_profile))
    poc = min_p + (poc_idx + 0.5) * pstp

    total_vol = max(np.sum(vol_profile), 1e-10)
    target_vol = total_vol * va_pct

    current_vol = vol_profile[poc_idx]
    laP, lbP = poc_idx, poc_idx

    while current_vol < target_vol:
        if lbP == 0 and laP == num_bins - 1:
            break

        vaP = vol_profile[laP + 1] if laP < num_bins - 1 else 0.0
        vbP = vol_profile[lbP - 1] if lbP > 0 else 0.0

        if vaP >= vbP:
            current_vol += vaP
            laP += 1
        else:
            current_vol += vbP
            lbP -= 1

    vah = min_p + (laP + 1.0) * pstp
    val = min_p + (lbP + 0.0) * pstp

    return float(poc), float(vah), float(val)


def calculate_volume_profile_gaps(df: pd.DataFrame, num_bins: int = 100, lookback_bars: int = 600, detection_pct: float = 0.07) -> list:
    """Detects Volume Profile Gaps aligned with Pine Script node detection."""
    if is_empty_data(df) or len(df) < 10:
        return []

    df_range = df.tail(lookback_bars).copy()
    pLST, pHST = float(df_range["low"].min()), float(df_range["high"].max())
    if pLST >= pHST or np.isnan(pLST) or np.isnan(pHST):
        return []

    pSTP = (pHST - pLST) / num_bins
    vt = np.zeros(num_bins)

    for _, row in df_range.iterrows():
        lL, lH, lV = float(row["low"]), float(row["high"]), float(row["volume"])
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
            vt[pLI] += lV * vPOR

    noN = int(num_bins * detection_pct)
    if noN < 1:
        noN = 1

    max_val = np.max(vt) if len(vt) > 0 else 1.0
    padded_vt = np.concatenate(([max_val] * noN, vt, [max_val] * noN))

    gaps = []
    for vn in range(2 * noN, num_bins + 2 * noN):
        curr_vol = padded_vt[vn - noN]

        uNth = all(curr_vol < padded_vt[cVN] for cVN in range(vn - 2 * noN, vn - noN))
        lNth = all(curr_vol < padded_vt[cVN] for cVN in range(vn - noN + 1, vn + 1))

        if uNth and lNth:
            gap_bin_idx = vn - 2 * noN
            gap_price = pLST + (gap_bin_idx + 0.5) * pSTP
            gaps.append(float(gap_price))

    return gaps

def check_candlestick_pattern(df: pd.DataFrame) -> bool:
    """Checks for bullish candlestick confirmation (Bullish Engulfing or Hammer)."""
    if len(df) < 2:
        return True

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    curr_open, curr_close = safe_float(curr["open"]), safe_float(curr["close"])
    curr_high, curr_low = safe_float(curr["high"]), safe_float(curr["low"])
    prev_open, prev_close = safe_float(prev["open"]), safe_float(prev["close"])

    # Bullish Engulfing
    if prev_close < prev_open and curr_close > curr_open and curr_close >= prev_open and curr_open <= prev_close:
        return True

    # Hammer
    body = abs(curr_close - curr_open)
    lower_wick = min(curr_open, curr_close) - curr_low
    if body > 0 and lower_wick >= 2 * body:
        return True

    return curr_close > curr_open


def evaluate_signals(df: pd.DataFrame, symbol: str, risk_pct: float = 1.0, account_balance: float = 100.0, allocated_slot_usd: float = 25.0) -> tuple:
    """Evaluates indicators and returns Spot Buy signals with position metrics. Strictly bypasses Sell/Short signals."""
    if is_empty_data(df) or len(df) < 50:
        return "NEUTRAL", "Insufficient data bars"

    cfg = load_symbol_config(symbol)
    
    close_s = df["close"]
    df["tema"] = calc_tema(close_s, period=int(cfg["tema_period"]))
    df["rsi"] = calc_rsi(close_s, period=int(cfg["rsi_period"]))
    df["adx"] = calc_adx(df, period=int(cfg["adx_period"]))
    df["atr"] = calc_atr(df, period=14)

    curr = df.iloc[-1]
    c_price = safe_float(curr["close"])
    c_tema = safe_float(curr["tema"])
    c_rsi = safe_float(curr["rsi"])
    c_adx = safe_float(curr["adx"])
    c_atr = safe_float(curr["atr"], default=c_price * 0.01)

    zone_tol = float(cfg["zone_tolerance"])
    upper_zone = c_tema * (1.0 + zone_tol)

    # Long/Buy Filters
    above_zone = c_price > upper_zone
    rsi_pass = c_rsi >= float(cfg["rsi_thresh"]) if cfg["use_rsi_filter"] else True
    adx_pass = c_adx >= float(cfg["adx_threshold"]) if cfg["use_adx_filter"] else True
    candle_pass = check_candlestick_pattern(df) if cfg["use_candlestick_confirm"] else True

    if above_zone and rsi_pass and adx_pass and candle_pass:
        # Calculate Risk and Target Levels
        stop_dist = max(c_atr * 1.5, c_price * 0.005)
        stop_loss = round(c_price - stop_dist, 5)
        take_profit = round(c_price + (stop_dist * float(cfg["min_rr"])), 5)

        risk_amount_usd = allocated_slot_usd * (risk_pct / 100.0)
        position_size_qty = round(risk_amount_usd / stop_dist, 4)
        position_size_usd = position_size_qty * c_price

        # Cap allocated slot amount
        if position_size_usd > allocated_slot_usd:
            position_size_usd = allocated_slot_usd
            position_size_qty = round(position_size_usd / c_price, 4)

        details = {
            "symbol": normalize_symbol(symbol),
            "entry_price": c_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size_qty,
            "position_size_usd": position_size_usd,
            "account_balance": allocated_slot_usd,
            "risk_reward_ratio": float(cfg["min_rr"]),
            "rsi": c_rsi,
            "adx": c_adx,
            "tema": c_tema
        }
        return "BUY", details

    return "NEUTRAL", f"No signal condition met (Price=${c_price:.4f}, TEMA=${c_tema:.4f}, RSI={c_rsi:.1f}, ADX={c_adx:.1f})"