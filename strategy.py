import os
import time
import math
import asyncio
import logging
import requests
import ccxt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from common import get_db_connection, release_db_connection
    HAS_DB = True
except ImportError:
    HAS_DB = False

logger = logging.getLogger("strategy_engine")

EXPECTED_MEXC_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume"
]

# ---------------------------------------------------------------------------
# GLOBAL TIMEFRAME CONSTANTS (RECONCILIATION)
# ---------------------------------------------------------------------------
EXECUTION_TIMEFRAME = "1h"       # Signal generation / entry timeframe
MACRO_TIMEFRAME = "4h"           # HTF confluence timeframe
HTF_TEMA_PERIOD = 200            # TEMA period for 4H macro trend


# ---------------------------------------------------------------------------
# 1. AI FINANCIAL SENTIMENT INTEGRATION
# ---------------------------------------------------------------------------

def get_ai_sentiment_score(text: str, api_key: Optional[str] = None) -> float:
    if not HAS_GENAI:
        logger.warning("Google GenAI SDK not installed. Defaulting sentiment to neutral (0.5).")
        return 0.5

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        logger.debug("No Gemini API key supplied. Defaulting sentiment to neutral (0.5).")
        return 0.5

    try:
        client = genai.Client(api_key=key)
        prompt = (
            "Analyze the following crypto/financial text and rate the sentiment from 0.0 (extremely bearish) "
            "to 1.0 (extremely bullish). Return ONLY a single numeric value float between 0.0 and 1.0.\n\n"
            f"Text: {text}"
        )
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        score_text = response.text.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"Failed to fetch AI sentiment score: {e}")
        return 0.5


# ---------------------------------------------------------------------------
# 2. MULTI-EXCHANGE KLINE DATA FETCHING & FALLBACKS
# ---------------------------------------------------------------------------

def format_ccxt_futures_symbol(symbol: str) -> str:
    """Formats raw symbols to CCXT unified futures symbol format."""
    clean = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
    if not clean.endswith(":USDT"):
        return f"{clean[:-4]}/USDT:USDT" if clean.endswith("USDT") else symbol
    return symbol


def fetch_cryptocompare_klines(symbol: str, interval: str = EXECUTION_TIMEFRAME, limit: int = 400) -> pd.DataFrame:
    try:
        clean_sym = symbol.replace("/", "").replace("_", "").upper()
        if clean_sym.endswith("USDT"):
            fsym = clean_sym[:-4]
            tsym = "USDT"
        elif clean_sym.endswith("USD"):
            fsym = clean_sym[:-3]
            tsym = "USD"
        else:
            fsym = clean_sym
            tsym = "USDT"

        endpoint = "histohour" if "h" in interval.lower() else "histominute"
        url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
        params = {"fsym": fsym, "tsym": tsym, "limit": limit}

        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()

        if data.get("Response") == "Success":
            raw_candles = data["Data"]["Data"]
            df = pd.DataFrame(raw_candles)
            df = df.rename(columns={
                "time": "timestamp", "open": "open", "high": "high",
                "low": "low", "close": "close", "volumeto": "volume"
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.error(f"[{symbol}] CryptoCompare fetch failed: {e}")

    return pd.DataFrame()


def fetch_klines(symbol: str, interval: str = EXECUTION_TIMEFRAME, limit: int = 400) -> pd.DataFrame:
    """
    Tier 1: Bybit Linear Futures (matches execution venue, eliminates basis risk)
    Tier 2: Binance USDT-M Futures
    Tier 3: CryptoCompare fallback
    """
    ccxt_symbol = format_ccxt_futures_symbol(symbol)

    try:
        bybit_ex = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}})
        ohlcv = bybit_ex.fetch_ohlcv(ccxt_symbol, timeframe=interval, limit=limit)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
    except Exception as e:
        logger.warning(f"[{symbol}] Tier 1 (Bybit) kline fetch failed: {e}")

    try:
        binance_ex = ccxt.binanceusdm()
        ohlcv = binance_ex.fetch_ohlcv(symbol.split(':')[0], timeframe=interval, limit=limit)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
    except Exception as e:
        logger.warning(f"[{symbol}] Tier 2 (Binance) kline fetch failed: {e}")

    logger.info(f"[{symbol}] Tier 3 (CryptoCompare) fallback engaged.")
    raw = fetch_cryptocompare_klines(symbol, interval, limit)
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        return raw

    return pd.DataFrame()


async def fetch_klines_async(symbol: str, interval: str = EXECUTION_TIMEFRAME, limit: int = 400) -> pd.DataFrame:
    """Async thread-pool wrapper to prevent blocking the event loop during fetch_klines execution."""
    return await asyncio.to_thread(fetch_klines, symbol, interval, limit)


# ---------------------------------------------------------------------------
# 3. TECHNICAL INDICATORS & INTEGRATED HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_tema(series: pd.Series, period: int) -> pd.Series:
    ema1 = calculate_ema(series, period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)
    return (3 * ema1) - (3 * ema2) + ema3


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or len(df) < period + 1:
        return pd.Series(0.0, index=df.index)

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1.0 / period, adjust=False).mean().fillna(0.0)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or len(df) < period + 1:
        return pd.Series(0.0, index=df.index)

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * (
        pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        / tr_smooth.replace(0, np.nan)
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        / tr_smooth.replace(0, np.nan)
    )

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean().fillna(0.0)
    return adx


def check_candlestick_rejection(last_row: pd.Series, prev_row: pd.Series, direction: str) -> bool:
    """
    Validates structural candlestick rejections at Value Area boundaries (VAL/VAH).
    - Long (VAL Rejection): Bullish Pinbar (long lower wick) OR Bullish Engulfing.
    - Short (VAH Rejection): Bearish Pinbar (long upper wick) OR Bearish Engulfing.
    """
    open_p = float(last_row["open"])
    close_p = float(last_row["close"])
    high_p = float(last_row["high"])
    low_p = float(last_row["low"])
    
    prev_open = float(prev_row["open"])
    prev_close = float(prev_row["close"])

    body = abs(close_p - open_p)
    candle_range = max(high_p - low_p, 1e-8)
    upper_wick = high_p - max(open_p, close_p)
    lower_wick = min(open_p, close_p) - low_p

    if direction in ["BUY", "LONG"]:
        is_pinbar = (lower_wick >= 2.0 * body) and (lower_wick / candle_range >= 0.45)
        is_engulfing = (close_p > open_p) and (close_p > prev_open) and (close_p > float(prev_row["high"]))
        return bool(is_pinbar or is_engulfing)

    elif direction in ["SELL", "SHORT"]:
        is_pinbar = (upper_wick >= 2.0 * body) and (upper_wick / candle_range >= 0.45)
        is_engulfing = (close_p < open_p) and (close_p < prev_open) and (close_p < float(prev_row["low"]))
        return bool(is_pinbar or is_engulfing)

    return False


def check_volatility_expansion(
    df: pd.DataFrame, 
    atr_series: pd.Series, 
    lookback_sma: int = 20, 
    expansion_threshold: float = 1.25
) -> bool:
    """
    Returns True if current relative volatility (ATR / Price) expands 
    beyond `expansion_threshold` (25% higher) relative to its N-period SMA.
    """
    if len(df) < lookback_sma or atr_series.empty:
        return False

    rel_vol = atr_series / df["close"]
    rel_vol_sma = rel_vol.rolling(lookback_sma).mean()

    current_rel_vol = rel_vol.iloc[-1]
    baseline_rel_vol = rel_vol_sma.iloc[-1]

    if np.isnan(baseline_rel_vol) or baseline_rel_vol == 0:
        return False

    return bool(current_rel_vol > (baseline_rel_vol * expansion_threshold))


# ---------------------------------------------------------------------------
# 4. REGIME SWITCHING ENGINE (TREND VS. RANGE)
# ---------------------------------------------------------------------------

def determine_market_regime(adx_val: float, adx_threshold: float = 25.0) -> str:
    """
    Explicit Market Regime Classifier.
    - TRENDING: ADX >= adx_threshold
    - RANGING:  ADX < adx_threshold
    """
    if np.isnan(adx_val):
        return "RANGING"
    return "TRENDING" if adx_val >= adx_threshold else "RANGING"


def determine_limit_entry(
    direction: str,
    current_price: float,
    current_tema: float,
    val: float,
    vah: float,
    near_val: bool,
    near_vah: bool,
    max_entry_offset_pct: float = 0.003
) -> float:
    """
    Calculates entry limit price bounded within a configurable offset threshold 
    of current market price to ensure fill probability during fast breakouts.
    """
    if direction == "LONG":
        if near_val and not np.isnan(val) and val < current_price:
            base_limit = float(val)
        else:
            base_limit = float(min(current_price, current_tema))
        
        min_allowed_price = current_price * (1.0 - max_entry_offset_pct)
        return float(max(base_limit, min_allowed_price))
        
    elif direction == "SHORT":
        if near_vah and not np.isnan(vah) and vah > current_price:
            base_limit = float(vah)
        else:
            base_limit = float(max(current_price, current_tema))
            
        max_allowed_price = current_price * (1.0 + max_entry_offset_pct)
        return float(min(base_limit, max_allowed_price))

    return current_price


def check_ltf_confirmation(
    symbol: str, 
    direction: str, 
    ltf_interval: str = "15m", 
    rsi_period: int = 14,
    df_ltf_override: Optional[pd.DataFrame] = None,
    is_backtest: bool = False
) -> bool:
    if df_ltf_override is not None and len(df_ltf_override) >= 20:
        df_ltf = df_ltf_override.copy()
    elif is_backtest:
        return True
    else:
        # Check if running within an active event loop
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # In async runtime, expect pre-fetched DataFrames via df_ltf_override to prevent blocking
                logger.warning(f"[{symbol}] Synchronous fetch_klines called inside running event loop. Use pre-fetched df_ltf_override.")
                return True
        except RuntimeError:
            pass
            
        df_ltf = fetch_klines(symbol=symbol, interval=ltf_interval, limit=50)

    if df_ltf.empty or len(df_ltf) < 20:
        return True

    df_ltf['ema9'] = calculate_ema(df_ltf['close'], 9)
    df_ltf['ema21'] = calculate_ema(df_ltf['close'], 21)
    df_ltf['rsi'] = calculate_rsi(df_ltf['close'], period=rsi_period)

    last_ltf = df_ltf.iloc[-2]
    prev_ltf = df_ltf.iloc[-3]

    if direction in ["BUY", "LONG"]:
        ema_cross = (last_ltf['ema9'] > last_ltf['ema21'])
        rsi_rising = (last_ltf['rsi'] > prev_ltf['rsi']) and (last_ltf['rsi'] > 40.0)
        return bool(ema_cross or rsi_rising)

    elif direction in ["SELL", "SHORT"]:
        ema_cross = (last_ltf['ema9'] < last_ltf['ema21'])
        rsi_falling = (last_ltf['rsi'] < prev_ltf['rsi']) and (last_ltf['rsi'] < 60.0)
        return bool(ema_cross or rsi_falling)

    return False


def check_volume_confirmation(df: pd.DataFrame, period: int = 20) -> bool:
    if "volume" not in df.columns or len(df) < period:
        return True

    vol_sma = df['volume'].rolling(period).mean().iloc[-1]
    last_volume = df['volume'].iloc[-1]

    return bool(last_volume > vol_sma)


def calculate_staged_targets(
    direction: str,
    entry_price: float,
    stop_loss: float,
    farthest_tp: float,
    tp1_rr: float = 1.5
) -> Dict[str, float]:
    sl_distance = abs(entry_price - stop_loss)

    if direction in ["BUY", "LONG"]:
        tp1 = entry_price + (sl_distance * tp1_rr)
        tp2 = max(farthest_tp, tp1)
    else:
        tp1 = entry_price - (sl_distance * tp1_rr)
        tp2 = min(farthest_tp, tp1)

    return {
        "tp1": round(tp1, 3),
        "tp1_ratio": 0.30,
        "tp2": round(tp2, 7),
        "tp2_ratio": 0.70
    }


def calculate_dynamic_trailing_stop(
    direction: str,
    entry_price: float,
    current_price: float,
    highest_price_since_entry: float,
    lowest_price_since_entry: float,
    current_atr: float,
    atr_mult: float = 2.5,
    tp1_hit: bool = False
) -> float:
    sl_distance = current_atr * atr_mult

    if direction in ["BUY", "LONG"]:
        trailing_sl = highest_price_since_entry - sl_distance
        if tp1_hit:
            trailing_sl = max(trailing_sl, entry_price)
        return float(max(trailing_sl, entry_price - (current_atr * atr_mult)))

    else:
        trailing_sl = lowest_price_since_entry + sl_distance
        if tp1_hit:
            trailing_sl = min(trailing_sl, entry_price)
        return float(min(trailing_sl, entry_price + (current_atr * atr_mult)))


def check_time_based_invalidation(
    bars_in_trade: int, 
    max_bars: int = 24, 
    pnl_pct: float = 0.0
) -> bool:
    if bars_in_trade >= max_bars and pnl_pct < 0.01:
        return True
    return False


# ---------------------------------------------------------------------------
# 5. HIGH-PERFORMANCE VECTORIZED VOLUME PROFILE & LIQUIDITY GAPS
# ---------------------------------------------------------------------------

def get_calibrated_lookback(timeframe: str, target_4h_bars: int = 600) -> int:
    tf = str(timeframe).lower().strip()
    if tf in ["1m", "1min"]:
        return target_4h_bars * 240
    elif tf in ["5m", "5min"]:
        return target_4h_bars * 48
    elif tf in ["15m", "15min"]:
        return target_4h_bars * 16
    elif tf in ["30m", "30min"]:
        return target_4h_bars * 8
    elif tf in ["1h", "60m"]:
        return target_4h_bars * 4
    elif tf in ["4h", "240m"]:
        return target_4h_bars
    elif tf in ["1d", "1day"]:
        return max(target_4h_bars // 6, 50)
    return target_4h_bars


def compute_volume_profile(
    df: pd.DataFrame, 
    num_bins: int = 100, 
    lookback_bars: int = 600, 
    va_pct: float = 0.70,
    timeframe: str = EXECUTION_TIMEFRAME
):
    if df.empty or len(df) < 5 or "volume" not in df.columns:
        return np.nan, np.nan, np.nan

    calibrated_bars = get_calibrated_lookback(timeframe, target_4h_bars=lookback_bars)
    df_range = df.tail(min(len(df), calibrated_bars))

    lows = df_range["low"].values
    highs = df_range["high"].values
    vols = df_range["volume"].values

    pLST, pHST = float(np.min(lows)), float(np.max(highs))
    if pLST >= pHST or np.isnan(pLST) or np.isnan(pHST):
        return np.nan, np.nan, np.nan

    pSTP = (pHST - pLST) / num_bins
    if pSTP <= 0:
        return np.nan, np.nan, np.nan

    vD_vt = np.zeros(num_bins, dtype=np.float64)
    ranges = np.maximum(highs - lows, 1e-8)

    sSI = np.maximum(np.floor((lows - pLST) / pSTP).astype(int), 0)
    eSI = np.minimum(np.floor((highs - pLST) / pSTP).astype(int), num_bins - 1)

    for i in range(len(df_range)):
        lL, lH, lV, lR = lows[i], highs[i], vols[i], ranges[i]
        start_b, end_b = sSI[i], eSI[i]

        for pLI in range(start_b, end_b + 1):
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
    poc = round(pLST + (pcL + 0.5) * pSTP, 2)

    ttV = max(np.sum(vD_vt), 1e-10) * va_pct
    va = vD_vt[pcL]
    laP, lbP = pcL, pcL
    iter_count = 0

    while va < ttV and iter_count < num_bins * 2:
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

    vaH = round(pLST + (laP + 1.0) * pSTP, 2)
    vaL = round(pLST + (lbP + 0.0) * pSTP, 2)

    return float(poc), float(vaH), float(vaL)


def calculate_volume_profile_gaps(
    df: pd.DataFrame,
    num_bins: int = 100,
    lookback_bars: int = 600,
    detection_pct: float = 0.07,
    cached_vp: Optional[Dict[str, Any]] = None,
    timeframe: str = EXECUTION_TIMEFRAME
) -> Dict[str, Any]:
    if cached_vp is not None:
        current_price = float(df.iloc[-1]["close"])
        gaps = cached_vp.get("unique_gaps", [])
        return {
            "poc": cached_vp["poc"],
            "vah": cached_vp["vah"],
            "val": cached_vp["val"],
            "overhead_gaps": sorted([g for g in gaps if g > current_price]),
            "underneath_gaps": sorted([g for g in gaps if g < current_price], reverse=True)
        }

    empty_res = {"poc": np.nan, "vah": np.nan, "val": np.nan, "overhead_gaps": [], "underneath_gaps": []}
    if df.empty or "volume" not in df.columns:
        return empty_res

    calibrated_bars = get_calibrated_lookback(timeframe, target_4h_bars=lookback_bars)
    poc, vah, val = compute_volume_profile(
        df, num_bins=num_bins, lookback_bars=lookback_bars, timeframe=timeframe
    )

    df_range = df.tail(min(len(df), calibrated_bars))
    lows, highs, vols = df_range["low"].values, df_range["high"].values, df_range["volume"].values
    pLST, pHST = float(np.min(lows)), float(np.max(highs))

    if pLST >= pHST or np.isnan(pLST) or np.isnan(pHST):
        return empty_res

    pSTP = (pHST - pLST) / num_bins
    if pSTP <= 0:
        return empty_res

    vD_vt = np.zeros(num_bins, dtype=np.float64)
    ranges = np.maximum(highs - lows, 1e-8)
    sSI = np.maximum(np.floor((lows - pLST) / pSTP).astype(int), 0)
    eSI = np.minimum(np.floor((highs - pLST) / pSTP).astype(int), num_bins - 1)

    for i in range(len(df_range)):
        for pLI in range(sSI[i], eSI[i] + 1):
            pL = pLST + pLI * pSTP
            vPOR = 1.0 if (lows[i] >= pL and highs[i] <= pL + pSTP) else (pSTP / ranges[i])
            vD_vt[pLI] += vols[i] * max(vPOR, 0.0)

    noN = max(int(num_bins * detection_pct), 1)
    
    gap_prices = []
    for i in range(noN, num_bins - noN):
        left_window = vD_vt[i - noN : i]
        right_window = vD_vt[i + 1 : i + 1 + noN]
        
        if len(left_window) == noN and len(right_window) == noN:
            if np.all(vD_vt[i] < left_window) and np.all(vD_vt[i] < right_window):
                gap_price = round(pLST + (i + 0.5) * pSTP, 2)
                gap_prices.append(gap_price)

    unique_gaps = list(dict.fromkeys(gap_prices))
    current_price = float(df.iloc[-1]["close"])

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "overhead_gaps": sorted([g for g in unique_gaps if g > current_price]),
        "underneath_gaps": sorted([g for g in unique_gaps if g < current_price], reverse=True),
        "unique_gaps": unique_gaps
    }


# ---------------------------------------------------------------------------
# 6. SYMBOL CONFIGURATION & PARAMETERS
# ---------------------------------------------------------------------------

def normalize_symbol(symbol: str) -> str:
    if not symbol:
        return ""
    s = str(symbol).replace('"', "").replace("'", "").strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/{s[-4:]}"
    return s


SYMBOL_PARAMETER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # High Beta / Mid-Caps
    "CAKE/USDT": {"adx_threshold": 25.0, "atr_mult": 3.5, "min_rr": 3.5},
    "UNI/USDT":  {"adx_threshold": 24.0, "atr_mult": 3.2, "min_rr": 3.0},
    "SOL/USDT":  {"adx_threshold": 27.0, "atr_mult": 4.0, "min_rr": 3.5},
    "ARB/USDT":  {"adx_threshold": 23.0, "atr_mult": 3.2, "min_rr": 3.0},
    
    # Mega-Caps & Lower Volatility Pairs
    "ETH/USDT":  {"adx_threshold": 28.0, "atr_mult": 4.2, "min_rr": 2.5, "zone_tolerance": 0.015},
    "BNB/USDT":  {"adx_threshold": 28.0, "atr_mult": 4.5, "min_rr": 2.5, "zone_tolerance": 0.015},
    "AAVE/USDT": {"adx_threshold": 26.0, "atr_mult": 4.0, "min_rr": 3.0},
    "XRP/USDT":  {"adx_threshold": 27.0, "atr_mult": 4.0, "min_rr": 3.0},
}


def load_symbol_config(symbol: str) -> Dict[str, Any]:
    formatted_symbol = normalize_symbol(symbol)
    raw_symbol = formatted_symbol.replace("/", "").replace(":", "").replace("_", "").strip().upper()

    default_config = {
        "tema_period": 200, 
        "rsi_period": 14, 
        "rsi_thresh": 42.0,
        "adx_period": 14, 
        "adx_threshold": 25.0,            
        "use_adx_filter": True, 
        "use_rsi_filter": True, 
        "use_candlestick_confirm": True,
        "zone_tolerance": 0.0075,           
        "max_sl_pct": 0.02, 
        "min_sentiment": 0.0,
        "min_rr": 3.5,                    
        "risk_pct": 1.0, 
        "vp_detection_pct": 0.07, 
        "lookback_bars": 600,
        "vp_va_pct": 0.70, 
        "atr_period": 14, 
        "atr_mult": 2.5,                  
        "atr_long_period": 100, 
        "atr_ratio_thresh": 0.85,         
        "use_atr_sl": True, 
        "disable_htf": False, 
        "spot_only": False
    }

    # Apply symbol-specific parameter overrides to default_config
    sym_override = (
        SYMBOL_PARAMETER_DEFAULTS.get(formatted_symbol) 
        or SYMBOL_PARAMETER_DEFAULTS.get(raw_symbol)
    )
    if sym_override:
        default_config.update(sym_override)

    # Database retrieval logic with slash and raw symbol matching
    if HAS_DB and os.getenv("SKIP_DB", "0") != "1":
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * 
                    FROM strategy_parameters 
                    WHERE UPPER(TRIM(REPLACE(REPLACE(REPLACE(symbol, '"', ''), '''', ''), '/', ''))) IN (%s, %s)
                       OR UPPER(TRIM(symbol)) IN (%s, %s)
                    ORDER BY updated_at DESC LIMIT 1;
                """, (raw_symbol, formatted_symbol.replace("/", ""), formatted_symbol, raw_symbol))
                row = cursor.fetchone()

                if row:
                    colnames = [desc[0] for desc in cursor.description] if cursor.description else []
                    cursor.close()
                    config = dict(zip(colnames, row)) if isinstance(row, tuple) else dict(row)

                    # Merge DB values over default_config (which contains symbol overrides)
                    for key, val in config.items():
                        if val is not None and key in default_config:
                            default_config[key] = type(default_config[key])(val)

                    return default_config
                cursor.close()
            except Exception as e:
                logger.error(f"[load_symbol_config] DB query failed for {formatted_symbol}/{raw_symbol}: {e}")
            finally:
                release_db_connection(conn)

    return default_config


# ---------------------------------------------------------------------------
# 7. HARMONIZED MULTI-DIRECTIONAL EVALUATION ENGINE
# ---------------------------------------------------------------------------

def evaluate_signals(
    df: pd.DataFrame,
    symbol: str,
    account_balance: float = 100.0,
    tema_period: Optional[int] = None,
    rsi_period: Optional[int] = None,
    rsi_thresh: Optional[float] = None,
    adx_period: Optional[int] = None,
    adx_threshold: Optional[float] = None,
    use_adx_filter: Optional[bool] = None,
    use_rsi_filter: Optional[bool] = None,
    use_candlestick_confirm: Optional[bool] = None,
    zone_tolerance: Optional[float] = None,
    max_sl_pct: Optional[float] = None,
    min_sentiment: Optional[float] = None,
    min_rr: Optional[float] = None,
    risk_pct: Optional[float] = None,
    atr_period: Optional[int] = None,
    atr_mult: Optional[float] = None,
    use_atr_sl: Optional[bool] = None,
    disable_htf: Optional[bool] = None,
    spot_only: Optional[bool] = None,
    sentiment_score: Optional[float] = None,
    cached_vp: Optional[Dict[str, Any]] = None,
    is_backtest: bool = False,
    timeframe: str = EXECUTION_TIMEFRAME
) -> Dict[str, Any]:
    no_signal = {
        "action": "HOLD", "symbol": symbol, "direction": "NONE",
        "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
        "atr": 0.0, "position_size": 0.0, "reason": "No condition met"
    }

    sym_cfg = load_symbol_config(symbol)

    tema_period = int(tema_period) if tema_period is not None else int(sym_cfg["tema_period"])
    rsi_period = int(rsi_period) if rsi_period is not None else int(sym_cfg["rsi_period"])
    rsi_thresh = float(rsi_thresh) if rsi_thresh is not None else float(sym_cfg["rsi_thresh"])
    adx_period = int(adx_period) if adx_period is not None else int(sym_cfg["adx_period"])
    adx_threshold = float(adx_threshold) if adx_threshold is not None else float(sym_cfg["adx_threshold"])
    zone_tolerance = float(zone_tolerance) if zone_tolerance is not None else float(sym_cfg["zone_tolerance"])
    max_sl_pct = float(max_sl_pct) if max_sl_pct is not None else float(sym_cfg["max_sl_pct"])
    min_sentiment = float(min_sentiment) if min_sentiment is not None else float(sym_cfg["min_sentiment"])
    min_rr = float(min_rr) if min_rr is not None else float(sym_cfg["min_rr"])
    risk_pct = float(risk_pct) if risk_pct is not None else float(sym_cfg["risk_pct"])
    atr_period = int(atr_period) if atr_period is not None else int(sym_cfg["atr_period"])
    atr_mult = float(atr_mult) if atr_mult is not None else float(sym_cfg["atr_mult"])

    use_adx_filter = use_adx_filter if use_adx_filter is not None else sym_cfg["use_adx_filter"]
    use_rsi_filter = use_rsi_filter if use_rsi_filter is not None else sym_cfg["use_rsi_filter"]
    use_candlestick_confirm = use_candlestick_confirm if use_candlestick_confirm is not None else sym_cfg["use_candlestick_confirm"]
    use_atr_sl = use_atr_sl if use_atr_sl is not None else sym_cfg["use_atr_sl"]
    disable_htf = disable_htf if disable_htf is not None else sym_cfg["disable_htf"]
    spot_only = spot_only if spot_only is not None else sym_cfg.get("spot_only", False)

    lookback_bars = int(sym_cfg.get("lookback_bars", 600))
    vp_detection_pct = float(sym_cfg.get("vp_detection_pct", 0.07))

    if df.empty or len(df) < max(tema_period, 50):
        no_signal["reason"] = "Insufficient data rows"
        return no_signal

    if not check_volume_confirmation(df, period=20):
        no_signal["reason"] = "Volume Gate: Trigger candle volume below 20-SMA volume"
        return no_signal

    tema_series = df["tema"] if "tema" in df.columns else calculate_tema(df["close"], tema_period)
    rsi_series = df["rsi"] if "rsi" in df.columns else calculate_rsi(df["close"], rsi_period)
    adx_series = df["adx"] if "adx" in df.columns else calculate_adx(df, adx_period)
    atr_series = df["atr"] if "atr" in df.columns else calculate_atr(df, atr_period)

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    current_price = float(last_row["close"])
    current_tema = float(tema_series.iloc[-1])
    current_rsi = float(rsi_series.iloc[-1])
    current_atr = float(atr_series.iloc[-1])
    current_adx = float(adx_series.iloc[-1])

    # Dynamic tolerance adjusting based on relative ATR % (e.g. Low volatility assets like ETH/BNB)
    current_atr_pct = (current_atr / current_price) if current_price > 0 else 0.0
    if current_atr_pct < 0.008:
        zone_tolerance = min(zone_tolerance, 0.015)

    final_sentiment = float(sentiment_score) if sentiment_score is not None else 0.5
    if final_sentiment < min_sentiment:
        no_signal["reason"] = f"Sentiment score ({final_sentiment:.2f}) below threshold ({min_sentiment:.2f})"
        return no_signal

    # Determine Active Market Regime
    effective_adx_threshold = float(adx_threshold)
    market_regime = determine_market_regime(current_adx, adx_threshold=effective_adx_threshold)

    # HTF Macro Confluence Logic
    macro_trend_long = True
    macro_trend_short = True

    if not disable_htf:
        try:
            if "macro_long" in df.columns and "macro_short" in df.columns:
                macro_trend_long = bool(last_row["macro_long"])
                macro_trend_short = bool(last_row["macro_short"])
            else:
                work_df = df.copy()
                if 'timestamp' not in work_df.columns and isinstance(work_df.index, pd.DatetimeIndex):
                    work_df = work_df.reset_index()
                
                if 'timestamp' in work_df.columns:
                    work_df['timestamp'] = pd.to_datetime(work_df['timestamp'])
                    
                    df_htf = work_df.set_index('timestamp').resample('4h').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna().reset_index()

                    if not df_htf.empty and len(df_htf) >= HTF_TEMA_PERIOD + 2:
                        df_htf["tema_htf"] = calculate_tema(df_htf["close"], HTF_TEMA_PERIOD)
                        
                        htf_close_last = float(df_htf.iloc[-2]["close"])
                        htf_tema_last = float(df_htf.iloc[-2]["tema_htf"])
                        htf_tema_prev = float(df_htf.iloc[-3]["tema_htf"])

                        macro_trend_long = (htf_close_last > htf_tema_last) and (htf_tema_last > htf_tema_prev)
                        macro_trend_short = (htf_close_last < htf_tema_last) and (htf_tema_last < htf_tema_prev)
                else:
                    logger.warning(f"[{symbol}] 'timestamp' column missing for HTF resampling.")
        except Exception as e:
            logger.warning(f"[{symbol}] Could not calculate HTF macro confluence: {e}")

    vp_data = calculate_volume_profile_gaps(
        df,
        lookback_bars=lookback_bars,
        detection_pct=vp_detection_pct,
        cached_vp=cached_vp,
        timeframe=timeframe
    )
    poc, vah, val = vp_data["poc"], vp_data["vah"], vp_data["val"]
    overhead_gaps, underneath_gaps = vp_data["overhead_gaps"], vp_data["underneath_gaps"]

    MIN_SL_PCT = 0.002

    upper_tema_zone = current_tema * (1.0 + zone_tolerance)
    lower_tema_zone = current_tema * (1.0 - zone_tolerance)
    near_tema = lower_tema_zone <= current_price <= upper_tema_zone

    near_val = (not np.isnan(val)) and (abs(current_price - val) / current_price <= zone_tolerance)
    near_vah = (not np.isnan(vah)) and (abs(current_price - vah) / current_price <= zone_tolerance)

    near_gap_support = (
        len(underneath_gaps) > 0
        and (current_price - underneath_gaps[0]) / current_price <= zone_tolerance
    )
    near_gap_resistance = (
        len(overhead_gaps) > 0
        and (overhead_gaps[0] - current_price) / current_price <= zone_tolerance
    )

    # -----------------------------------------------------------------------
    # REGIME 1: TRENDING STRATEGY
    # -----------------------------------------------------------------------
    if market_regime == "TRENDING":
        long_candlestick = True
        if use_candlestick_confirm:
            long_candlestick = (
                float(last_row["close"]) > float(last_row["open"])
                or float(last_row["close"]) > float(prev_row["high"])
            )

        long_rsi_valid = (not use_rsi_filter) or (current_rsi >= rsi_thresh)
        long_zone_ok = near_tema or near_val or near_gap_support

        if (
            (current_price > current_tema)
            and long_zone_ok
            and long_rsi_valid
            and long_candlestick
            and macro_trend_long
        ):
            if not check_ltf_confirmation(
                symbol, direction="LONG", ltf_interval="15m", rsi_period=rsi_period, is_backtest=is_backtest
            ):
                no_signal["reason"] = "Trending LTF Gate: 15m momentum cross/RSI not aligned for LONG"
                return no_signal

            entry_price = determine_limit_entry(
                direction="LONG", current_price=current_price, current_tema=current_tema,
                val=val, vah=vah, near_val=near_val, near_vah=near_vah
            )

            if use_atr_sl and current_atr > 0:
                sl_dist = max(current_atr * atr_mult, entry_price * MIN_SL_PCT)
                stop_loss = entry_price - sl_dist
            else:
                sl_anchor = val if (not np.isnan(val) and val < entry_price) else current_tema
                stop_loss = min(sl_anchor * (1.0 - zone_tolerance), entry_price * 0.99)

            max_allowed_sl = entry_price * (1.0 - max_sl_pct)
            if stop_loss < max_allowed_sl:
                stop_loss = max_allowed_sl

            sl_distance = entry_price - stop_loss
            if sl_distance < (entry_price * MIN_SL_PCT):
                return no_signal

            min_tp = entry_price + (sl_distance * min_rr)
            tp_candidates = [float(g) for g in overhead_gaps if float(g) >= min_tp]

            for ref in (vah, poc):
                if ref is not None and not (isinstance(ref, float) and np.isnan(ref)):
                    if float(ref) >= min_tp:
                        tp_candidates.append(float(ref))

            farthest_tp = max(tp_candidates) if tp_candidates else min_tp
            computed_rr = (farthest_tp - entry_price) / sl_distance

            if computed_rr >= min_rr:
                risk_amt = float(account_balance) * (risk_pct / 100.0)
                position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0

                staged_targets = calculate_staged_targets(
                    direction="LONG", entry_price=entry_price, stop_loss=stop_loss,
                    farthest_tp=farthest_tp, tp1_rr=1.5
                )

                return {
                    "action": "BUY", "order_type": "LIMIT", "symbol": symbol, "direction": "LONG",
                    "regime": "TRENDING",
                    "entry_price": float(round(entry_price, 5)),
                    "stop_loss": float(round(stop_loss, 5)),
                    "take_profit": float(round(farthest_tp, 5)),
                    "tp1": staged_targets["tp1"], "tp1_ratio": staged_targets["tp1_ratio"],
                    "tp2": staged_targets["tp2"], "tp2_ratio": staged_targets["tp2_ratio"],
                    "max_bar_duration": 24, "atr": float(round(current_atr, 4)),
                    "rr_ratio": float(computed_rr), "risk_pct": float(risk_pct),
                    "position_size": float(position_size),
                    "reason": f"Trending Long Confluence: ADX={current_adx:.2f} >= {effective_adx_threshold:.1f}, R:R={computed_rr:.2f}"
                }

        # SHORT TRENDING PATH
        if not spot_only:
            short_candlestick = True
            if use_candlestick_confirm:
                short_candlestick = (
                    float(last_row["close"]) < float(last_row["open"])
                    or float(last_row["close"]) < float(prev_row["low"])
                )

            short_rsi_valid = (not use_rsi_filter) or (current_rsi <= (100.0 - rsi_thresh))
            short_zone_ok = near_tema or near_vah or near_gap_resistance

            if (
                (current_price < current_tema)
                and short_zone_ok
                and short_rsi_valid
                and short_candlestick
                and macro_trend_short
            ):
                if not check_ltf_confirmation(
                    symbol, direction="SHORT", ltf_interval="15m", rsi_period=rsi_period, is_backtest=is_backtest
                ):
                    no_signal["reason"] = "Trending LTF Gate: 15m momentum cross/RSI not aligned for SHORT"
                    return no_signal

                entry_price = determine_limit_entry(
                    direction="SHORT", current_price=current_price, current_tema=current_tema,
                    val=val, vah=vah, near_val=near_val, near_vah=near_vah
                )

                if use_atr_sl and current_atr > 0:
                    sl_dist = max(current_atr * atr_mult, entry_price * MIN_SL_PCT)
                    stop_loss = entry_price + sl_dist
                else:
                    sl_anchor = vah if (not np.isnan(vah) and vah > entry_price) else current_tema
                    stop_loss = max(sl_anchor * (1.0 + zone_tolerance), entry_price * 1.01)

                max_allowed_sl = entry_price * (1.0 + max_sl_pct)
                if stop_loss > max_allowed_sl:
                    stop_loss = max_allowed_sl

                sl_distance = stop_loss - entry_price
                if sl_distance < (entry_price * MIN_SL_PCT):
                    return no_signal

                min_tp = entry_price - (sl_distance * min_rr)
                tp_candidates = [float(g) for g in underneath_gaps if float(g) <= min_tp]

                for ref in (val, poc):
                    if ref is not None and not (isinstance(ref, float) and np.isnan(ref)):
                        if float(ref) <= min_tp:
                            tp_candidates.append(float(ref))

                farthest_tp = min(tp_candidates) if tp_candidates else min_tp
                computed_rr = (entry_price - farthest_tp) / sl_distance

                if computed_rr >= min_rr:
                    risk_amt = float(account_balance) * (risk_pct / 100.0)
                    position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0

                    staged_targets = calculate_staged_targets(
                        direction="SHORT", entry_price=entry_price, stop_loss=stop_loss,
                        farthest_tp=farthest_tp, tp1_rr=1.5
                    )

                    return {
                        "action": "SELL", "order_type": "LIMIT", "symbol": symbol, "direction": "SHORT",
                        "regime": "TRENDING",
                        "entry_price": float(round(entry_price, 5)),
                        "stop_loss": float(round(stop_loss, 5)),
                        "take_profit": float(round(farthest_tp, 5)),
                        "tp1": staged_targets["tp1"], "tp1_ratio": staged_targets["tp1_ratio"],
                        "tp2": staged_targets["tp2"], "tp2_ratio": staged_targets["tp2_ratio"],
                        "max_bar_duration": 24, "atr": float(round(current_atr, 4)),
                        "rr_ratio": float(computed_rr), "risk_pct": float(risk_pct),
                        "position_size": float(position_size),
                        "reason": f"Trending Short Confluence: ADX={current_adx:.2f} >= {effective_adx_threshold:.1f}, R:R={computed_rr:.2f}"
                    }

    # -----------------------------------------------------------------------
    # REGIME 2: RANGING MEAN-REVERSION STRATEGY
    # -----------------------------------------------------------------------
    elif market_regime == "RANGING":
        if check_volatility_expansion(df, atr_series, lookback_sma=20, expansion_threshold=1.25):
            no_signal["reason"] = "Ranging Gate: ATR/Price volatility is expanding rapidly"
            return no_signal

        # Mean Reversion Long (Buy near VAL)
        if near_val and (current_rsi < 45.0):
            if not macro_trend_long:
                no_signal["reason"] = "Ranging Long Gate: Suppressed by strongly bearish 4H macro trend"
                return no_signal

            if not check_candlestick_rejection(last_row, prev_row, direction="LONG"):
                no_signal["reason"] = "Ranging Long Gate: Missing bullish pinbar or engulfing rejection at VAL"
                return no_signal

            entry_price = float(current_price)
            stop_loss = entry_price - max(current_atr * atr_mult, entry_price * 0.008)
            sl_distance = entry_price - stop_loss
            farthest_tp = vah if not np.isnan(vah) else (poc if not np.isnan(poc) else entry_price + (sl_distance * 2.0))

            computed_rr = (farthest_tp - entry_price) / max(sl_distance, 1e-6)
            if computed_rr >= min_rr:
                risk_amt = float(account_balance) * (risk_pct / 100.0)
                position_size = round(risk_amt / sl_distance, 6)
                staged_targets = calculate_staged_targets("LONG", entry_price, stop_loss, farthest_tp)

                return {
                    "action": "BUY", "order_type": "LIMIT", "symbol": symbol, "direction": "LONG",
                    "regime": "RANGING",
                    "entry_price": float(round(entry_price, 5)),
                    "stop_loss": float(round(stop_loss, 5)),
                    "take_profit": float(round(farthest_tp, 5)),
                    "tp1": staged_targets["tp1"], "tp1_ratio": staged_targets["tp1_ratio"],
                    "tp2": staged_targets["tp2"], "tp2_ratio": staged_targets["tp2_ratio"],
                    "max_bar_duration": 24, "atr": float(round(current_atr, 4)),
                    "rr_ratio": float(computed_rr), "risk_pct": float(risk_pct),
                    "position_size": float(position_size),
                    "reason": f"Ranging Mean Reversion Long: ADX={current_adx:.2f} < 20.0, Confirmed VAL Rejection, R:R={computed_rr:.2f}"
                }

        # Mean Reversion Short (Sell near VAH)
        if near_vah and (current_rsi > 55.0) and not spot_only:
            if not macro_trend_short:
                no_signal["reason"] = "Ranging Short Gate: Suppressed by strongly bullish 4H macro trend"
                return no_signal

            if not check_candlestick_rejection(last_row, prev_row, direction="SHORT"):
                no_signal["reason"] = "Ranging Short Gate: Missing bearish pinbar or engulfing rejection at VAH"
                return no_signal

            entry_price = float(current_price)
            stop_loss = entry_price + max(current_atr * atr_mult, entry_price * 0.008)
            sl_distance = stop_loss - entry_price
            farthest_tp = val if not np.isnan(val) else (poc if not np.isnan(poc) else entry_price - (sl_distance * 2.0))

            computed_rr = (entry_price - farthest_tp) / max(sl_distance, 1e-6)
            if computed_rr >= min_rr:
                risk_amt = float(account_balance) * (risk_pct / 100.0)
                position_size = round(risk_amt / sl_distance, 6)
                staged_targets = calculate_staged_targets("SHORT", entry_price, stop_loss, farthest_tp)

                return {
                    "action": "SELL", "order_type": "LIMIT", "symbol": symbol, "direction": "SHORT",
                    "regime": "RANGING",
                    "entry_price": float(round(entry_price, 5)),
                    "stop_loss": float(round(stop_loss, 5)),
                    "take_profit": float(round(farthest_tp, 5)),
                    "tp1": staged_targets["tp1"], "tp1_ratio": staged_targets["tp1_ratio"],
                    "tp2": staged_targets["tp2"], "tp2_ratio": staged_targets["tp2_ratio"],
                    "max_bar_duration": 24, "atr": float(round(current_atr, 4)),
                    "rr_ratio": float(computed_rr), "risk_pct": float(risk_pct),
                    "position_size": float(position_size),
                    "reason": f"Ranging Mean Reversion Short: ADX={current_adx:.2f} < 20.0, Confirmed VAH Rejection, R:R={computed_rr:.2f}"
                }

    no_signal["reason"] = f"No setup triggered in regime {market_regime} (ADX: {current_adx:.2f})"
    return no_signal


# ---------------------------------------------------------------------------
# 8. EVENT-DRIVEN SIGNAL GENERATION CLASS
# ---------------------------------------------------------------------------

class StrategyEngine:
    def __init__(self, symbol: str, config: Optional[Any] = None, event_bus: Optional[Any] = None):
        self.symbol = symbol
        self.config = config
        self.event_bus = event_bus

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None,
        position_side: str = "LONG",
        params: Optional[dict] = None
    ) -> Dict[str, Any]:
        if df.empty or len(df) < 50:
            return {"signal": "HOLD", "action": "HOLD"}

        params = params or {}
        
        evaluated_res = evaluate_signals(
            df=df,
            symbol=self.symbol,
            account_balance=float(params.get("account_balance", 100.0)),
            adx_threshold=float(params.get("adx_threshold", 25.0)),
            rsi_thresh=float(params.get("rsi_thresh", 42.0)),
            min_rr=float(params.get("min_rr", getattr(self.config, "MIN_RR", 3.5) if self.config else 3.5)),
            atr_mult=float(params.get("atr_mult", getattr(self.config, "ATR_MULT", 2.5) if self.config else 2.5)),
            atr_period=int(params.get("atr_period", getattr(self.config, "ATR_PERIOD", 14) if self.config else 14)),
            risk_pct=float(params.get("risk_pct", 1.0)),
            spot_only=(position_side.upper() == "LONG_ONLY")
        )

        action = evaluated_res.get("action", "HOLD")
        evaluated_res["signal"] = action
        evaluated_res["side"] = evaluated_res.get("direction", "NONE")

        if action != "HOLD" and getattr(self, "event_bus", None) and hasattr(self.event_bus, "publish"):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.event_bus.publish("TRADE_SIGNAL", evaluated_res))
            except Exception as e:
                logger.warning(f"Failed to publish signal to event bus: {e}")

        return evaluated_res


# ---------------------------------------------------------------------------
# 9. CONVENIENCE: FULL EVALUATION WRAPPERS
# ---------------------------------------------------------------------------

def generate_live_signal(
    symbol: str,
    account_balance: float = 100.0,
    execution_timeframe: str = EXECUTION_TIMEFRAME,
    limit: int = 600,
    **kwargs
) -> Dict[str, Any]:
    df = fetch_klines(symbol=symbol, interval=execution_timeframe, limit=limit)
    if df.empty:
        return {
            "action": "HOLD", "symbol": symbol, "direction": "NONE",
            "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
            "atr": 0.0, "position_size": 0.0, "reason": "Failed to fetch klines"
        }
    return evaluate_signals(df, symbol, account_balance=account_balance, timeframe=execution_timeframe, **kwargs)


async def generate_live_signal_async(
    symbol: str,
    account_balance: float = 100.0,
    execution_timeframe: str = EXECUTION_TIMEFRAME,
    limit: int = 600,
    **kwargs
) -> Dict[str, Any]:
    """Async variant of generate_live_signal for FastAPI and event-driven runtime loops."""
    df = await fetch_klines_async(symbol=symbol, interval=execution_timeframe, limit=limit)
    if df.empty:
        return {
            "action": "HOLD", "symbol": symbol, "direction": "NONE",
            "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
            "atr": 0.0, "position_size": 0.0, "reason": "Failed to fetch klines"
        }
    return evaluate_signals(df, symbol, account_balance=account_balance, timeframe=execution_timeframe, **kwargs)