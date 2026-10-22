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

# ---------------------------------------------------------------------------
# GLOBAL TIMEFRAME CONSTANTS (4H TEMA OVERHAUL)
# ---------------------------------------------------------------------------
EXECUTION_TIMEFRAME = "4h"       # 4-Hour execution timeframe from screenshots
MACRO_TIMEFRAME = "4h"           
HTF_TEMA_PERIOD = 200            # 200 TEMA baseline


# ---------------------------------------------------------------------------
# 1. AI FINANCIAL SENTIMENT INTEGRATION
# ---------------------------------------------------------------------------

def get_ai_sentiment_score(text: str, api_key: Optional[str] = None) -> float:
    if not HAS_GENAI:
        logger.warning("Google GenAI SDK not installed. Defaulting sentiment to neutral (0.5).")
        return 0.5

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return 0.5

    try:
        client = genai.Client(api_key=key)
        prompt = (
            "Analyze the following crypto/financial text and rate sentiment from 0.0 (extremely bearish) "
            "to 1.0 (extremely bullish). Return ONLY a single numeric float between 0.0 and 1.0.\n\n"
            f"Text: {text}"
        )
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return max(0.0, min(1.0, float(response.text.strip())))
    except Exception as e:
        logger.error(f"Failed to fetch AI sentiment score: {e}")
        return 0.5


# ---------------------------------------------------------------------------
# 2. MULTI-EXCHANGE KLINE DATA FETCHING
# ---------------------------------------------------------------------------

def format_ccxt_futures_symbol(symbol: str) -> str:
    clean = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
    if not clean.endswith(":USDT"):
        return f"{clean[:-4]}/USDT:USDT" if clean.endswith("USDT") else symbol
    return symbol


def fetch_klines(symbol: str, interval: str = EXECUTION_TIMEFRAME, limit: int = 500) -> pd.DataFrame:
    ccxt_symbol = format_ccxt_futures_symbol(symbol)
    try:
        bybit_ex = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}})
        ohlcv = bybit_ex.fetch_ohlcv(ccxt_symbol, timeframe=interval, limit=limit)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
    except Exception as e:
        logger.warning(f"[{symbol}] Bybit kline fetch failed: {e}")

    try:
        binance_ex = ccxt.binanceusdm()
        ohlcv = binance_ex.fetch_ohlcv(symbol.split(':')[0], timeframe=interval, limit=limit)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
    except Exception as e:
        logger.warning(f"[{symbol}] Binance kline fetch failed: {e}")

    return pd.DataFrame()


async def fetch_klines_async(symbol: str, interval: str = EXECUTION_TIMEFRAME, limit: int = 500) -> pd.DataFrame:
    return await asyncio.to_thread(fetch_klines, symbol, interval, limit)


# ---------------------------------------------------------------------------
# 3. TECHNICAL INDICATORS & STRUCTURAL HELPERS
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
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or len(df) < period + 1:
        return pd.Series(0.0, index=df.index)
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().fillna(0.0)


def check_candlestick_rejection(last_row: pd.Series, prev_row: pd.Series, direction: str) -> bool:
    open_p, close_p = float(last_row["open"]), float(last_row["close"])
    high_p, low_p = float(last_row["high"]), float(last_row["low"])
    prev_open, prev_close = float(prev_row["open"]), float(prev_row["close"])

    body = abs(close_p - open_p)
    candle_range = max(high_p - low_p, 1e-8)
    upper_wick = high_p - max(open_p, close_p)
    lower_wick = min(open_p, close_p) - low_p

    if direction in ["BUY", "LONG"]:
        is_pinbar = (lower_wick >= 2.0 * body) and (lower_wick / candle_range >= 0.40)
        is_engulfing = (close_p > open_p) and (close_p > prev_open) and (close_p > float(prev_row["high"]))
        is_strong_close = (close_p > open_p) and (body / candle_range >= 0.50)
        return bool(is_pinbar or is_engulfing or is_strong_close)

    elif direction in ["SELL", "SHORT"]:
        is_pinbar = (upper_wick >= 2.0 * body) and (upper_wick / candle_range >= 0.40)
        is_engulfing = (close_p < open_p) and (close_p < prev_open) and (close_p < float(prev_row["low"]))
        is_strong_close = (close_p < open_p) and (body / candle_range >= 0.50)
        return bool(is_pinbar or is_engulfing or is_strong_close)

    return False


def check_volume_spike(df: pd.DataFrame, period: int = 20, multiplier: float = 1.0) -> bool:
    """Verifies that trigger candle volume exceeds the N-period volume SMA."""
    if "volume" not in df.columns or len(df) < period:
        return True
    vol_sma = df['volume'].rolling(period).mean().iloc[-1]
    last_volume = df['volume'].iloc[-1]
    if np.isnan(vol_sma) or vol_sma == 0:
        return True
    return bool(last_volume >= (vol_sma * multiplier))


def find_structural_swing_point(df: pd.DataFrame, direction: str, lookback: int = 5) -> float:
    if df.empty or len(df) < lookback + 2:
        return float(df.iloc[-1]["close"])
    recent_df = df.tail(lookback)
    if direction in ["BUY", "LONG"]:
        return float(recent_df["low"].min())
    else:
        return float(recent_df["high"].max())


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
    atr_mult: float = 3.0,
    tp1_hit: bool = False
) -> float:
    sl_distance = current_atr * atr_mult
    if direction in ["BUY", "LONG"]:
        trailing_sl = highest_price_since_entry - sl_distance
        if tp1_hit:
            trailing_sl = max(trailing_sl, entry_price)
        return float(max(trailing_sl, entry_price - sl_distance))
    else:
        trailing_sl = lowest_price_since_entry + sl_distance
        if tp1_hit:
            trailing_sl = min(trailing_sl, entry_price)
        return float(min(trailing_sl, entry_price + sl_distance))


def check_time_based_invalidation(bars_in_trade: int, max_bars: int = 36, pnl_pct: float = 0.0) -> bool:
    return bool(bars_in_trade >= max_bars and pnl_pct < 0.01)


# ---------------------------------------------------------------------------
# 4. VOLUME PROFILE & GAPS
# ---------------------------------------------------------------------------

def compute_volume_profile(df: pd.DataFrame, num_bins: int = 100, lookback_bars: int = 500, va_pct: float = 0.70):
    if df.empty or len(df) < 5 or "volume" not in df.columns:
        return np.nan, np.nan, np.nan

    df_range = df.tail(min(len(df), lookback_bars))
    lows, highs, vols = df_range["low"].values, df_range["high"].values, df_range["volume"].values
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
        for pLI in range(sSI[i], eSI[i] + 1):
            pL = pLST + pLI * pSTP
            vPOR = 1.0 if (lL >= pL and lH <= pL + pSTP) else (pSTP / lR)
            vD_vt[pLI] += lV * max(vPOR, 0.0)

    pcL = int(np.argmax(vD_vt))
    poc = round(pLST + (pcL + 0.5) * pSTP, 2)

    ttV = max(np.sum(vD_vt), 1e-10) * va_pct
    va = vD_vt[pcL]
    laP, lbP = pcL, pcL
    while va < ttV and (lbP > 0 or laP < num_bins - 1):
        vaP = vD_vt[laP + 1] if laP < num_bins - 1 else 0.0
        vbP = vD_vt[lbP - 1] if lbP > 0 else 0.0
        if vaP >= vbP:
            va += vaP
            laP += 1
        else:
            va += vbP
            lbP -= 1

    return float(poc), float(round(pLST + (laP + 1.0) * pSTP, 2)), float(round(pLST + (lbP + 0.0) * pSTP, 2))


def calculate_volume_profile_gaps(df: pd.DataFrame, num_bins: int = 100, lookback_bars: int = 500, detection_pct: float = 0.07) -> Dict[str, Any]:
    empty_res = {"poc": np.nan, "vah": np.nan, "val": np.nan, "overhead_gaps": [], "underneath_gaps": []}
    if df.empty or "volume" not in df.columns:
        return empty_res

    poc, vah, val = compute_volume_profile(df, num_bins=num_bins, lookback_bars=lookback_bars)
    df_range = df.tail(min(len(df), lookback_bars))
    lows, highs, vols = df_range["low"].values, df_range["high"].values, df_range["volume"].values
    pLST, pHST = float(np.min(lows)), float(np.max(highs))
    if pLST >= pHST:
        return empty_res

    pSTP = (pHST - pLST) / num_bins
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
        if np.all(vD_vt[i] < vD_vt[i - noN : i]) and np.all(vD_vt[i] < vD_vt[i + 1 : i + 1 + noN]):
            gap_prices.append(round(pLST + (i + 0.5) * pSTP, 2))

    current_price = float(df.iloc[-1]["close"])
    unique_gaps = list(dict.fromkeys(gap_prices))

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "overhead_gaps": sorted([g for g in unique_gaps if g > current_price]),
        "underneath_gaps": sorted([g for g in unique_gaps if g < current_price], reverse=True)
    }

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or len(df) < period + 1:
        return pd.Series(25.0, index=df.index)
    
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    plus_dm_s = pd.Series(plus_dm, index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)
    
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_s.ewm(alpha=1.0/period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm_s.ewm(alpha=1.0/period, adjust=False).mean() / atr.replace(0, np.nan))
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0/period, adjust=False).mean().fillna(25.0)

# ---------------------------------------------------------------------------
# 5. SYMBOL CONFIGURATION & 200 TEMA DEFAULTS
# ---------------------------------------------------------------------------

def normalize_symbol(symbol: str) -> str:
    if not symbol:
        return ""
    s = str(symbol).replace('"', "").replace("'", "").strip().upper()
    return f"{s[:-4]}/{s[-4:]}" if s.endswith("USDT") and "/" not in s else s


def load_symbol_config(symbol: str) -> Dict[str, Any]:
    return {
        "tema_period": 200,
        "rsi_period": 14,
        "adx_period": 14,
        "min_adx": 20.0,             # <--- Added ADX trend strength threshold
        "zone_tolerance": 0.02,
        "volume_multiplier": 0.75,
        "max_sl_pct": 0.03,
        "min_rr": 3.0,
        "risk_pct": 1.0,
        "atr_period": 14,
        "atr_mult": 3.0,
        "disable_htf": True,
        "spot_only": False,
        "leverage": 5
    }


# ---------------------------------------------------------------------------
# 6. STREAMLINED 4H 200-TEMA SIGNAL EVALUATION ENGINE
# ---------------------------------------------------------------------------

def evaluate_signals(
    df: pd.DataFrame,
    symbol: str,
    account_balance: float = 100.0,
    tema_period: int = 200,
    zone_tolerance: float = 0.02,
    min_rr: float = 3.0,
    risk_pct: float = 1.0,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    min_adx: Optional[float] = None,       # <--- Accept min_adx explicitly
    spot_only: bool = False,
    sentiment_score: Optional[float] = None,
    is_backtest: bool = False,
    timeframe: str = EXECUTION_TIMEFRAME,
    **kwargs
) -> Dict[str, Any]:
    no_signal = {
        "action": "HOLD", "symbol": symbol, "direction": "NONE",
        "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
        "atr": 0.0, "position_size": 0.0, "reason": "No condition met"
    }

    sym_cfg = load_symbol_config(symbol)
    volume_mult = float(sym_cfg.get("volume_multiplier", 0.75))
    
    # Requirement 5: Volume Spike Check using configurable multiplier
    if not check_volume_spike(df, period=20, multiplier=volume_mult):
        no_signal["reason"] = f"Volume Gate: Trigger candle volume below {volume_mult}x 20-period Volume SMA"
        return no_signal

    # Pre-compute Indicators
    df = df.copy()
    df["tema"] = calculate_tema(df["close"], tema_period)
    df["atr"] = calculate_atr(df, atr_period)
    df["adx"] = calculate_adx(df, period=int(sym_cfg.get("adx_period", 14)))

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    current_price = float(last_row["close"])
    current_tema = float(last_row["tema"])
    current_atr = float(last_row["atr"])
    current_adx = float(last_row["adx"])

    # Single Unified ADX Threshold Check
    adx_threshold = float(min_adx if min_adx is not None else kwargs.get("min_adx", sym_cfg.get("min_adx", 20.0)))

    if current_adx < adx_threshold:
        no_signal["reason"] = f"ADX Gate: Trend strength ({current_adx:.1f}) below minimum threshold ({adx_threshold})"
        return no_signal

    vp_data = calculate_volume_profile_gaps(df, lookback_bars=500)
    poc, vah, val = vp_data["poc"], vp_data["vah"], vp_data["val"]
    overhead_gaps, underneath_gaps = vp_data["overhead_gaps"], vp_data["underneath_gaps"]

    # TEMA Touch / Proximity & Crossover Mechanics
    distance_to_tema_pct = abs(current_price - current_tema) / current_tema
    near_tema = distance_to_tema_pct <= zone_tolerance
    crossed_above_tema = (float(prev_row["close"]) <= float(prev_row["tema"])) and (current_price > current_tema)
    crossed_below_tema = (float(prev_row["close"]) >= float(prev_row["tema"])) and (current_price < current_tema)

    long_trigger = (current_price >= current_tema) and (near_tema or crossed_above_tema)
    short_trigger = (current_price <= current_tema) and (near_tema or crossed_below_tema) and not spot_only

    MIN_SL_PCT = 0.003

    # -----------------------------------------------------------------------
    # LONG SETUP (4H 200 TEMA Touch / Cross + Volume Spike + Structure SL/TP)
    # -----------------------------------------------------------------------
    if long_trigger:
        if not check_candlestick_rejection(last_row, prev_row, direction="LONG"):
            no_signal["reason"] = "Long Gate: Missing bullish candlestick confirmation at 200 TEMA"
            return no_signal

        entry_price = float(current_price)

        # Requirement 6: Stop Loss below market structure (TEMA line, VAL, swing low, or POC)
        swing_low = find_structural_swing_point(df, direction="LONG", lookback=5)
        sl_candidates = [current_tema - current_atr, swing_low - (current_atr * 0.5)]
        if not np.isnan(val):
            sl_candidates.append(val - (current_atr * 0.2))
        
        stop_loss = min(sl_candidates)
        max_allowed_sl = entry_price * (1.0 - 0.03)
        if stop_loss < max_allowed_sl:
            stop_loss = max_allowed_sl

        sl_distance = entry_price - stop_loss
        if sl_distance < (entry_price * MIN_SL_PCT):
            stop_loss = entry_price - (current_atr * atr_mult)
            sl_distance = entry_price - stop_loss

        # Requirement 7: Take Profit placed at market structure including Volume Gaps & VAH/POC
        min_tp = entry_price + (sl_distance * min_rr)
        tp_candidates = [float(g) for g in overhead_gaps if float(g) >= min_tp]
        for ref in (vah, poc):
            if ref is not None and not np.isnan(ref) and float(ref) >= min_tp:
                tp_candidates.append(float(ref))

        farthest_tp = max(tp_candidates) if tp_candidates else min_tp
        computed_rr = (farthest_tp - entry_price) / sl_distance

        if computed_rr >= min_rr:
            risk_amt = float(account_balance) * (risk_pct / 100.0)
            position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0
            staged_targets = calculate_staged_targets("LONG", entry_price, stop_loss, farthest_tp)

            return {
                "action": "BUY", "order_type": "LIMIT", "symbol": symbol, "direction": "LONG",
                "regime": "4H_TEMA_200",
                "entry_price": float(round(entry_price, 5)),
                "stop_loss": float(round(stop_loss, 5)),
                "take_profit": float(round(farthest_tp, 5)),
                "tp1": staged_targets["tp1"], "tp1_ratio": staged_targets["tp1_ratio"],
                "tp2": staged_targets["tp2"], "tp2_ratio": staged_targets["tp2_ratio"],
                "max_bar_duration": 36, "atr": float(round(current_atr, 4)),
                "rr_ratio": float(computed_rr), "risk_pct": float(risk_pct),
                "position_size": float(position_size),
                "reason": f"4H 200-TEMA Long: SL={stop_loss:.4f}, TP={farthest_tp:.4f}, R:R={computed_rr:.2f}"
            }

    # -----------------------------------------------------------------------
    # SHORT SETUP (4H 200 TEMA Touch / Cross + Volume Spike + Structure SL/TP)
    # -----------------------------------------------------------------------
    if short_trigger:
        if not check_candlestick_rejection(last_row, prev_row, direction="SHORT"):
            no_signal["reason"] = "Short Gate: Missing bearish candlestick confirmation at 200 TEMA"
            return no_signal

        entry_price = float(current_price)

        # Requirement 6: Stop Loss above market structure (TEMA line, VAH, swing high, or POC)
        swing_high = find_structural_swing_point(df, direction="SHORT", lookback=5)
        sl_candidates = [current_tema + current_atr, swing_high + (current_atr * 0.5)]
        if not np.isnan(vah):
            sl_candidates.append(vah + (current_atr * 0.2))

        stop_loss = max(sl_candidates)
        max_allowed_sl = entry_price * (1.0 + 0.03)
        if stop_loss > max_allowed_sl:
            stop_loss = max_allowed_sl

        sl_distance = stop_loss - entry_price
        if sl_distance < (entry_price * MIN_SL_PCT):
            stop_loss = entry_price + (current_atr * atr_mult)
            sl_distance = stop_loss - entry_price

        # Requirement 7: Take Profit placed at market structure including Volume Gaps & VAL/POC
        min_tp = entry_price - (sl_distance * min_rr)
        tp_candidates = [float(g) for g in underneath_gaps if float(g) <= min_tp]
        for ref in (val, poc):
            if ref is not None and not np.isnan(ref) and float(ref) <= min_tp:
                tp_candidates.append(float(ref))

        farthest_tp = min(tp_candidates) if tp_candidates else min_tp
        computed_rr = (entry_price - farthest_tp) / sl_distance

        if computed_rr >= min_rr:
            risk_amt = float(account_balance) * (risk_pct / 100.0)
            position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0
            staged_targets = calculate_staged_targets("SHORT", entry_price, stop_loss, farthest_tp)

            return {
                "action": "SELL", "order_type": "LIMIT", "symbol": symbol, "direction": "SHORT",
                "regime": "4H_TEMA_200",
                "entry_price": float(round(entry_price, 5)),
                "stop_loss": float(round(stop_loss, 5)),
                "take_profit": float(round(farthest_tp, 5)),
                "tp1": staged_targets["tp1"], "tp1_ratio": staged_targets["tp1_ratio"],
                "tp2": staged_targets["tp2"], "tp2_ratio": staged_targets["tp2_ratio"],
                "max_bar_duration": 36, "atr": float(round(current_atr, 4)),
                "rr_ratio": float(computed_rr), "risk_pct": float(risk_pct),
                "position_size": float(position_size),
                "reason": f"4H 200-TEMA Short: SL={stop_loss:.4f}, TP={farthest_tp:.4f}, R:R={computed_rr:.2f}"
            }

    no_signal["reason"] = "No valid 200-TEMA touch/crossover with volume spike and candle confirmation"
    return no_signal


# ---------------------------------------------------------------------------
# 7. EVENT-DRIVEN STRATEGY ENGINE CLASS
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
            min_rr=float(params.get("min_rr", 3.0)),
            spot_only=(position_side.upper() == "LONG_ONLY")
        )

        action = evaluated_res.get("action", "HOLD")
        evaluated_res["signal"] = action
        evaluated_res["side"] = evaluated_res.get("direction", "NONE")
        return evaluated_res


def generate_live_signal(
    symbol: str,
    account_balance: float = 100.0,
    execution_timeframe: str = EXECUTION_TIMEFRAME,
    limit: int = 500,
    **kwargs
) -> Dict[str, Any]:
    df = fetch_klines(symbol=symbol, interval=execution_timeframe, limit=limit)
    if df.empty:
        return {
            "action": "HOLD", "symbol": symbol, "direction": "NONE",
            "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
            "atr": 0.0, "position_size": 0.0, "reason": "Failed to fetch 4H klines"
        }
    return evaluate_signals(df, symbol, account_balance=account_balance, timeframe=execution_timeframe, **kwargs)