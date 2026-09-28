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


# ---------------------------------------------------------------------------
# 3. TECHNICAL INDICATORS
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
    """
    RECONCILIATION: Wilder's ATR smoothing (EMA with alpha=1/period)
    instead of simple rolling mean. Matches dry-run script behavior.
    """
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
    """
    RECONCILIATION: Wilder's smoothing for DM/TR/DI/DX/ADX.
    Uses EMA with alpha=1/period instead of simple rolling mean.
    Matches dry-run script behavior.
    """
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


# ---------------------------------------------------------------------------
# 4. VOLUME PROFILE & LIQUIDITY GAPS
# ---------------------------------------------------------------------------

def compute_volume_profile(df: pd.DataFrame, num_bins: int = 100, lookback_bars: int = 600, va_pct: float = 0.70):
    if df.empty or "volume" not in df.columns or "high" not in df.columns or "low" not in df.columns:
        return np.nan, np.nan, np.nan

    df_range = df.tail(lookback_bars).copy()
    if df_range.empty:
        return np.nan, np.nan, np.nan

    pLST, pHST = float(df_range["low"].min()), float(df_range["high"].max())
    if pLST >= pHST or np.isnan(pLST) or np.isnan(pHST):
        return np.nan, np.nan, np.nan

    pSTP = (pHST - pLST) / num_bins
    if pSTP <= 0:
        return np.nan, np.nan, np.nan

    vD_vt = np.zeros(num_bins)

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
    detection_pct: float = 0.07
) -> Dict[str, Any]:
    empty_res = {"poc": np.nan, "vah": np.nan, "val": np.nan, "overhead_gaps": [], "underneath_gaps": []}
    if df.empty or "volume" not in df.columns or "high" not in df.columns or "low" not in df.columns:
        return empty_res

    poc, vah, val = compute_volume_profile(df, num_bins=num_bins, lookback_bars=lookback_bars)

    df_range = df.tail(lookback_bars).copy()
    if df_range.empty:
        return empty_res

    pLST, pHST = float(df_range["low"].min()), float(df_range["high"].max())
    if pLST >= pHST or np.isnan(pLST) or np.isnan(pHST):
        return empty_res

    pSTP = (pHST - pLST) / num_bins
    if pSTP <= 0:
        return empty_res

    vD_vt = np.zeros(num_bins)

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

            vD_vt[pLI] += lV * max(vPOR, 0.0)

    noN = max(int(num_bins * detection_pct), 1)
    max_val = np.max(vD_vt)
    tVT = list(vD_vt)

    for _ in range(noN):
        tVT.insert(0, max_val)
        tVT.append(max_val)

    gap_prices = []
    for vn in range(2 * noN, num_bins + 2 * noN):
        uNth = all(tVT[vn - noN] < tVT[cVN] for cVN in range(vn - 2 * noN, vn - noN))
        lNth = all(tVT[vn - noN] < tVT[cVN] for cVN in range(vn - noN + 1, vn + 1))

        if uNth and lNth:
            bin_idx = vn - 2 * noN
            gap_price = round(pLST + (bin_idx + 0.5) * pSTP, 2)
            if pLST <= gap_price <= pHST:
                gap_prices.append(gap_price)

    unique_gaps = list(dict.fromkeys(gap_prices))
    current_price = float(df.iloc[-1]["close"])

    overhead_gaps = sorted([g for g in unique_gaps if g > current_price])
    underneath_gaps = sorted([g for g in unique_gaps if g < current_price], reverse=True)

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "overhead_gaps": overhead_gaps,
        "underneath_gaps": underneath_gaps
    }


# ---------------------------------------------------------------------------
# 5. SYMBOL CONFIGURATION & PARAMETERS
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


def load_symbol_config(symbol: str) -> Dict[str, Any]:
    formatted_symbol = normalize_symbol(symbol)
    raw_symbol = formatted_symbol.replace("/", "").upper()

    if HAS_DB:
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * 
                    FROM strategy_parameters 
                    WHERE UPPER(TRIM(REPLACE(REPLACE(symbol, '"', ''), '''', ''))) IN (%s, %s)
                    ORDER BY updated_at DESC LIMIT 1;
                """, (formatted_symbol, raw_symbol))
                row = cursor.fetchone()

                if row:
                    colnames = [desc[0] for desc in cursor.description] if cursor.description else []
                    cursor.close()
                    config = dict(zip(colnames, row)) if isinstance(row, tuple) else dict(row)

                    config["tema_period"] = int(config.get("tema_period", 200))
                    config["rsi_period"] = int(config.get("rsi_period", 14))
                    config["rsi_thresh"] = float(config.get("rsi_thresh", 42.0))
                    config["adx_period"] = int(config.get("adx_period", 14))
                    config["adx_threshold"] = float(config.get("adx_threshold", 22.0))
                    config["max_sl_pct"] = float(config.get("max_sl_pct", 0.015))
                    config["zone_tolerance"] = float(config.get("zone_tolerance", 0.005))
                    config["min_sentiment"] = float(config.get("min_sentiment", 0.0))
                    config["risk_pct"] = float(config.get("risk_pct", 0.5))
                    config["min_rr"] = float(config.get("min_rr", 2.0))
                    # RECONCILIATION: default lookback_bars reduced from 1200 -> 600
                    config["lookback_bars"] = int(config.get("lookback_bars", 600))
                    config["vp_detection_pct"] = float(config.get("vp_detection_pct", 0.07))
                    config["vp_va_pct"] = float(config.get("vp_va_pct", 0.70))
                    config["atr_period"] = int(config.get("atr_period", 14))
                    config["atr_mult"] = float(config.get("atr_mult", 1.5))

                    config.setdefault("use_rsi_filter", True)
                    config.setdefault("use_candlestick_confirm", True)
                    config.setdefault("use_adx_filter", True)
                    config.setdefault("use_atr_sl", True)
                    config.setdefault("disable_htf", False)
                    config.setdefault("spot_only", False)
                    return config
                cursor.close()
            except Exception as e:
                logger.error(f"[load_symbol_config] DB query failed for {formatted_symbol}/{raw_symbol}: {e}")
            finally:
                release_db_connection(conn)

    return {
        "tema_period": 200, "rsi_period": 14, "rsi_thresh": 42.0,
        "adx_period": 14, "adx_threshold": 22.0,
        "use_adx_filter": True, "use_rsi_filter": True, "use_candlestick_confirm": True,
        "zone_tolerance": 0.005, "max_sl_pct": 0.015, "min_sentiment": 0.0,
        "min_rr": 2.0, "risk_pct": 0.5, "vp_detection_pct": 0.07, "lookback_bars": 600,
        "vp_va_pct": 0.70, "atr_period": 14, "atr_mult": 1.5,
        "use_atr_sl": True, "disable_htf": False, "spot_only": False
    }


# ---------------------------------------------------------------------------
# 6. HARMONIZED MULTI-DIRECTIONAL EVALUATION ENGINE
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
    sentiment_score: Optional[float] = None
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

    df = df.copy()
    df["tema"] = calculate_tema(df["close"], tema_period)
    df["rsi"] = calculate_rsi(df["close"], rsi_period)
    df["adx"] = calculate_adx(df, adx_period)
    df["atr"] = calculate_atr(df, atr_period)

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    current_price = float(last_row["close"])
    current_tema = float(last_row["tema"])
    current_rsi = float(last_row["rsi"])
    current_adx = float(last_row["adx"])
    current_atr = float(last_row["atr"])

    final_sentiment = float(sentiment_score) if sentiment_score is not None else 0.5
    if final_sentiment < min_sentiment:
        no_signal["reason"] = f"Sentiment score ({final_sentiment:.2f}) below threshold ({min_sentiment:.2f})"
        return no_signal

    # -----------------------------------------------------------------------
    # HTF CONFLUENCE (RECONCILIATION: 4H TEMA(200) with slope confirmation)
    # -----------------------------------------------------------------------
    macro_trend_long = True
    macro_trend_short = True

    if not disable_htf:
        try:
            df_htf = fetch_klines(symbol, interval=MACRO_TIMEFRAME, limit=max(HTF_TEMA_PERIOD + 20, 250))
            if not df_htf.empty and len(df_htf) >= HTF_TEMA_PERIOD + 2:
                df_htf["tema_htf"] = calculate_tema(df_htf["close"], HTF_TEMA_PERIOD)
                htf_last = df_htf.iloc[-1]
                htf_prev = df_htf.iloc[-2]

                htf_close = float(htf_last["close"])
                htf_tema_now = float(htf_last["tema_htf"])
                htf_tema_prev = float(htf_prev["tema_htf"])

                macro_trend_long = (htf_close > htf_tema_now) and (htf_tema_now > htf_tema_prev)
                macro_trend_short = (htf_close < htf_tema_now) and (htf_tema_now < htf_tema_prev)
        except Exception as e:
            logger.warning(f"[{symbol}] Could not calculate HTF {MACRO_TIMEFRAME} TEMA({HTF_TEMA_PERIOD}) confluence: {e}")

    vp_data = calculate_volume_profile_gaps(df, num_bins=100, lookback_bars=lookback_bars, detection_pct=vp_detection_pct)
    poc = vp_data["poc"]
    vah = vp_data["vah"]
    val = vp_data["val"]
    overhead_gaps = vp_data["overhead_gaps"]
    underneath_gaps = vp_data["underneath_gaps"]

    adx_valid = (not use_adx_filter) or (current_adx >= adx_threshold)
    MIN_SL_PCT = 0.002

    # -----------------------------------------------------------------------
    # PROXIMITY ZONES (RECONCILIATION: TEMA + VAL/VAH + gap proximity)
    # -----------------------------------------------------------------------
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
    # LONG PATH
    # -----------------------------------------------------------------------
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
        and adx_valid
        and long_candlestick
        and macro_trend_long
    ):
        entry_price = float(current_price)

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

        # RECONCILIATION: farthest valid TP candidate selection
        min_tp = entry_price + (sl_distance * min_rr)
        tp_candidates: List[float] = []

        for g in overhead_gaps:
            gf = float(g)
            if gf >= min_tp:
                tp_candidates.append(gf)

        for ref in (vah, poc):
            if ref is not None and not (isinstance(ref, float) and np.isnan(ref)):
                rf = float(ref)
                if rf >= min_tp:
                    tp_candidates.append(rf)

        if tp_candidates:
            take_profit = max(tp_candidates)
        else:
            take_profit = min_tp

        computed_rr = (take_profit - entry_price) / sl_distance
        if computed_rr >= min_rr:
            risk_amt = float(account_balance) * (risk_pct / 100.0)
            position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0

            return {
                "action": "BUY", "symbol": symbol, "direction": "LONG",
                "entry_price": float(entry_price), "stop_loss": float(stop_loss),
                "take_profit": float(take_profit), "atr": float(round(current_atr, 4)),
                "rr_ratio": float(computed_rr),
                "risk_pct": float(risk_pct),
                "position_size": float(position_size),
                "reason": (
                    f"Long confluence: 4H TEMA({HTF_TEMA_PERIOD}) macro bullish, "
                    f"price in TEMA/VAL/gap zone, R:R={computed_rr:.2f}"
                )
            }

    # -----------------------------------------------------------------------
    # SHORT PATH
    # -----------------------------------------------------------------------
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
            and adx_valid
            and short_candlestick
            and macro_trend_short
        ):
            entry_price = float(current_price)

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

            # RECONCILIATION: farthest valid TP candidate selection (lowest for shorts)
            min_tp = entry_price - (sl_distance * min_rr)
            tp_candidates = []

            for g in underneath_gaps:
                gf = float(g)
                if gf <= min_tp:
                    tp_candidates.append(gf)

            for ref in (val, poc):
                if ref is not None and not (isinstance(ref, float) and np.isnan(ref)):
                    rf = float(ref)
                    if rf <= min_tp:
                        tp_candidates.append(rf)

            if tp_candidates:
                take_profit = min(tp_candidates)
            else:
                take_profit = min_tp

            computed_rr = (entry_price - take_profit) / sl_distance
            if computed_rr >= min_rr:
                risk_amt = float(account_balance) * (risk_pct / 100.0)
                position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0

                return {
                    "action": "SELL", "symbol": symbol, "direction": "SHORT",
                    "entry_price": float(entry_price), "stop_loss": float(stop_loss),
                    "take_profit": float(take_profit), "atr": float(round(current_atr, 4)),
                    "rr_ratio": float(computed_rr),
                    "risk_pct": float(risk_pct),
                    "position_size": float(position_size),
                    "reason": (
                        f"Short confluence: 4H TEMA({HTF_TEMA_PERIOD}) macro bearish, "
                        f"price in TEMA/VAH/gap zone, R:R={computed_rr:.2f}"
                    )
                }

    no_signal["reason"] = "Market conditions did not meet strategy entry criteria"
    return no_signal


# ---------------------------------------------------------------------------
# 7. EVENT-DRIVEN SIGNAL GENERATION CLASS
# ---------------------------------------------------------------------------

class StrategyEngine:
    def __init__(self, symbol: str, config: Optional[Any] = None, event_bus: Optional[Any] = None):
        self.symbol = symbol
        self.config = config
        self.event_bus = event_bus

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_price: float = None,
        position_side: str = "LONG",
        params: dict = None
    ) -> Dict[str, Any]:
        """
        Incremental, event-driven signal generation from a pre-computed dataframe.
        FIX #7 preserved: take_profit always computed locally to prevent NameError.
        """
        if df.empty or len(df) < 50:
            return {"signal": "HOLD", "action": "HOLD"}

        if current_price is None:
            current_price = float(df.iloc[-1]["close"])

        params = params or {}
        min_rr = float(params.get("min_rr", getattr(self.config, "MIN_RR", 2.5) if self.config else 2.5))
        atr_multiplier = float(params.get("atr_mult", getattr(self.config, "ATR_MULT", 2.5) if self.config else 2.5))
        atr_period = int(params.get("atr_period", getattr(self.config, "ATR_PERIOD", 14) if self.config else 14))
        risk_pct = float(params.get("risk_pct", 0.5))
        account_balance = float(params.get("account_balance", 100.0))

        if "atr" in df.columns:
            current_atr = float(df["atr"].iloc[-1])
        else:
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            # RECONCILIATION: Wilder's smoothing for inline ATR calc
            current_atr = float(tr.ewm(alpha=1.0 / atr_period, adjust=False).mean().iloc[-1])

        latest = df.iloc[-1]
        rsi_thresh = float(params.get("rsi_thresh", 42.0))
        direction = position_side.upper()

        if "tema" in df.columns and "rsi" in df.columns:
            if latest["close"] > latest["tema"] and latest["rsi"] > rsi_thresh:
                direction = "BUY"
            elif latest["close"] < latest["tema"] and latest["rsi"] < (100.0 - rsi_thresh):
                direction = "SELL"
            else:
                return {"signal": "HOLD", "action": "HOLD"}

        sl_dist = max(current_atr * atr_multiplier, current_price * 0.002)
        if direction in ["BUY", "LONG"]:
            stop_loss = current_price - sl_dist
            take_profit = current_price + (sl_dist * min_rr)
        else:
            stop_loss = current_price + sl_dist
            take_profit = current_price - (sl_dist * min_rr)

        sl_val = float(round(stop_loss, 5))
        tp_val = float(round(take_profit, 5))

        risk_amt = account_balance * (risk_pct / 100.0)
        position_size = round(risk_amt / sl_dist, 6) if sl_dist > 0 else 0.0

        signal_payload = {
            "signal": direction,
            "action": direction,
            "pair": self.symbol,
            "symbol": self.symbol,
            "side": direction,
            "direction": direction,
            "entry_price": float(current_price),
            "stop_loss": sl_val,
            "sl_price": sl_val,
            "take_profit": tp_val,
            "tp_price": tp_val,
            "risk_reward": min_rr,
            "risk_pct": risk_pct,
            "position_size": float(position_size),
            "atr": float(round(current_atr, 4)),
            "timestamp": int(time.time()),
        }

        if getattr(self, "event_bus", None) and hasattr(self.event_bus, "publish"):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.event_bus.publish("TRADE_SIGNAL", signal_payload))
            except Exception:
                pass

        return signal_payload


# ---------------------------------------------------------------------------
# 8. CONVENIENCE: FULL EVALUATION WRAPPER (1H EXECUTION TF)
# ---------------------------------------------------------------------------

def generate_live_signal(
    symbol: str,
    account_balance: float = 100.0,
    execution_timeframe: str = EXECUTION_TIMEFRAME,
    limit: int = 600,
    **kwargs
) -> Dict[str, Any]:
    """
    Fetches 1H execution-timeframe klines and runs the full evaluation engine,
    which internally uses 4H TEMA(200) for macro confluence.
    """
    df = fetch_klines(symbol=symbol, interval=execution_timeframe, limit=limit)
    if df.empty:
        return {
            "action": "HOLD", "symbol": symbol, "direction": "NONE",
            "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
            "atr": 0.0, "position_size": 0.0,
            "reason": "Failed to fetch klines"
        }
    return evaluate_signals(df, symbol, account_balance=account_balance, **kwargs)