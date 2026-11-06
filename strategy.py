import os
import time
import math
import logging
import json
import ccxt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

logger = logging.getLogger("scalp_strategy_engine")

# ---------------------------------------------------------------------------
# GLOBAL SCALPING TIMEFRAME & DEFAULT PARAMETERS
# ---------------------------------------------------------------------------
EXECUTION_TIMEFRAME = "15m"      
DEMA_FAST_PERIOD = 9
EMA_MEDIUM_PERIOD = 50
EMA_TREND_PERIOD = 200
VOLUME_SMA_PERIOD = 20
RSI_PERIOD = 14
ADX_PERIOD = 14
MIN_ADX_THRESHOLD = 20.0         
CHOP_PERIOD = 14                 
MAX_CHOP_THRESHOLD = 61.8       
TARGET_RR_RATIO = 1.8            
ATR_SL_MULTIPLIER = 2.0          
ROUND_TRIP_FEE_PCT = 0.0010      

def get_symbol_overrides(symbol: str) -> dict:
    clean_sym = symbol.split(':')[0].upper()
    if not clean_sym.endswith('/USDT') and '/' not in clean_sym:
        clean_sym = f"{clean_sym}/USDT"

    config = {
        "target_risk_reward_ratio": TARGET_RR_RATIO,
        "atr_multiplier": ATR_SL_MULTIPLIER,
        "use_ema_200_filter": True  # Default global macro filter state
    }

    overrides_table = {
        # Assets performing better with 1H 200 DEMA Filter = FALSE
        "MAGMA/USDT":   {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": False},
        "JCT/USDT":     {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": False},
        "SNXX/USDT":    {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": False},
        "MU/USDT":      {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": False},
        "SOXL/USDT":    {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": False},
        "BZ/USDT":      {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": False},
        "AMD/USDT":     {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": False},
        "1000PEPE/USDT":{"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": False},
        "SAMSUNG/USDT": {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": False},
        "ADA/USDT":     {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": False},

        # Assets performing well with 1H 200 DEMA Filter = TRUE (or remaining default)
        "HYPE/USDT":    {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
        "LINK/USDT":    {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
        "DOGE/USDT":    {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
        "ETH/USDT":     {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
        "SKHYNIX/USDT": {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.2, "use_ema_200_filter": True},
        "AGT/USDT":     {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
        "BNB/USDT":     {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
        "BMT/USDT":     {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
        "ENA/USDT":     {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
        "CAKE/USDT":    {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
        "SNDK/USDT":    {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
        "CL/USDT":      {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
        "SOXS/USDT":    {"target_risk_reward_ratio": 1.8, "atr_multiplier": 2.0, "use_ema_200_filter": True},
    }

    if clean_sym in overrides_table:
        config.update(overrides_table[clean_sym])

    return config


def format_price(price: float) -> float:
    if price <= 0:
        return 0.0
    elif price < 0.0001:
        return round(price, 8)
    elif price < 0.01:
        return round(price, 6)
    elif price < 1.0:
        return round(price, 5)
    elif price < 100.0:
        return round(price, 4)
    else:
        return round(price, 2)


def fetch_top_volume_symbols(limit: int = 10) -> List[str]:
    try:
        exchange = ccxt.binanceusdm({'enableRateLimit': True})
        tickers = exchange.fetch_tickers()
        
        usdt_futures = []
        for symbol, data in tickers.items():
            if '/USDT:USDT' in symbol or (symbol.endswith('USDT') and ':' not in symbol):
                quote_volume = data.get('quoteVolume', 0) or 0
                clean_sym = symbol.split(':')[0]
                if not clean_sym.endswith('USDT'):
                    clean_sym = f"{clean_sym}/USDT"
                usdt_futures.append((clean_sym, quote_volume))
                
        usdt_futures.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [item[0] for item in usdt_futures[:limit]]
        logger.info(f"Successfully fetched top {len(top_symbols)} volume symbols.")
        return top_symbols
    except Exception as e:
        logger.error(f"Failed to scan top volume symbols: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT"]


def fetch_klines(symbol: str, interval: str = "15m", limit: int = 600) -> pd.DataFrame:
    try:
        exchange = ccxt.binanceusdm({'enableRateLimit': True})
        clean_symbol = symbol.split(':')[0]
        if not clean_symbol.endswith('USDT') and '/' not in clean_symbol:
            clean_symbol = f"{clean_symbol}/USDT"
            
        ohlcv = exchange.fetch_ohlcv(clean_symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            return pd.DataFrame()
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch klines for {symbol}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. TECHNICAL INDICATORS & MULTI-TIMEFRAME HELPERS
# ---------------------------------------------------------------------------

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_dema(series: pd.Series, period: int = 9) -> pd.Series:
    e1 = calculate_ema(series, period)
    e2 = calculate_ema(e1, period)
    return 2 * e1 - e2


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


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    return dx.rolling(period).mean().fillna(0)


def calculate_choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or len(df) < period:
        return pd.Series(50.0, index=df.index)
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    sum_atr = tr.rolling(window=period).sum()
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    range_hl = highest_high - lowest_low
    range_hl = range_hl.replace(0, np.nan)
    
    chop = 100.0 * np.log10(sum_atr / range_hl) / np.log10(period)
    return chop.fillna(50.0)


def compute_1h_dema_200(df_15m: pd.DataFrame, df_1h: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Computes 200 DEMA on 1h data and merges it forward onto the 15m index.
    If df_1h is not provided, falls back to resampling 15m to 1h.
    """
    if df_1h is None or df_1h.empty:
        df_temp = df_15m.copy()
        df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
        df_1h = df_temp.set_index("timestamp").resample("1h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna().reset_index()

    e1 = calculate_ema(df_1h["close"], 200)
    e2 = calculate_ema(e1, 200)
    df_1h["dema_200_1h"] = 2 * e1 - e2

    df_15m_indexed = df_15m.copy()
    df_15m_indexed["timestamp"] = pd.to_datetime(df_15m_indexed["timestamp"])
    df_1h_indexed = df_1h.set_index("timestamp")

    merged = pd.merge_asof(
        df_15m_indexed.sort_values("timestamp"),
        df_1h_indexed[["dema_200_1h"]].sort_index(),
        on="timestamp",
        direction="backward"
    )
    return merged["dema_200_1h"].bfill()


# ---------------------------------------------------------------------------
# 3. SCALPING SIGNAL EVALUATION ENGINE (Multi-Timeframe Macro Filter)
# ---------------------------------------------------------------------------

def evaluate_scalp_signals(
    df: pd.DataFrame,
    symbol: str,
    account_balance: float = 100.0,
    risk_pct: float = 1.0,
    use_ema_200_filter: bool = True,
    use_chop_filter: bool = True,
    df_1h: Optional[pd.DataFrame] = None,
    **kwargs
) -> Dict[str, Any]:
    no_signal = {
        "action": "HOLD", "symbol": symbol, "direction": "NONE",
        "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
        "reason": "No condition met"
    }

    if df.empty or len(df) < 200:
        no_signal["reason"] = "Insufficient kline data for indicators"
        return no_signal

    asset_cfg = get_symbol_overrides(symbol)
    target_rr = asset_cfg["target_risk_reward_ratio"]
    atr_mult = asset_cfg["atr_multiplier"]
    
    # Respect symbol override for 1H 200 DEMA Filter
    active_use_ema_200 = asset_cfg.get("use_ema_200_filter", use_ema_200_filter)

    df = df.copy()
    df["dema_9"] = calculate_dema(df["close"], DEMA_FAST_PERIOD)
    df["ema_50"] = calculate_ema(df["close"], EMA_MEDIUM_PERIOD)
    
    # Macro Filter: 200 DEMA on 1-Hour Timeframe
    df["macro_trend_filter"] = compute_1h_dema_200(df, df_1h)

    df["vol_sma20"] = df["volume"].rolling(VOLUME_SMA_PERIOD).mean()
    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    df["adx"] = calculate_adx(df, ADX_PERIOD)
    df["chop"] = calculate_choppiness_index(df, CHOP_PERIOD)
    df["atr"] = calculate_atr(df, period=14)

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    current_price = float(last_row["close"])
    dema_9 = float(last_row["dema_9"])
    ema_50 = float(last_row["ema_50"])
    macro_200_dema_1h = float(last_row["macro_trend_filter"])
    
    prev_dema = float(prev_row["dema_9"])
    prev_ema50 = float(prev_row["ema_50"])
    
    current_volume = float(last_row["volume"])
    vol_sma20 = float(last_row["vol_sma20"])
    prev_volume = float(prev_row["volume"])
    prev_vol_sma20 = float(prev_row["vol_sma20"]) if not np.isnan(prev_row["vol_sma20"]) else vol_sma20
    
    current_rsi = float(last_row["rsi"])
    current_adx = float(last_row["adx"])
    current_chop = float(last_row["chop"])
    current_atr = float(last_row["atr"])

    # Filter 1: ADX Trend Filter
    if current_adx < MIN_ADX_THRESHOLD:
        no_signal["reason"] = f"Trend Filter: ADX ({current_adx:.1f}) below threshold {MIN_ADX_THRESHOLD}"
        return no_signal

    # Filter 2: Choppiness Index Filter
    if use_chop_filter and current_chop > MAX_CHOP_THRESHOLD:
        no_signal["reason"] = f"Choppiness Filter: CHOP ({current_chop:.1f}) above limit {MAX_CHOP_THRESHOLD}"
        return no_signal

    # Crossover Detection & Flexible Volume Spikes
    bullish_crossover = (prev_dema <= prev_ema50) and (dema_9 > ema_50)
    bearish_crossover = (prev_dema >= prev_ema50) and (dema_9 < ema_50)

    is_bullish_volume_spike = (last_row["close"] > last_row["open"]) and (
        current_volume >= vol_sma20 * 1.05 or prev_volume >= prev_vol_sma20 * 1.05
    )
    is_bearish_volume_spike = (last_row["close"] < last_row["open"]) and (
        current_volume >= vol_sma20 * 1.05 or prev_volume >= prev_vol_sma20 * 1.05
    )

    valid_long_trigger = bullish_crossover and is_bullish_volume_spike
    valid_short_trigger = bearish_crossover and is_bearish_volume_spike

    # -----------------------------------------------------------------------
    # LONG SCALP SETUP (Relaxed RSI Threshold: RSI > 50)
    # -----------------------------------------------------------------------
    trend_condition_long = (not active_use_ema_200) or (current_price > macro_200_dema_1h)

    if valid_long_trigger and trend_condition_long and (current_rsi > 50.0):
        entry_price = current_price
        
        stop_loss = entry_price - (current_atr * atr_mult)
        if stop_loss >= entry_price:
            stop_loss = entry_price * 0.99

        sl_distance = entry_price - stop_loss
        take_profit = entry_price + (sl_distance * target_rr)

        risk_amt = float(account_balance) * (risk_pct / 100.0)
        position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0

        return {
            "action": "BUY", "order_type": "MARKET", "symbol": symbol, "direction": "LONG",
            "regime": "DEMA_1H_MACRO_SCALP",
            "entry_price": format_price(entry_price),
            "stop_loss": format_price(stop_loss),
            "take_profit": format_price(take_profit),
            "rr_ratio": float(target_rr),
            "position_size": float(position_size),
            "reason": f"1H Macro Long + Cross + Vol + {target_rr}R TP"
        }

    # -----------------------------------------------------------------------
    # SHORT SCALP SETUP (Relaxed RSI Threshold: RSI < 50)
    # -----------------------------------------------------------------------
    trend_condition_short = (not active_use_ema_200) or (current_price < macro_200_dema_1h)

    if valid_short_trigger and trend_condition_short and (current_rsi < 50.0):
        entry_price = current_price
        
        stop_loss = entry_price + (current_atr * atr_mult)
        if stop_loss <= entry_price:
            stop_loss = entry_price * 1.01

        sl_distance = stop_loss - entry_price
        take_profit = entry_price - (sl_distance * target_rr)

        risk_amt = float(account_balance) * (risk_pct / 100.0)
        position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0

        return {
            "action": "SELL", "order_type": "MARKET", "symbol": symbol, "direction": "SHORT",
            "regime": "DEMA_1H_MACRO_SCALP",
            "entry_price": format_price(entry_price),
            "stop_loss": format_price(stop_loss),
            "take_profit": format_price(take_profit),
            "rr_ratio": float(target_rr),
            "position_size": float(position_size),
            "reason": f"1H Macro Short + Cross + Vol + {target_rr}R TP"
        }

    no_signal["reason"] = "No valid crossover matching 1H macro rules"
    return no_signal


def load_symbol_config(symbol: str) -> dict:
    return get_symbol_overrides(symbol)

evaluate_signals = evaluate_scalp_signals