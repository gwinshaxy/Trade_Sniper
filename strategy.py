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
EXECUTION_TIMEFRAME = "15m"      # 15m default execution
DEMA_FAST_PERIOD = 9
EMA_MEDIUM_PERIOD = 50
EMA_TREND_PERIOD = 200
VOLUME_SMA_PERIOD = 20
RSI_PERIOD = 14
ADX_PERIOD = 14
MIN_ADX_THRESHOLD = 22.0         # Existing ADX threshold
CHOP_PERIOD = 14                 # Choppiness Index lookback period
MAX_CHOP_THRESHOLD = 61.8        # Upper limit: filters out choppy/sideways markets if CHOP > 58
TARGET_PROFIT_PCT = 0.025        # Fixed 2.5% target profit
ATR_SL_MULTIPLIER = 2.0          # Dynamic ATR multiplier for initial stop loss
ROUND_TRIP_FEE_PCT = 0.0010      # 0.10% covers taker entry + exit fees


# ---------------------------------------------------------------------------
# 1. ASSET-SPECIFIC OVERRIDES & CONFIG HELPERS
# ---------------------------------------------------------------------------

def get_symbol_overrides(symbol: str) -> dict:
    clean_sym = symbol.split(':')[0].upper()
    if not clean_sym.endswith('/USDT') and '/' not in clean_sym:
        clean_sym = f"{clean_sym}/USDT"

    config = {
        "target_profit_pct": TARGET_PROFIT_PCT,
        "atr_multiplier": ATR_SL_MULTIPLIER
    }

    overrides_table = {
        "HYPE/USDT": {"target_profit_pct": 0.020, "atr_multiplier": 2.0},
        "LINK/USDT": {"target_profit_pct": 0.030, "atr_multiplier": 1.5},
        "CRCL/USDT": {"target_profit_pct": 0.025, "atr_multiplier": 1.5},
        "DOGE/USDT":  {"target_profit_pct": 0.025, "atr_multiplier": 1.5},
        "ETH/USDT":  {"target_profit_pct": 0.025, "atr_multiplier": 1.5},
        "SKHYNIX/USDT": {"target_profit_pct": 0.025, "atr_multiplier": 2.0},
        "SNXX/USDT": {"target_profit_pct": 0.025, "atr_multiplier": 1.5},
        "MU/USDT":  {"target_profit_pct": 0.025, "atr_multiplier": 1.5},
        "NEAR/USDT": {"target_profit_pct": 0.025, "atr_multiplier": 2.0},
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
# 2. TECHNICAL INDICATORS
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
    """
    Calculates the Choppiness Index (CHOP).
    CHOP = 100 * LOG10( SUM(ATR, n) / (MAX_HIGH(n) - MIN_LOW(n)) ) / LOG10(n)
    """
    if df.empty or len(df) < period:
        return pd.Series(50.0, index=df.index)
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 1. True Range for each bar
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 2. Sum of ATR over period
    sum_atr = tr.rolling(window=period).sum()
    
    # 3. Highest High and Lowest Low over period
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    # 4. CHOP Formula
    range_hl = highest_high - lowest_low
    range_hl = range_hl.replace(0, np.nan)  # avoid division by zero
    
    chop = 100.0 * np.log10(sum_atr / range_hl) / np.log10(period)
    return chop.fillna(50.0)


# ---------------------------------------------------------------------------
# 3. SCALPING SIGNAL EVALUATION ENGINE (With CHOP Filter)
# ---------------------------------------------------------------------------

def evaluate_scalp_signals(
    df: pd.DataFrame,
    symbol: str,
    account_balance: float = 100.0,
    risk_pct: float = 1.0,
    use_ema_200_filter: bool = False,
    use_chop_filter: bool = True,
    **kwargs
) -> Dict[str, Any]:
    no_signal = {
        "action": "HOLD", "symbol": symbol, "direction": "NONE",
        "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
        "reason": "No condition met"
    }

    if df.empty or len(df) < 200:
        no_signal["reason"] = "Insufficient kline data for 200 EMA calculation"
        return no_signal

    asset_cfg = get_symbol_overrides(symbol)
    target_profit_pct = asset_cfg["target_profit_pct"]
    atr_mult = asset_cfg["atr_multiplier"]

    df = df.copy()
    df["dema_9"] = calculate_dema(df["close"], DEMA_FAST_PERIOD)
    df["ema_50"] = calculate_ema(df["close"], EMA_MEDIUM_PERIOD)
    df["ema_200"] = calculate_ema(df["close"], EMA_TREND_PERIOD)
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
    ema_200 = float(last_row["ema_200"])
    
    prev_dema = float(prev_row["dema_9"])
    prev_ema50 = float(prev_row["ema_50"])
    
    current_volume = float(last_row["volume"])
    vol_sma20 = float(last_row["vol_sma20"])
    
    current_rsi = float(last_row["rsi"])
    current_adx = float(last_row["adx"])
    current_chop = float(last_row["chop"])
    current_atr = float(last_row["atr"])

    # Filter 1: ADX Trend Filter
    if current_adx < MIN_ADX_THRESHOLD:
        no_signal["reason"] = f"Trend Filter: ADX ({current_adx:.1f}) below threshold {MIN_ADX_THRESHOLD}"
        return no_signal

    # Filter 2: Choppiness Index Filter (CHOP > MAX_CHOP_THRESHOLD indicates choppy range)
    if use_chop_filter and current_chop > MAX_CHOP_THRESHOLD:
        no_signal["reason"] = f"Choppiness Filter: CHOP ({current_chop:.1f}) above limit {MAX_CHOP_THRESHOLD}"
        return no_signal

    # 1. Base Crossover Detection
    bullish_crossover = (prev_dema <= prev_ema50) and (dema_9 > ema_50)
    bearish_crossover = (prev_dema >= prev_ema50) and (dema_9 < ema_50)

    # 2. Strict Directional Volume Spikes on the EXACT Crossover Bar
    is_bullish_volume_spike = (last_row["close"] > last_row["open"]) and (current_volume >= vol_sma20 * 1.5)
    is_bearish_volume_spike = (last_row["close"] < last_row["open"]) and (current_volume >= vol_sma20 * 1.5)

    # 3. Combined Entry Gate: Crossover AND Volume Spike MUST co-occur on the current bar
    valid_long_trigger = bullish_crossover and is_bullish_volume_spike
    valid_short_trigger = bearish_crossover and is_bearish_volume_spike

    # -----------------------------------------------------------------------
    # LONG SCALP SETUP
    # -----------------------------------------------------------------------
    trend_condition_long = (not use_ema_200_filter) or (current_price > ema_200)

    if valid_long_trigger and trend_condition_long and (55.0 < current_rsi < 70.0):
        entry_price = current_price
        stop_loss = entry_price - (current_atr * atr_mult)
        take_profit = entry_price * (1.0 + target_profit_pct)
        
        sl_distance = entry_price - stop_loss
        risk_amt = float(account_balance) * (risk_pct / 100.0)
        position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0
        computed_rr = (take_profit - entry_price) / sl_distance if sl_distance > 0 else 0.0

        return {
            "action": "BUY", "order_type": "MARKET", "symbol": symbol, "direction": "LONG",
            "regime": "DEMA_EMA_STRICT_CROSS_VOL_CHOP",
            "entry_price": format_price(entry_price),
            "stop_loss": format_price(stop_loss),
            "take_profit": format_price(take_profit),
            "rr_ratio": float(round(computed_rr, 2)),
            "position_size": float(position_size),
            "reason": f"Simultaneous Bullish Cross + Vol + RSI ({current_rsi:.1f}) + CHOP ({current_chop:.1f})"
        }

    # -----------------------------------------------------------------------
    # SHORT SCALP SETUP
    # -----------------------------------------------------------------------
    trend_condition_short = (not use_ema_200_filter) or (current_price < ema_200)

    if valid_short_trigger and trend_condition_short and (25.0 < current_rsi < 50.0):
        entry_price = current_price
        stop_loss = entry_price + (current_atr * atr_mult)
        take_profit = entry_price * (1.0 - target_profit_pct)
        
        sl_distance = stop_loss - entry_price
        risk_amt = float(account_balance) * (risk_pct / 100.0)
        position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0
        computed_rr = (entry_price - take_profit) / sl_distance if sl_distance > 0 else 0.0

        return {
            "action": "SELL", "order_type": "MARKET", "symbol": symbol, "direction": "SHORT",
            "regime": "DEMA_EMA_STRICT_CROSS_VOL_CHOP",
            "entry_price": format_price(entry_price),
            "stop_loss": format_price(stop_loss),
            "take_profit": format_price(take_profit),
            "rr_ratio": float(round(computed_rr, 2)),
            "position_size": float(position_size),
            "reason": f"Simultaneous Bearish Cross + Vol + RSI ({current_rsi:.1f}) + CHOP ({current_chop:.1f})"
        }

    no_signal["reason"] = "No valid simultaneous crossover and volume spike matching filter conditions"
    return no_signal

def load_symbol_config(symbol: str) -> dict:
    """Alias for get_symbol_overrides to support legacy main.py imports."""
    return get_symbol_overrides(symbol)

evaluate_signals = evaluate_scalp_signals