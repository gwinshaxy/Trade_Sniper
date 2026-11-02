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
# GLOBAL TIMEFRAME & DEFAULT PARAMETERS (Optimized for 15m / 30m / 1hr)
# ---------------------------------------------------------------------------
EXECUTION_TIMEFRAME = "15m"      
AA_VWAP_PERIOD = 48             # Increased lookback for stable intraday structure (12 hrs on 15m)
VOLUME_SMA_PERIOD = 20
RSI_PERIOD = 14
ADX_PERIOD = 14
MIN_ADX_THRESHOLD = 22.0         # Stricter trend filter
CHOP_PERIOD = 14                 
MAX_CHOP_THRESHOLD = 58.0        # Choppiness filter
TARGET_RR_RATIO = 2.2            # Optimized Risk-Reward ratio
ATR_SL_MULTIPLIER = 1.8          
ROUND_TRIP_FEE_PCT = 0.0010      


def get_symbol_overrides(symbol: str) -> dict:
    clean_sym = symbol.split(':')[0].upper()
    if not clean_sym.endswith('/USDT') and '/' not in clean_sym:
        clean_sym = f"{clean_sym}/USDT"

    config = {
        "target_risk_reward_ratio": TARGET_RR_RATIO,
        "atr_multiplier": ATR_SL_MULTIPLIER
    }

    overrides_table = {
        "HYPE/USDT":    {"target_risk_reward_ratio": 2.2, "atr_multiplier": 1.8},
        "SOL/USDT":     {"target_risk_reward_ratio": 2.2, "atr_multiplier": 2.0},
        "BTC/USDT":     {"target_risk_reward_ratio": 2.0, "atr_multiplier": 1.6},
        "ETH/USDT":     {"target_risk_reward_ratio": 2.2, "atr_multiplier": 1.8},
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
        return top_symbols
    except Exception as e:
        logger.error(f"Failed to scan top volume symbols: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]


def fetch_klines(symbol: str, interval: str = EXECUTION_TIMEFRAME, limit: int = 600) -> pd.DataFrame:
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
# TECHNICAL INDICATORS & STRUCTURAL CALCULATIONS
# ---------------------------------------------------------------------------

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


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
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    sum_atr = tr.rolling(window=period).sum()
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    range_hl = (highest_high - lowest_low).replace(0, np.nan)
    chop = 100.0 * np.log10(sum_atr / range_hl) / np.log10(period)
    return chop.fillna(50.0)


def calculate_auto_anchored_vwap(df: pd.DataFrame, period: int = AA_VWAP_PERIOD, num_stds: list = [1, 2, 3]) -> pd.DataFrame:
    df = df.copy()
    hlc3 = (df['high'] + df['low'] + df['close']) / 3.0
    vol = df['volume']
    
    cum_pv = (hlc3 * vol).rolling(window=period, min_periods=1).sum()
    cum_v = vol.rolling(window=period, min_periods=1).sum().replace(0, np.nan)
    vwap = cum_pv / cum_v
    
    variance_numerator = (vol * (hlc3 - vwap)**2).rolling(window=period, min_periods=1).sum()
    variance = variance_numerator / cum_v
    stdev = np.sqrt(variance.fillna(0))
    
    df['aa_vwap'] = vwap
    for std in num_stds:
        df[f'upper_band_{std}'] = vwap + (stdev * std)
        df[f'lower_band_{std}'] = vwap - (stdev * std)
    return df


def calculate_volume_profile_levels(df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    sub_df = df.tail(lookback)
    if sub_df.empty:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "hvn": 0.0}
    
    min_p, max_p = sub_df['low'].min(), sub_df['high'].max()
    if min_p == max_p:
        return {"poc": min_p, "vah": max_p, "val": min_p, "hvn": min_p}
        
    bins = 50
    hist, bin_edges = np.histogram(sub_df['close'], bins=bins, weights=sub_df['volume'])
    poc_idx = np.argmax(hist)
    poc = (bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2.0
    
    total_vol = hist.sum()
    if total_vol == 0:
        return {"poc": poc, "vah": max_p, "val": min_p, "hvn": poc}
        
    target_vol = total_vol * 0.68
    current_vol = hist[poc_idx]
    lower_idx, upper_idx = poc_idx, poc_idx
    
    while current_vol < target_vol and (lower_idx > 0 or upper_idx < bins - 1):
        added = False
        if upper_idx < bins - 1:
            upper_idx += 1
            current_vol += hist[upper_idx]
            added = True
        if lower_idx > 0:
            lower_idx -= 1
            current_vol += hist[lower_idx]
            added = True
        if not added:
            break
            
    val = bin_edges[lower_idx]
    vah = bin_edges[upper_idx + 1]
    hvn = poc
    return {"poc": float(poc), "vah": float(vah), "val": float(val), "hvn": float(hvn)}


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    body = (df['close'] - df['open']).abs()
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    
    # Strict Rejection Patterns
    df['is_hammer'] = (lower_shadow >= 1.8 * body) & (upper_shadow <= body * 0.6) & (body > 0) & (df['close'] > df['open'])
    prev_close = df['close'].shift(1)
    prev_open = df['open'].shift(1)
    df['is_bullish_engulfing'] = (prev_close < prev_open) & (df['close'] > df['open']) & (df['close'] >= prev_open) & (df['open'] <= prev_close)
    
    df['is_shooting_star'] = (upper_shadow >= 1.8 * body) & (lower_shadow <= body * 0.6) & (body > 0) & (df['close'] < df['open'])
    df['is_bearish_engulfing'] = (prev_close > prev_open) & (df['close'] < df['open']) & (df['close'] <= prev_open) & (df['open'] >= prev_close)
    
    return df


# ---------------------------------------------------------------------------
# SIGNAL EVALUATION ENGINE (Corrected Confluence & Decoupled Rejection Rules)
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

    if df.empty or len(df) < 120:
        no_signal["reason"] = "Insufficient kline data"
        return no_signal

    asset_cfg = get_symbol_overrides(symbol)
    target_rr = asset_cfg["target_risk_reward_ratio"]
    atr_mult = asset_cfg["atr_multiplier"]

    df = calculate_auto_anchored_vwap(df, period=AA_VWAP_PERIOD)
    df = detect_candlestick_patterns(df)
    df["vol_sma20"] = df["volume"].rolling(VOLUME_SMA_PERIOD).mean()
    df["ema_200"] = calculate_ema(df["close"], 200)
    df["adx"] = calculate_adx(df, ADX_PERIOD)
    df["chop"] = calculate_choppiness_index(df, CHOP_PERIOD)
    df["atr"] = calculate_atr(df, 14)

    vp_levels = calculate_volume_profile_levels(df, lookback=100)

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    current_price = float(last_row["close"])
    current_volume = float(last_row["volume"])
    vol_sma = float(last_row["vol_sma20"])
    current_adx = float(last_row["adx"])
    current_chop = float(last_row["chop"])
    current_atr = float(last_row["atr"])
    ema_200 = float(last_row["ema_200"])

    aa_vwap = float(last_row["aa_vwap"])
    lower_1 = float(last_row["lower_band_1"])
    lower_2 = float(last_row["lower_band_2"])
    upper_1 = float(last_row["upper_band_1"])
    upper_2 = float(last_row["upper_band_2"])

    # Filters
    if current_adx < MIN_ADX_THRESHOLD:
        no_signal["reason"] = f"ADX Filter: ({current_adx:.1f}) < {MIN_ADX_THRESHOLD}"
        return no_signal

    if use_chop_filter and current_chop > MAX_CHOP_THRESHOLD:
        no_signal["reason"] = f"CHOP Filter: ({current_chop:.1f}) > {MAX_CHOP_THRESHOLD}"
        return no_signal

    # Volume Validation Gate (Rule Set D): Require 1.5x volume spike on trigger candle & dry-up on pullback
    is_volume_spike = current_volume >= (vol_sma * 1.5)
    is_pullback_contracting = prev_row["volume"] < vol_sma

    if not (is_volume_spike and is_pullback_contracting):
        no_signal["reason"] = "Volume Validation Gate: Insufficient volume spike or lack of pullback contraction"
        return no_signal

    # -----------------------------------------------------------------------
    # RULE SET A: LONG ENTRY (Strict Confluence: Support + Hammer/Engulfing/Spring)
    # -----------------------------------------------------------------------
    trend_ok_long = (not use_ema_200_filter) or (current_price > ema_200)
    
    structural_long_support = (
        (abs(current_price - lower_1) / lower_1 < 0.006) or
        (abs(current_price - lower_2) / lower_2 < 0.006) or
        (abs(current_price - aa_vwap) / aa_vwap < 0.004) or
        (abs(current_price - vp_levels["val"]) / vp_levels["val"] < 0.006) or
        (current_price < lower_1 and last_row["close"] > lower_1) # Spring / False breakdown reclaim
    )

    # Candlestick pattern must explicitly confirm rejection
    bullish_candle_confirmation = bool(last_row["is_hammer"] or last_row["is_bullish_engulfing"] or (current_price < lower_1 and last_row["close"] > lower_1))

    if structural_long_support and trend_ok_long and bullish_candle_confirmation:
        entry_price = current_price
        
        local_swing_low = float(df['low'].tail(5).min())
        stop_loss = min(local_swing_low - (current_atr * 0.5), lower_2 if not math.isnan(lower_2) else entry_price * 0.98)
        if stop_loss >= entry_price:
            stop_loss = entry_price - (current_atr * atr_mult)

        sl_distance = entry_price - stop_loss
        take_profit = entry_price + (sl_distance * target_rr)

        risk_amt = float(account_balance) * (risk_pct / 100.0)
        position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0

        return {
            "action": "BUY", "order_type": "MARKET", "symbol": symbol, "direction": "LONG",
            "regime": "AA_VWAP_VP_STRICT_LONG",
            "entry_price": format_price(entry_price),
            "stop_loss": format_price(stop_loss),
            "take_profit": format_price(take_profit),
            "rr_ratio": float(target_rr),
            "position_size": float(position_size),
            "reason": "AA-VWAP Support + Volume Contraction/Spike + Strict Bullish Rejection"
        }

    # -----------------------------------------------------------------------
    # RULE SET B: SHORT ENTRY (Strict Confluence: Resistance + Shooting Star/Engulfing)
    # -----------------------------------------------------------------------
    trend_ok_short = (not use_ema_200_filter) or (current_price < ema_200)

    structural_short_resistance = (
        (abs(current_price - upper_1) / upper_1 < 0.006) or
        (abs(current_price - upper_2) / upper_2 < 0.006) or
        (abs(current_price - aa_vwap) / aa_vwap < 0.004) or
        (abs(current_price - vp_levels["poc"]) / vp_levels["poc"] < 0.006) or
        (current_price > upper_1 and last_row["close"] < upper_1) # Failed upthrust / sweep
    )

    bearish_candle_confirmation = bool(last_row["is_shooting_star"] or last_row["is_bearish_engulfing"] or (current_price > upper_1 and last_row["close"] < upper_1))

    if structural_short_resistance and trend_ok_short and bearish_candle_confirmation:
        entry_price = current_price
        
        local_swing_high = float(df['high'].tail(5).max())
        stop_loss = max(local_swing_high + (current_atr * 0.5), upper_2 if not math.isnan(upper_2) else entry_price * 1.02)
        if stop_loss <= entry_price:
            stop_loss = entry_price + (current_atr * atr_mult)

        sl_distance = stop_loss - entry_price
        take_profit = entry_price - (sl_distance * target_rr)

        risk_amt = float(account_balance) * (risk_pct / 100.0)
        position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0

        return {
            "action": "SELL", "order_type": "MARKET", "symbol": symbol, "direction": "SHORT",
            "regime": "AA_VWAP_VP_STRICT_SHORT",
            "entry_price": format_price(entry_price),
            "stop_loss": format_price(stop_loss),
            "take_profit": format_price(take_profit),
            "rr_ratio": float(target_rr),
            "position_size": float(position_size),
            "reason": "AA-VWAP Resistance + Volume Contraction/Spike + Strict Bearish Rejection"
        }

    no_signal["reason"] = "No valid strict confluence setup"
    return no_signal


def load_symbol_config(symbol: str) -> dict:
    return get_symbol_overrides(symbol)

evaluate_signals = evaluate_scalp_signals