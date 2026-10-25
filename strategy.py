import os
import time
import math
import asyncio
import logging
import json
import ccxt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

logger = logging.getLogger("scalp_strategy_engine")

# ---------------------------------------------------------------------------
# GLOBAL SCALPING TIMEFRAME & PARAMETERS
# ---------------------------------------------------------------------------
EXECUTION_TIMEFRAME = "15m"      # Configurable to "5m" or "15m"
FAST_EMA_PERIOD = 20
SLOW_EMA_PERIOD = 50
RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14                  # Added for ATR tracking support
MIN_ADX_THRESHOLD = 20.0         # Filters out ranging/choppy markets
MAX_ADX_THRESHOLD = 42.0         # Upper ADX ceiling filter (prevents late-stage trend entry)
TARGET_PROFIT_PCT = 0.02         # 2% target market movement (1% to 3% range)
MAX_SL_PCT = 0.010               # Calibrated stop loss percentage


# ---------------------------------------------------------------------------
# 1. TOP 40 VOLUME SCANNER & CONFIG HELPERS
# ---------------------------------------------------------------------------

def fetch_top_volume_symbols(limit: int = 40) -> List[str]:
    """Scans Binance or Bybit USDT linear futures for the top N coins by 24h quote volume."""
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
                
        # Sort descending by 24h volume
        usdt_futures.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [item[0] for item in usdt_futures[:limit]]
        logger.info(f"Successfully fetched top {len(top_symbols)} volume symbols.")
        return top_symbols
    except Exception as e:
        logger.error(f"Failed to scan top volume symbols: {e}")
        # Fallback default scalping list
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT"]


def load_symbol_config(symbol: str = "") -> dict:
    """Loads configuration settings for symbols from strategy_config.json with safe defaults."""
    default_config = {
        "account_balance": 100.0,
        "risk_pct": 1.0,
        "leverage": 5,
        "tema_period": 200,
        "rsi_period": 14,
        "rsi_thresh": 42.0,
        "adx_period": 14,
        "adx_threshold": 20.0,
        "use_adx_filter": True,
        "max_sl_pct": 0.01,
        "atr_period": 14,
        "atr_mult": 2.0,
        "use_atr_sl": True,
        "disable_htf": False
    }
    if os.path.exists("strategy_config.json"):
        try:
            with open("strategy_config.json", "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    default_config.update(data)
        except Exception as e:
            logger.warning(f"Could not load strategy_config.json: {e}")
    return default_config


def fetch_klines(symbol: str, interval: str = "15m", limit: int = 600) -> pd.DataFrame:
    """Fetches historical kline OHLCV data for live orchestration loops."""
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
# 2. TECHNICAL INDICATORS FOR SCALPING
# ---------------------------------------------------------------------------

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_tema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Triple Exponential Moving Average (TEMA)."""
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
    """Calculates Average True Range (ATR) for trailing stop management."""
    if df.empty or len(df) < period + 1:
        return pd.Series(0.0, index=df.index)
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().fillna(0.0)

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average Directional Index (ADX) to measure trend strength."""
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

def check_volume_spike(df: pd.DataFrame, period: int = 20, multiplier: float = 1.15) -> bool:
    """Ensures trigger volume is significantly higher than the average volume."""
    if "volume" not in df.columns or len(df) < period:
        return True
    vol_sma = df['volume'].rolling(period).mean().iloc[-1]
    last_volume = df['volume'].iloc[-1]
    if np.isnan(vol_sma) or vol_sma == 0:
        return True
    return bool(last_volume >= (vol_sma * multiplier))


# ---------------------------------------------------------------------------
# 3. SCALPING SIGNAL EVALUATION ENGINE (20/50 EMA + Trend/Momentum Filters)
# ---------------------------------------------------------------------------

def evaluate_scalp_signals(
    df: pd.DataFrame,
    symbol: str,
    account_balance: float = 100.0,
    risk_pct: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    no_signal = {
        "action": "HOLD", "symbol": symbol, "direction": "NONE",
        "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
        "reason": "No condition met"
    }

    if df.empty or len(df) < 60:
        no_signal["reason"] = "Insufficient kline data"
        return no_signal

    # Compute Indicators
    df = df.copy()
    df["ema_fast"] = calculate_ema(df["close"], FAST_EMA_PERIOD)
    df["ema_slow"] = calculate_ema(df["close"], SLOW_EMA_PERIOD)
    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    df["adx"] = calculate_adx(df, ADX_PERIOD)
    df["atr"] = calculate_atr(df, ATR_PERIOD)

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    current_price = float(last_row["close"])
    fast_ema = float(last_row["ema_fast"])
    slow_ema = float(last_row["ema_slow"])
    prev_fast = float(prev_row["ema_fast"])
    prev_slow = float(prev_row["ema_slow"])
    
    current_rsi = float(last_row["rsi"])
    current_adx = float(last_row["adx"])

    # 1. Momentum and ADX Bandpass Filter (20 <= ADX <= 42)
    if current_adx < MIN_ADX_THRESHOLD:
        no_signal["reason"] = f"Trend Filter: ADX ({current_adx:.1f}) too low (market is chopping)"
        return no_signal

    if current_adx > MAX_ADX_THRESHOLD:
        no_signal["reason"] = f"Exhaustion Filter: ADX ({current_adx:.1f}) exceeds ceiling threshold ({MAX_ADX_THRESHOLD})"
        return no_signal

    if not check_volume_spike(df, period=20, multiplier=1.15):
        no_signal["reason"] = "Volume Gate: Insufficient volume spike on execution candle"
        return no_signal

    # 2. EMA Crossover Mechanics (20 crosses 50)
    bullish_crossover = (prev_fast <= prev_slow) and (fast_ema > slow_ema)
    bearish_crossover = (prev_fast >= prev_slow) and (fast_ema < slow_ema)

    # -----------------------------------------------------------------------
    # LONG SCALP SETUP
    # -----------------------------------------------------------------------
    if bullish_crossover and current_rsi > 50.0 and current_rsi < 75.0:
        entry_price = current_price
        stop_loss = entry_price * (1.0 - MAX_SL_PCT)
        take_profit = entry_price * (1.0 + TARGET_PROFIT_PCT)
        
        sl_distance = entry_price - stop_loss
        risk_amt = float(account_balance) * (risk_pct / 100.0)
        position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0
        computed_rr = (take_profit - entry_price) / sl_distance

        return {
            "action": "BUY", "order_type": "MARKET", "symbol": symbol, "direction": "LONG",
            "regime": "EMA_SCALP_5M_15M",
            "entry_price": float(round(entry_price, 4)),
            "stop_loss": float(round(stop_loss, 4)),
            "take_profit": float(round(take_profit, 4)),
            "rr_ratio": float(round(computed_rr, 2)),
            "position_size": float(position_size),
            "entry_rsi": float(round(current_rsi, 2)),
            "entry_adx": float(round(current_adx, 2)),
            "reason": f"Bullish EMA Cross + RSI ({current_rsi:.1f}) + ADX ({current_adx:.1f}): TP={take_profit}"
        }

    # -----------------------------------------------------------------------
    # SHORT SCALP SETUP
    # -----------------------------------------------------------------------
    if bearish_crossover and current_rsi < 50.0 and current_rsi > 25.0:
        entry_price = current_price
        stop_loss = entry_price * (1.0 + MAX_SL_PCT)
        take_profit = entry_price * (1.0 - TARGET_PROFIT_PCT)
        
        sl_distance = stop_loss - entry_price
        risk_amt = float(account_balance) * (risk_pct / 100.0)
        position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0
        computed_rr = (entry_price - take_profit) / sl_distance

        return {
            "action": "SELL", "order_type": "MARKET", "symbol": symbol, "direction": "SHORT",
            "regime": "EMA_SCALP_5M_15M",
            "entry_price": float(round(entry_price, 4)),
            "stop_loss": float(round(stop_loss, 4)),
            "take_profit": float(round(take_profit, 4)),
            "rr_ratio": float(round(computed_rr, 2)),
            "position_size": float(position_size),
            "entry_rsi": float(round(current_rsi, 2)),
            "entry_adx": float(round(current_adx, 2)),
            "reason": f"Bearish EMA Cross + RSI ({current_rsi:.1f}) + ADX ({current_adx:.1f}): TP={take_profit}"
        }

    no_signal["reason"] = "No valid 20/50 EMA crossover with momentum filters met"
    return no_signal


# ---------------------------------------------------------------------------
# MODULE-LEVEL ALIASES FOR MAIN.PY COMPATIBILITY
# ---------------------------------------------------------------------------
evaluate_signals = evaluate_scalp_signals