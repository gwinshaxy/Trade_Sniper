import logging
import ccxt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

logger = logging.getLogger("breakout_strategy_engine")

# ---------------------------------------------------------------------------
# GLOBAL TREND-BREAKOUT TIMEFRAME & UNIVERSAL PARAMETERS (4H VALIDATED)
# ---------------------------------------------------------------------------
EXECUTION_TIMEFRAME = "4h"       # Validated 4H timeframe
DONCHIAN_PERIOD = 48             # 48 bars on 4H = 8-day structural breakout channel
EMA_TREND_PERIOD = 200           # 200-bar macro trend filter
VOLUME_SMA_PERIOD = 20
ADX_PERIOD = 14
MIN_ADX_THRESHOLD = 18.0         # Allow breakouts emerging from quiet consolidations
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 2.5          # Initial stop-loss distance
ATR_TRAIL_MULTIPLIER = 3.0       # Dynamic Chandelier trailing stop distance
TRAIL_ACTIVATION_R = 1.5         # Activate Chandelier trail once trade reaches +1.5R
ROUND_TRIP_FEE_PCT = 0.0010      
MIN_DAILY_QUOTE_VOLUME = 20_000_000.0  # $20 Million minimum daily quote volume filter[cite: 22]


def get_symbol_overrides(symbol: str) -> dict:
    """Universal parameters across all assets — zero per-coin curve-fitting."""
    return {
        "atr_sl_multiplier": ATR_SL_MULTIPLIER,
        "atr_trail_multiplier": ATR_TRAIL_MULTIPLIER,
        "trail_activation_r": TRAIL_ACTIVATION_R,
        "use_ema_200_filter": True
    }


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


def fetch_klines(symbol: str, interval: str = EXECUTION_TIMEFRAME, limit: int = 300) -> pd.DataFrame:
    """Fetches OHLCV klines from Bybit Linear Futures for live signal & ATR evaluation."""
    try:
        clean_symbol = symbol.split(":")[0]
        if not clean_symbol.endswith("USDT") and "/" not in clean_symbol:
            clean_symbol = f"{clean_symbol}/USDT"
        if ":" not in clean_symbol:
            clean_symbol = f"{clean_symbol}:USDT"

        exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})
        ohlcv = exchange.fetch_ohlcv(clean_symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logger.error(f"[{symbol}] Failed to fetch {interval} klines: {e}")
        return pd.DataFrame()


def fetch_top_volume_symbols(limit: int = 10, min_daily_volume: float = MIN_DAILY_QUOTE_VOLUME) -> List[str]:
    """
    Scans Bybit/Binance Linear Futures for top volume symbols, filtering out non-crypto perpetuals 
    and enforcing a minimum daily quote volume floor to eliminate low-liquidity noise.
    """
    excluded_prefixes = (
        "XAU", "XAG", "CL", "QQQ", "SPY", "NVDA", "AMD", "MU", "INTC",
        "SOXL", "SOXS", "SNDK", "MSTR", "SAMSUNG", "SKHYNIX", "SKHY"
    )
    try:
        exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}})
        tickers = exchange.fetch_tickers()
        usdt_futures = []
        for symbol, data in tickers.items():
            if 'USDT' in symbol:
                clean_sym = symbol.split(':')[0]
                if not clean_sym.endswith('USDT') and '/' not in clean_sym:
                    clean_sym = f"{clean_sym}/USDT"
                base = clean_sym.split('/')[0].upper()
                
                if base.startswith(excluded_prefixes):
                    continue
                
                quote_volume = data.get('quoteVolume', 0) or 0
                
                # Apply minimum daily volume filter floor[cite: 22]
                if quote_volume < min_daily_volume:
                    continue
                    
                usdt_futures.append((clean_sym, quote_volume))
                
        usdt_futures.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        unique_symbols = []
        for sym, _ in usdt_futures:
            if sym not in seen:
                seen.add(sym)
                unique_symbols.append(sym)
                
        return unique_symbols[:limit]
    except Exception as e:
        logger.error(f"Failed to scan top volume symbols from Bybit: {e}")
        try:
            # Fallback to Binance USDM if Bybit ticker fetch fails
            exchange_binance = ccxt.binanceusdm({'enableRateLimit': True})
            tickers = exchange_binance.fetch_tickers()
            usdt_futures = []
            for symbol, data in tickers.items():
                if '/USDT:USDT' in symbol or (symbol.endswith('USDT') and ':' not in symbol):
                    clean_sym = symbol.split(':')[0]
                    if not clean_sym.endswith('USDT'):
                        clean_sym = f"{clean_sym}/USDT"
                    base = clean_sym.split('/')[0].upper()
                    if base.startswith(excluded_prefixes):
                        continue
                    quote_volume = data.get('quoteVolume', 0) or 0
                    if quote_volume < min_daily_volume:
                        continue
                    usdt_futures.append((clean_sym, quote_volume))
            usdt_futures.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            unique_symbols = []
            for sym, _ in usdt_futures:
                if sym not in seen:
                    seen.add(sym)
                    unique_symbols.append(sym)
            return unique_symbols[:limit]
        except Exception as ex_binance:
            logger.error(f"Fallback Binance scan also failed: {ex_binance}")
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT", "NEAR/USDT", "UNI/USDT", "BNB/USDT"]


# ---------------------------------------------------------------------------
# TECHNICAL INDICATORS
# ---------------------------------------------------------------------------

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


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


def precompute_indicators(df: pd.DataFrame, df_1h: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Precomputes Donchian Channels, ATR, ADX, and clean breakout rules."""
    df = df.copy()
    df["ema_200"] = calculate_ema(df["close"], EMA_TREND_PERIOD)
    df["vol_sma20"] = df["volume"].rolling(VOLUME_SMA_PERIOD).mean()
    df["adx"] = calculate_adx(df, ADX_PERIOD)
    df["atr"] = calculate_atr(df, period=ATR_PERIOD)
    df["atr_sma20"] = df["atr"].rolling(20).mean()

    # Shift by 1 bar so we compare current close against the PRIOR 48-bar range (no lookahead bias)
    df["donchian_high"] = df["high"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["donchian_low"] = df["low"].rolling(DONCHIAN_PERIOD).min().shift(1)

    df["regime_valid"] = df["adx"] >= MIN_ADX_THRESHOLD
    df["vol_breakout"] = df["volume"] >= df["vol_sma20"]

    df["bull_breakout"] = (df["close"] > df["donchian_high"]) & (df["close"] > df["open"])
    df["bear_breakout"] = (df["close"] < df["donchian_low"]) & (df["close"] < df["open"])

    return df


# ---------------------------------------------------------------------------
# LIVE SIGNAL EVALUATION ENGINE
# ---------------------------------------------------------------------------

def evaluate_breakout_signals(
    df: pd.DataFrame,
    symbol: str,
    account_balance: float = 100.0,
    risk_pct: float = 1.0,
    use_ema_200_filter: bool = True,
    custom_atr_mult: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    no_signal = {
        "action": "HOLD", "symbol": symbol, "direction": "NONE",
        "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
        "reason": "No condition met"
    }

    if df.empty or len(df) < 210:
        no_signal["reason"] = "Insufficient kline data for 200 EMA and Donchian indicators"
        return no_signal

    atr_mult = custom_atr_mult if custom_atr_mult is not None else ATR_SL_MULTIPLIER
    df = precompute_indicators(df)
    last_row = df.iloc[-1]

    current_price = float(last_row["close"])
    ema_200 = float(last_row["ema_200"])
    current_adx = float(last_row["adx"])
    current_atr = float(last_row["atr"])

    if not bool(last_row["regime_valid"]):
        no_signal["reason"] = f"Regime Gate: ADX ({current_adx:.1f}) < {MIN_ADX_THRESHOLD}"
        return no_signal

    if not bool(last_row["vol_breakout"]):
        no_signal["reason"] = "Volume Gate: Breakout volume below 20-SMA"
        return no_signal

    # LONG BREAKOUT SETUP (Dynamic Chandelier ATR Trail — no static TP ceiling)
    trend_long_ok = (not use_ema_200_filter) or (current_price > ema_200)
    if bool(last_row["bull_breakout"]) and trend_long_ok:
        entry_price = current_price
        stop_loss = entry_price - (current_atr * atr_mult)
        sl_distance = entry_price - stop_loss
        risk_amt = float(account_balance) * (risk_pct / 100.0)
        position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0

        return {
            "action": "BUY", "order_type": "MARKET", "symbol": symbol, "direction": "LONG",
            "regime": "4H_DONCHIAN_VOLATILITY_BREAKOUT",
            "entry_price": format_price(entry_price),
            "stop_loss": format_price(stop_loss),
            "take_profit": 0.0,
            "trailing_atr_mult": float(ATR_TRAIL_MULTIPLIER),
            "trail_activation_r": float(TRAIL_ACTIVATION_R),
            "position_size": float(position_size),
            "reason": f"4H Donchian-{DONCHIAN_PERIOD} High Breakout + Vol + ADX {current_adx:.1f}"
        }

    # SHORT BREAKOUT SETUP
    trend_short_ok = (not use_ema_200_filter) or (current_price < ema_200)
    if bool(last_row["bear_breakout"]) and trend_short_ok:
        entry_price = current_price
        stop_loss = entry_price + (current_atr * atr_mult)
        sl_distance = stop_loss - entry_price
        risk_amt = float(account_balance) * (risk_pct / 100.0)
        position_size = round(risk_amt / sl_distance, 6) if sl_distance > 0 else 0.0

        return {
            "action": "SELL", "order_type": "MARKET", "symbol": symbol, "direction": "SHORT",
            "regime": "4H_DONCHIAN_VOLATILITY_BREAKOUT",
            "entry_price": format_price(entry_price),
            "stop_loss": format_price(stop_loss),
            "take_profit": 0.0,
            "trailing_atr_mult": float(ATR_TRAIL_MULTIPLIER),
            "trail_activation_r": float(TRAIL_ACTIVATION_R),
            "position_size": float(position_size),
            "reason": f"4H Donchian-{DONCHIAN_PERIOD} Low Breakdown + Vol + ADX {current_adx:.1f}"
        }

    no_signal["reason"] = "No valid Donchian breakout matching macro trend rules"
    return no_signal


evaluate_signals = evaluate_breakout_signals
load_symbol_config = get_symbol_overrides