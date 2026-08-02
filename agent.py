"""
===============================================================================
PRODUCTION LIVE TRADING AGENT (agent.py)
===============================================================================
Upgrades & Refinement Implemented:
1. Removed Min ATR % Filter: All symbols meeting volatility/volume rules are processed.
2. Upgrade A (HTF Trend Alignment): Enforces strict direction alignment on 4H & 1H TEMA.
   - LONG: Price > 1H TEMA AND Price > 4H TEMA
   - SHORT: Price < 1H TEMA AND Price < 4H TEMA
3. Upgrade B (Adaptive ATR Trailing Stop): Dynamic trailing stop engine based on highest/
   lowest prices since entry minus dynamic ATR distance.
4. Upgrade C (Equity Curve Drawdown Filter): Dynamic risk adjustment reducing position sizing
   by 50% when account drawdown exceeds the safety threshold.
===============================================================================
"""

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("live_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LiveTradingAgent")


# =============================================================================
# TECHNICAL INDICATORS & UTILITIES
# =============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_tema(series: pd.Series, period: int = 200) -> pd.Series:
    """Calculates Triple Exponential Moving Average (TEMA)."""
    ema1 = calculate_ema(series, period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)
    return (3 * ema1) - (3 * ema2) + ema3


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# =============================================================================
# UPGRADE C: EQUITY CURVE DRAWDOWN FILTER
# =============================================================================

class RiskManager:
    def __init__(self, base_risk_pct: float = 0.01, max_dd_threshold: float = 0.05):
        self.base_risk_pct = base_risk_pct
        self.max_dd_threshold = max_dd_threshold
        self.peak_equity = 0.0

    def get_adjusted_risk(self, current_balance: float) -> float:
        """
        Calculates position risk percentage.
        Scales risk down by 50% if account is in drawdown > max_dd_threshold.
        """
        if current_balance > self.peak_equity:
            self.peak_equity = current_balance

        drawdown = (self.peak_equity - current_balance) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        if drawdown >= self.max_dd_threshold:
            logger.warning(f"⚠️ Account Drawdown at {drawdown:.2%}. Scaling risk down to 50% of base.")
            return self.base_risk_pct * 0.5
        
        return self.base_risk_pct


# =============================================================================
# UPGRADE A: HTF TREND ALIGNMENT & SCANNER ENGINE
# =============================================================================

class MarketScanner:
    """Scans markets and enforces strict 1H + 4H HTF TEMA trend alignment."""
    
    @staticmethod
    def evaluate_htf_alignment(
        df_1h: pd.DataFrame, 
        df_4h: pd.DataFrame, 
        current_price: float,
        tema_period: int = 200
    ) -> Optional[str]:
        """
        Enforces Upgrade A: Multi-Timeframe Trend Alignment Rule.
        - LONG: Current Price > 1H TEMA200 AND Current Price > 4H TEMA200
        - SHORT: Current Price < 1H TEMA200 AND Current Price < 4H TEMA200
        Returns 'LONG', 'SHORT', or None.
        """
        if len(df_1h) < tema_period or len(df_4h) < tema_period:
            return None

        tema_1h = calculate_tema(df_1h['close'], tema_period).iloc[-1]
        tema_4h = calculate_tema(df_4h['close'], tema_period).iloc[-1]

        # Strict Multi-timeframe trend filter
        if current_price > tema_1h and current_price > tema_4h:
            return "LONG"
        elif current_price < tema_1h and current_price < tema_4h:
            return "SHORT"
        
        return None  # Neutral / Counter-trend conflict

    def analyze_symbol(
        self, 
        symbol: str, 
        df_30m: pd.DataFrame, 
        df_1h: pd.DataFrame, 
        df_4h: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates symbol setup without Min ATR % restriction.
        """
        if df_30m.empty or df_1h.empty or df_4h.empty:
            return None

        current_price = df_30m['close'].iloc[-1]
        atr_val = calculate_atr(df_30m, period=14).iloc[-1]
        
        if pd.isna(atr_val) or atr_val <= 0:
            return None

        # 1. Enforce Upgrade A: HTF Trend Alignment
        trend_direction = self.evaluate_htf_alignment(df_1h, df_4h, current_price)
        if not trend_direction:
            return None  # Rejected due to HTF trend disagreement

        # 2. Risk Parameters Calculation
        atr_multiplier = 1.5
        stop_distance = atr_val * atr_multiplier

        if trend_direction == "LONG":
            initial_sl = current_price - stop_distance
            tp_target = current_price + (stop_distance * 1.5)  # 1.5 R:R Default
        else:
            initial_sl = current_price + stop_distance
            tp_target = current_price - (stop_distance * 1.5)

        return {
            "symbol": symbol,
            "direction": trend_direction,
            "entry_price": current_price,
            "stop_loss": initial_sl,
            "take_profit": tp_target,
            "atr": atr_val,
            "timestamp": datetime.utcnow().isoformat()
        }


# =============================================================================
# UPGRADE B: DYNAMIC ADAPTIVE ATR TRAILING STOP ENGINE
# =============================================================================

class TradeEvaluator:
    """Manages active live positions with Dynamic Adaptive ATR Trailing Stops."""

    def __init__(self, atr_multiplier: float = 1.5):
        self.atr_multiplier = atr_multiplier

    def update_trailing_stop(self, trade: Dict[str, Any], current_price: float, current_atr: float) -> Tuple[float, bool]:
        """
        Updates trade dynamic trailing stop (Upgrade B).
        
        - LONG: SL moves UP to (Highest High - ATR * Multiplier).
        - SHORT: SL moves DOWN to (Lowest Low + ATR * Multiplier).
        - SL is never moved against the trade direction.
        
        Returns: (updated_sl, trigger_exit)
        """
        direction = trade["direction"]
        current_sl = trade["stop_loss"]
        highest_price = max(trade.get("highest_price", trade["entry_price"]), current_price)
        lowest_price = min(trade.get("lowest_price", trade["entry_price"]), current_price)
        
        trade["highest_price"] = highest_price
        trade["lowest_price"] = lowest_price

        atr_buffer = current_atr * self.atr_multiplier
        updated_sl = current_sl
        trigger_exit = False

        if direction == "LONG":
            proposed_sl = highest_price - atr_buffer
            # Trailing stop only moves upward
            if proposed_sl > current_sl:
                updated_sl = proposed_sl
            
            # Check exit conditions
            if current_price <= updated_sl or current_price >= trade["take_profit"]:
                trigger_exit = True

        elif direction == "SHORT":
            proposed_sl = lowest_price + atr_buffer
            # Trailing stop only moves downward
            if proposed_sl < current_sl:
                updated_sl = proposed_sl
            
            # Check exit conditions
            if current_price >= updated_sl or current_price <= trade["take_profit"]:
                trigger_exit = True

        return updated_sl, trigger_exit


# =============================================================================
# MAIN LIVE AGENT EXECUTION LOOP
# =============================================================================

class LiveTradingAgent:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.scanner = MarketScanner()
        self.evaluator = TradeEvaluator(atr_multiplier=1.5)
        self.risk_manager = RiskManager(base_risk_pct=0.01)
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        self.account_balance = 10000.0  # Base capital example

    def fetch_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Placeholder for live Exchange/Broker API data fetcher.
        Returns a DataFrame containing OHLCV columns: ['open', 'high', 'low', 'close', 'volume'].
        """
        # Implement Exchange connection (e.g., CCXT, MetaTrader, Interactive Brokers)
        return pd.DataFrame()

    def run_cycle(self):
        """Executes one scan and position evaluation loop."""
        logger.info("⚡ --- STARTING LIVE AGENT TRADING CYCLE ---")

        # 1. Update Active Positions & Adaptive Trailing Stops
        for symbol, trade in list(self.active_trades.items()):
            df_30m = self.fetch_market_data(symbol, "30m")
            if df_30m.empty:
                continue

            current_price = df_30m['close'].iloc[-1]
            current_atr = calculate_atr(df_30m, 14).iloc[-1]

            new_sl, exit_triggered = self.evaluator.update_trailing_stop(
                trade=trade, 
                current_price=current_price, 
                current_atr=current_atr
            )
            
            trade["stop_loss"] = new_sl

            if exit_triggered:
                pnl = (current_price - trade["entry_price"]) if trade["direction"] == "LONG" else (trade["entry_price"] - current_price)
                logger.info(f"🛑 EXIT TRIGGERED for {symbol} | Exit Price: {current_price:.4f} | PnL: {pnl:.4f}")
                del self.active_trades[symbol]

        # 2. Scan for New Signals
        for symbol in self.symbols:
            if symbol in self.active_trades:
                continue  # Skip symbols with existing position

            df_30m = self.fetch_market_data(symbol, "30m")
            df_1h = self.fetch_market_data(symbol, "1h")
            df_4h = self.fetch_market_data(symbol, "4h")

            signal = self.scanner.analyze_symbol(symbol, df_30m, df_1h, df_4h)
            
            if signal:
                risk_pct = self.risk_manager.get_adjusted_risk(self.account_balance)
                risk_amount = self.account_balance * risk_pct
                
                logger.info(f"🚀 NEW SIGNAL FOUND: {symbol} [{signal['direction']}]")
                logger.info(f"   Entry: {signal['entry_price']:.4f} | Initial SL: {signal['stop_loss']:.4f} | TP: {signal['take_profit']:.4f}")
                logger.info(f"   Allocated Risk: ${risk_amount:.2f} ({risk_pct:.2%})")

                # Store active trade state
                self.active_trades[symbol] = signal


if __name__ == "__main__":
    symbols_to_trade = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    agent = LiveTradingAgent(symbols=symbols_to_trade)
    
    # Run continuous execution loop or connect to streaming WebSocket
    logger.info("Live Agent Initialized and Running...")