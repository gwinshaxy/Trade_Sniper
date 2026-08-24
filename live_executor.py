import os
import logging
import ccxt
from common import get_db_connection, execute_query, release_db_connection

# Initialize Module Logger
logger = logging.getLogger("live_executor")

DEFAULT_LEVERAGE = 10
MIN_NOTIONAL = 5.0
RISK_PER_TRADE_PCT = 0.01

def get_account_balance() -> float:
    """Fetches account balance (can be updated to pull live MEXC USDT balance)."""
    return 100.0  

def calculate_position_size(entry_price: float, atr_val: float, leverage: int = DEFAULT_LEVERAGE) -> float:
    account_balance = get_account_balance()
    risk_usd = account_balance * RISK_PER_TRADE_PCT
    
    stop_distance = atr_val * 1.5 if atr_val > 0 else entry_price * 0.02
    if stop_distance == 0:
        return 0.0

    raw_qty = risk_usd / stop_distance
    notional_usd = raw_qty * entry_price
    
    max_notional_allowed = account_balance * leverage * 0.95
    if notional_usd > max_notional_allowed:
        notional_usd = max_notional_allowed
        raw_qty = notional_usd / entry_price

    if notional_usd < MIN_NOTIONAL:
        logger.warning(f"Calculated notional (${notional_usd:.2f}) below MIN_NOTIONAL (${MIN_NOTIONAL}). Adjusting up.")
        raw_qty = MIN_NOTIONAL / entry_price

    return round(raw_qty, 4)

def execute_signal(symbol: str, direction: str, entry_price: float, atr_val: float):
    qty = calculate_position_size(entry_price, atr_val)
    notional = qty * entry_price

    if qty <= 0:
        logger.error(f"[{symbol}] Invalid position quantity calculated: {qty}")
        return

    stop_loss = entry_price - (1.5 * atr_val) if direction == "LONG" else entry_price + (1.5 * atr_val)
    take_profit = entry_price + (3.0 * atr_val) if direction == "LONG" else entry_price - (3.0 * atr_val)

    logger.info(f"[{symbol}] Executing {direction}: Qty={qty}, Notional=${notional:.2f}, SL={stop_loss:.4f}, TP={take_profit:.4f}")

    conn = get_db_connection()
    try:
        # Changed 'quantity' to 'amount' to match database schema
        query = """
            INSERT INTO trade_setups (pair, direction, entry_price, amount, stop_loss, take_profit, status, trade_state, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN', NOW());
        """
        execute_query(conn, query, (symbol.replace("/", ""), direction, entry_price, qty, stop_loss, take_profit))
        logger.info(f"[{symbol}] Trade stored in trade_setups successfully.")
    except Exception as e:
        logger.error(f"[{symbol}] Failed to log trade execution in database: {e}")
    finally:
        release_db_connection(conn)

class LiveExecutionEngine:
    """Execution engine class expected by main_trading_engine.py and ws_monitor.py"""
    def __init__(self):
        self.api_key = os.getenv("MEXC_API_KEY", "")
        self.api_secret = os.getenv("MEXC_SECRET_KEY", "")
        
        self.exchange = ccxt.mexc({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        logger.info("LiveExecutionEngine initialized with CCXT MEXC integration.")

    def execute_trade(self, symbol: str, direction: str, entry_price: float, atr_val: float):
        """Standard signal execution route using ATR dynamic risk."""
        execute_signal(symbol, direction, entry_price, atr_val)

    # Inside LiveExecutionEngine class:
    def execute_live_order(self, symbol: str, side: str, amount: float, stop_loss: float = None, take_profit: float = None, entry_price: float = 0.0, risk_pct: float = 1.0):
        """Wrapper method invoked directly by main_trading_engine.py."""
        db_symbol = symbol.replace("/", "").upper()
        logger.info(f"Executing Live Order for {db_symbol}: Side={side}, Amount={amount}, Entry={entry_price}, SL={stop_loss}, TP={take_profit}, Risk%={risk_pct}")
        
        # Dynamic R:R Calculation
        risk_reward_ratio = 2.0
        if entry_price > 0 and stop_loss and take_profit:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            if risk > 0:
                risk_reward_ratio = round(reward / risk, 2)

        # Fetch Account Balance
        balance = get_account_balance()

        conn = get_db_connection()
        try:
            # Included both 'amount' and 'position_size' to satisfy schema requirements
            query = """
                INSERT INTO trade_setups (pair, direction, entry_price, amount, position_size, stop_loss, take_profit, risk_reward_ratio, account_balance, risk_pct, status, trade_state, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN', NOW());
            """
            execute_query(conn, query, (db_symbol, side.upper(), entry_price, amount, amount, stop_loss, take_profit, risk_reward_ratio, balance, risk_pct))
            logger.info(f"[{db_symbol}] Live order successfully logged with amount/position_size={amount}.")
            return {"status": "SUCCESS", "pair": db_symbol}
        except Exception as e:
            logger.error(f"[{db_symbol}] Failed to execute live order: {e}")
            return {"status": "FAILED", "reason": str(e)}
        finally:
            release_db_connection(conn)
    def close_live_position_mexc(self, symbol: str, position_side: str = "BOTH"):
        """Wrapper method invoked directly by main_trading_engine.py for position closes."""
        raw_symbol = symbol.upper()
        clean_symbol = raw_symbol.replace("/", "")
        logger.info(f"Closing Live Position for {raw_symbol} / {clean_symbol} (Side: {position_side})")
        
        conn = get_db_connection()
        try:
            # Updates trade_state, status, and records the exact closing timestamp
            # Handles both slash and non-slash pair variations stored in trade_setups
            query = """
                UPDATE trade_setups
                SET trade_state = 'CLOSED', status = 'CLOSED', closed_at = NOW()
                WHERE (pair = %s OR pair = %s) AND trade_state = 'OPEN';
            """
            execute_query(conn, query, (raw_symbol, clean_symbol))
            logger.info(f"[{clean_symbol}] Position successfully closed in trade_setups.")
            return {"status": "SUCCESS"}
        except Exception as e:
            logger.error(f"[{clean_symbol}] Failed to close position: {e}")
            return {"status": "FAILED", "reason": str(e)}
        finally:
            release_db_connection(conn)