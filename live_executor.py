import os
import logging
import ccxt
from common import get_db_connection, execute_query, release_db_connection

logger = logging.getLogger("live_executor")

DEFAULT_LEVERAGE = 10
MIN_NOTIONAL = 5.0
RISK_PER_TRADE_PCT = 0.01

def safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def get_account_balance() -> float:
    return float(os.getenv("ACCOUNT_BALANCE", 100.0))  

def calculate_position_size(entry_price: float, atr_val: float, leverage: int = DEFAULT_LEVERAGE, max_trade_pct: float = 0.25) -> float:
    account_balance = get_account_balance()
    # Limit maximum notional per asset to its assigned percentage of total wallet
    effective_slot_balance = account_balance * max_trade_pct
    risk_usd = effective_slot_balance * RISK_PER_TRADE_PCT
    
    stop_distance = atr_val * 1.5 if atr_val > 0 else entry_price * 0.02
    if stop_distance == 0:
        return 0.0

    raw_qty = risk_usd / stop_distance
    notional_usd = raw_qty * entry_price
    
    max_notional_allowed = effective_slot_balance * leverage * 0.95
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
    """Execution engine with robust multi-symbol format database settlement."""
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

    def format_precision_amount(self, symbol: str, amount: float) -> float:
        """Safely rounds order quantity using exchange market precision or standard rounding."""
        try:
            if self.exchange.markets and symbol in self.exchange.markets:
                return float(self.exchange.amount_to_precision(symbol, amount))
        except Exception as e:
            logger.debug(f"[{symbol}] Precision formatting fallback triggered: {e}")
        return round(amount, 4)

    def buy_spot_mexc(self, symbol: str, amount_usd: float) -> dict:
        clean_symbol = symbol.replace("-", "/").replace("_", "")
        if "/" not in clean_symbol and clean_symbol.endswith("USDT"):
            clean_symbol = f"{clean_symbol[:-4]}/USDT"

        logger.info(f"[{clean_symbol}] Executing Spot Market Buy for ${amount_usd:.2f} USD")
        
        try:
            if not self.api_key or not self.api_secret:
                logger.warning(f"[{clean_symbol}] MEXC API credentials missing. Running in Simulation mode.")
                ticker = self.exchange.fetch_ticker(clean_symbol) if self.exchange else {"last": 1.0}
                fill_price = safe_float(ticker.get("last", 1.0))
                executed_qty = self.format_precision_amount(clean_symbol, amount_usd / fill_price) if fill_price > 0 else 0.0
                return {
                    "status": "SUCCESS",
                    "id": "SIMULATED_SPOT_ORDER",
                    "fill_price": fill_price,
                    "executed_qty": executed_qty
                }
                
            ticker = self.exchange.fetch_ticker(clean_symbol)
            fill_price = safe_float(ticker.get("last") or ticker.get("close"), 1.0)
            estimated_qty = self.format_precision_amount(clean_symbol, amount_usd / fill_price) if fill_price > 0 else 0.0

            try:
                order = self.exchange.create_market_buy_order(clean_symbol, estimated_qty, {'cost': amount_usd})
            except Exception as inner_e:
                logger.debug(f"[{clean_symbol}] Primary market buy failed ({inner_e}), trying fallback order placement...")
                order = self.exchange.create_order(clean_symbol, 'market', 'buy', estimated_qty)

            order_price = safe_float(order.get("price") or order.get("average"), default=fill_price)
            executed_qty = safe_float(order.get("filled") or order.get("amount"), default=estimated_qty)
            
            return {
                "status": "SUCCESS",
                "id": str(order.get("id", "EXECUTED")),
                "fill_price": order_price,
                "executed_qty": executed_qty
            }
        except Exception as e:
            logger.error(f"[{clean_symbol}] MEXC Spot Buy Execution Failed: {e}")
            return {"status": "FAILED", "reason": str(e)}

    def sell_spot_mexc(self, symbol: str, amount: float) -> dict:
        clean_symbol = symbol.replace("-", "/").replace("_", "")
        if "/" not in clean_symbol and clean_symbol.endswith("USDT"):
            clean_symbol = f"{clean_symbol[:-4]}/USDT"

        formatted_amount = self.format_precision_amount(clean_symbol, amount)
        logger.info(f"[{clean_symbol}] Executing Spot Market Sell for {formatted_amount} units")
        
        try:
            if not self.api_key or not self.api_secret:
                logger.warning(f"[{clean_symbol}] MEXC API credentials missing. Running in Simulation Exit mode.")
                ticker = self.exchange.fetch_ticker(clean_symbol) if self.exchange else {"last": 1.0}
                fill_price = safe_float(ticker.get("last", 1.0))
                return {
                    "status": "SUCCESS",
                    "id": "SIMULATED_SPOT_SELL_ORDER",
                    "fill_price": fill_price,
                    "executed_qty": formatted_amount
                }
                
            order = self.exchange.create_market_sell_order(clean_symbol, formatted_amount)
            order_price = safe_float(order.get("price") or order.get("average"), default=0.0)
            executed_qty = safe_float(order.get("filled") or order.get("amount"), default=formatted_amount)
            
            return {
                "status": "SUCCESS",
                "id": str(order.get("id", "CLOSED")),
                "fill_price": order_price,
                "executed_qty": executed_qty
            }
        except Exception as e:
            logger.error(f"[{clean_symbol}] MEXC Spot Sell Execution Failed: {e}")
            return {"status": "FAILED", "reason": str(e)}

    def close_live_position_mexc(self, symbol: str, position_size: float = None, current_price: float = None, outcome: str = None):
        """Executes exchange market sell order and updates DB state, exit price, and PnL metrics to CLOSED."""
        raw_symbol = symbol.upper().strip()
        entry_price = 0.0
        
        # 1. Fetch open position details from DB if missing
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT position_size, entry_price FROM trade_setups WHERE pair IN (%s, %s) AND trade_state = 'OPEN';",
                    (raw_symbol, raw_symbol.replace("/", ""))
                )
                res = cursor.fetchone()
                if res:
                    if position_size is None or position_size <= 0:
                        position_size = safe_float(res[0])
                    entry_price = safe_float(res[1])
                cursor.close()
            finally:
                release_db_connection(conn)

        # 2. Place spot market sell order on MEXC
        exit_price = current_price or 0.0
        if position_size and position_size > 0:
            logger.info(f"[{raw_symbol}] Initiating exchange spot sell for {position_size} units...")
            sell_res = self.sell_spot_mexc(symbol=raw_symbol, amount=position_size)
            if sell_res.get("status") != "SUCCESS":
                logger.error(f"[{raw_symbol}] Market sell failed on exchange: {sell_res.get('reason')}")
                return {"status": "FAILED", "reason": sell_res.get("reason")}
            
            if sell_res.get("fill_price") and sell_res["fill_price"] > 0:
                exit_price = sell_res["fill_price"]

        # 3. Calculate PnL Metrics
        pnl_usd = 0.0
        pnl_pct = 0.0
        if entry_price > 0 and exit_price > 0 and position_size > 0:
            pnl_usd = (exit_price - entry_price) * position_size
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        # 4. Update DB record state with exit logging
        clean_symbol = raw_symbol.replace("/", "").replace("-", "")
        formatted_slash = f"{clean_symbol[:-4]}/{clean_symbol[-4:]}" if clean_symbol.endswith("USDT") else raw_symbol

        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                query = """
                    UPDATE trade_setups
                    SET trade_state = 'CLOSED', 
                        status = 'CLOSED', 
                        closed_at = NOW(),
                        exit_price = %s,
                        pnl_usd = %s,
                        pnl_pct = %s,
                        outcome = %s
                    WHERE (pair = %s OR pair = %s OR pair = %s) AND trade_state = 'OPEN';
                """
                cursor.execute(query, (exit_price, pnl_usd, pnl_pct, outcome or "CLOSED", raw_symbol, clean_symbol, formatted_slash))
                conn.commit()
                cursor.close()
                logger.info(f"[{raw_symbol}] Position successfully updated to CLOSED in DB. PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%)")
                return {"status": "SUCCESS", "exit_price": exit_price, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct}
            except Exception as e:
                conn.rollback()
                logger.error(f"[{raw_symbol}] Failed to close position in DB: {e}")
                return {"status": "FAILED", "reason": str(e)}
            finally:
                release_db_connection(conn)
        return {"status": "FAILED", "reason": "No DB connection"}