import logging
from typing import Dict, Any
from common import get_db_connection, release_db_connection

logger = logging.getLogger("dynamic_trade_manager")


class DynamicTradeManager:
    """
    Handles dynamic active position management, trailing stops,
    and breakeven locking based on real-time price updates and exchange state.
    """
    def __init__(self, trailing_mult: float = 1.5, be_rr_trigger: float = 1.0):
        self.trailing_mult = trailing_mult
        self.be_rr_trigger = be_rr_trigger

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert database values to float, handling NoneType and invalid types."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _update_db_sl_tp(self, trade_id: int, stop_loss: float, take_profit: float):
        """Persists newly calculated default SL and TP directly to the DB."""
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE trade_setups 
                        SET stop_loss = %s, take_profit = %s 
                        WHERE id = %s;
                        """,
                        (stop_loss, take_profit, trade_id)
                    )
                    conn.commit()
                    logger.info(f"Persisted calculated SL (${stop_loss:.4f}) and TP (${take_profit:.4f}) to DB for Trade #{trade_id}")
            except Exception as e:
                logger.error(f"Failed to update DB SL/TP for trade #{trade_id}: {e}")
            finally:
                release_db_connection(conn)

    def process_trade(self, trade: Dict[str, Any], latest_candle: Dict[str, Any], live_pos_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluates an open trade against market candle data and Bybit exchange state.
        """
        if not trade or not latest_candle:
            return {"action": "HOLD"}

        trade_id = trade.get("id")
        direction = str(trade.get("direction") or "").upper()
        
        entry_price = self._safe_float(trade.get("entry_price"))
        current_sl = self._safe_float(trade.get("stop_loss"))
        current_tp = self._safe_float(trade.get("take_profit"))
        trade_state = trade.get("trade_state") or "OPEN"

        close_price = self._safe_float(latest_candle.get("close"))
        high_price = self._safe_float(latest_candle.get("high"), close_price)
        low_price = self._safe_float(latest_candle.get("low"), close_price)
        atr = self._safe_float(latest_candle.get("atr"))

        if entry_price <= 0 or close_price <= 0:
            return {"action": "HOLD"}

        is_long = direction in ["BUY", "LONG"]

        # Auto-calculate and persist missing dynamic SL and TP to Database
        db_needs_update = False
        if current_sl == 0.0 and entry_price > 0:
            current_sl = entry_price * 0.98 if is_long else entry_price * 1.02
            db_needs_update = True

        if current_tp == 0.0 and entry_price > 0:
            risk = abs(entry_price - current_sl)
            current_tp = entry_price + (risk * 2.5) if is_long else entry_price - (risk * 2.5)
            db_needs_update = True

        if db_needs_update and trade_id is not None:
            self._update_db_sl_tp(trade_id, current_sl, current_tp)

        # 1. Exchange Position Closure Check (Detects if Bybit already closed via Market SL/TP)
        if live_pos_info and not live_pos_info.get("error") and live_pos_info.get("contracts", 0.0) <= 0.001:
            return {
                "action": "SYNC_CLOSED_FROM_EXCHANGE",
                "msg": f"⚠️ Trade #{trade_id} closed on Bybit exchange. Syncing DB state..."
            }

        # 2. Hard Trigger Checks: Send Market Order to Exchange instead of assuming hypothetical fill
        if is_long:
            if current_tp > 0 and high_price >= current_tp:
                return {
                    "action": "EXECUTE_CLOSE_TP",
                    "target_price": current_tp,
                    "msg": f"🎯 Target TP hit for #{trade_id}. Executing market close on Bybit..."
                }
            if current_sl > 0 and low_price <= current_sl:
                return {
                    "action": "EXECUTE_CLOSE_SL",
                    "target_price": current_sl,
                    "msg": f"🛑 Target SL hit for #{trade_id}. Executing market close on Bybit..."
                }
        else:  # SHORT
            if current_tp > 0 and low_price <= current_tp:
                return {
                    "action": "EXECUTE_CLOSE_TP",
                    "target_price": current_tp,
                    "msg": f"🎯 Target TP hit for #{trade_id}. Executing market close on Bybit..."
                }
            if current_sl > 0 and high_price >= current_sl:
                return {
                    "action": "EXECUTE_CLOSE_SL",
                    "target_price": current_sl,
                    "msg": f"🛑 Target SL hit for #{trade_id}. Executing market close on Bybit..."
                }

        # 3. Dynamic Trailing Stop & Breakeven Adjustments
        if atr > 0:
            risk_dist = abs(entry_price - current_sl) if current_sl > 0 else (entry_price * 0.02)

            if is_long:
                unrealized_profit = close_price - entry_price

                # Move to Breakeven
                if trade_state == "OPEN" and unrealized_profit >= (risk_dist * self.be_rr_trigger):
                    new_sl = entry_price + (atr * 0.1)
                    if new_sl > current_sl:
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": new_sl,
                            "new_state": "BE_LOCKED",
                            "msg": f"🔒 Trade #{trade_id} moved to Breakeven @ ${new_sl:.5f}"
                        }

                # Trailing Stop Update
                if trade_state in ["BE_LOCKED", "TRAILING"]:
                    trail_sl = close_price - (atr * self.trailing_mult)
                    if trail_sl > current_sl:
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": trail_sl,
                            "new_state": "TRAILING",
                            "msg": f"📈 Trailing Stop updated for Trade #{trade_id} to ${trail_sl:.5f}"
                        }

            else:  # SHORT
                unrealized_profit = entry_price - close_price

                # Move to Breakeven
                if trade_state == "OPEN" and unrealized_profit >= (risk_dist * self.be_rr_trigger):
                    new_sl = entry_price - (atr * 0.1)
                    if current_sl == 0 or new_sl < current_sl:
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": new_sl,
                            "new_state": "BE_LOCKED",
                            "msg": f"🔒 Trade #{trade_id} moved to Breakeven @ ${new_sl:.5f}"
                        }

                # Trailing Stop Update
                if trade_state in ["BE_LOCKED", "TRAILING"]:
                    trail_sl = close_price + (atr * self.trailing_mult)
                    if current_sl == 0 or trail_sl < current_sl:
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": trail_sl,
                            "new_state": "TRAILING",
                            "msg": f"📉 Trailing Stop updated for Trade #{trade_id} to ${trail_sl:.5f}"
                        }

        return {"action": "HOLD"}