import logging
from typing import Dict, Any
from common import get_db_connection, release_db_connection

logger = logging.getLogger("dynamic_trade_manager")


class DynamicTradeManager:
    """
    Handles dynamic active position management, trailing stops,
    and breakeven locking based on real-time price updates and exchange state.
    """

    def __init__(self, trailing_mult: float = 3.0, be_rr_trigger: float = 1.5, trail_rr_trigger: float = 2.0):
        # Increased defaults: wider ATR multiplier and delayed R:R triggers
        self.trailing_mult = trailing_mult
        self.be_rr_trigger = be_rr_trigger
        self.trail_rr_trigger = trail_rr_trigger

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _update_db_sl_tp(self, trade_id: int, stop_loss: float, take_profit: float):
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE trade_setups 
                        SET stop_loss = %s, take_profit = %s, updated_at = CURRENT_TIMESTAMP 
                        WHERE id = %s;
                    """, (stop_loss, take_profit, trade_id))
                    conn.commit()
                    logger.info(f"Persisted calculated SL (${stop_loss:.4f}) and TP (${take_profit:.4f}) to DB for Trade #{trade_id}")
            except Exception as e:
                logger.error(f"Failed to update DB SL/TP for trade #{trade_id}: {e}")
            finally:
                release_db_connection(conn)

    def process_trade(self, trade: Dict[str, Any], latest_candle: Dict[str, Any], live_pos_info: Dict[str, Any] = None) -> Dict[str, Any]:
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
        db_needs_update = False

        # Directional validation before processing updates (Guard against Inverted SL)
        if is_long:
            if current_sl >= entry_price and trade_state == "OPEN":
                current_sl = entry_price * 0.98  # Sanitize corrupt SL
                db_needs_update = True
        elif not is_long:
            if current_sl <= entry_price and current_sl > 0 and trade_state == "OPEN":
                current_sl = entry_price * 1.02  # Sanitize corrupt SL
                db_needs_update = True

        # Directionally correct fallback SL
        if current_sl == 0.0 and entry_price > 0:
            current_sl = entry_price * (1.0 - 0.02) if is_long else entry_price * (1.0 + 0.02)
            db_needs_update = True

        if current_tp == 0.0 and entry_price > 0:
            risk = abs(entry_price - current_sl)
            current_tp = entry_price + (risk * 2.5) if is_long else entry_price - (risk * 2.5)
            db_needs_update = True

        if db_needs_update and trade_id is not None:
            self._update_db_sl_tp(trade_id, current_sl, current_tp)

        # 1. Exchange Position Closure Check
        if live_pos_info and not live_pos_info.get("error") and live_pos_info.get("contracts", 0.0) <= 0.001:
            return {
                "action": "SYNC_CLOSED_FROM_EXCHANGE",
                "msg": f"⚠️ Trade #{trade_id} closed on Bybit exchange. Syncing DB state..."
            }

        # 2. Hard Trigger Checks: intrabar High/Low extremes
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

        # -------------------------------------------------------------------------
        # 3. Dynamic Trailing Stop & Breakeven Adjustments (Delayed R:R activation)
        # -------------------------------------------------------------------------
        if atr > 0:
            risk_dist = abs(entry_price - current_sl) if current_sl > 0 else (entry_price * 0.02)

            if is_long:
                unrealized_profit = close_price - entry_price

                # Lock Breakeven only after reaching be_rr_trigger (e.g., 1.5R)
                if trade_state == "OPEN" and unrealized_profit >= (risk_dist * self.be_rr_trigger):
                    candidate_sl = entry_price + (atr * 0.1)
                    if candidate_sl > entry_price and candidate_sl > current_sl:
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": candidate_sl,
                            "new_state": "BE_LOCKED",
                            "msg": f"🔒 Trade #{trade_id} moved to Breakeven @ ${candidate_sl:.5f} (Target 1.5R reached)"
                        }

                # Start Trailing only after reaching trail_rr_trigger (e.g., 2.0R)
                if trade_state in ["BE_LOCKED", "TRAILING"] and unrealized_profit >= (risk_dist * self.trail_rr_trigger):
                    candidate_sl = close_price - (atr * self.trailing_mult)
                    if candidate_sl > entry_price and candidate_sl > current_sl:
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": candidate_sl,
                            "new_state": "TRAILING",
                            "msg": f"📈 Trailing Stop updated for Trade #{trade_id} to ${candidate_sl:.5f}"
                        }
            else:  # SHORT
                unrealized_profit = entry_price - close_price

                # Lock Breakeven only after reaching be_rr_trigger (e.g., 1.5R)
                if trade_state == "OPEN" and unrealized_profit >= (risk_dist * self.be_rr_trigger):
                    candidate_sl = entry_price - (atr * 0.1)
                    if candidate_sl < entry_price and (current_sl == 0 or candidate_sl < current_sl):
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": candidate_sl,
                            "new_state": "BE_LOCKED",
                            "msg": f"🔒 Trade #{trade_id} moved to Breakeven @ ${candidate_sl:.5f} (Target 1.5R reached)"
                        }

                # Start Trailing only after reaching trail_rr_trigger (e.g., 2.0R)
                if trade_state in ["BE_LOCKED", "TRAILING"] and unrealized_profit >= (risk_dist * self.trail_rr_trigger):
                    candidate_sl = close_price + (atr * self.trailing_mult)
                    if candidate_sl < entry_price and (current_sl == 0 or candidate_sl < current_sl):
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": candidate_sl,
                            "new_state": "TRAILING",
                            "msg": f"📉 Trailing Stop updated for Trade #{trade_id} to ${candidate_sl:.5f}"
                        }

        return {"action": "HOLD"}