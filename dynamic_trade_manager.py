import logging
import os
from typing import Dict, Any
from common import (
    get_db_connection,
    release_db_connection,
    finalize_trade_in_db,
    calculate_pnl,
)
from live_executor import BybitFuturesLiveExecutor

logger = logging.getLogger("dynamic_trade_manager")


class DynamicTradeManager:
    """
    Handles dynamic active position management synced with Chandelier trailing stops (+1.5R activation).
    Pushes real-time Stop Loss updates directly to Bybit UTA via exchange API.
    """

    def __init__(
        self,
        trailing_mult: float = 3.0,
        trail_activation_r: float = 1.5,
        fee_buffer_pct: float = 0.0012,
        disable_static_tp: bool = True,
        enabled: bool = True,
        executor: BybitFuturesLiveExecutor = None,
    ):
        self.trailing_mult = trailing_mult
        self.trail_activation_r = trail_activation_r
        self.fee_buffer_pct = fee_buffer_pct
        self.disable_static_tp = disable_static_tp
        self.enabled = enabled
        self.executor = executor

        if not self.enabled:
            logger.info("⚠️ DynamicTradeManager is currently DISABLED via configuration.")

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def process_trade(
        self,
        trade: Dict[str, Any],
        latest_candle: Dict[str, Any],
        live_pos_info: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"action": "HOLD"}

        try:
            if not trade or not latest_candle:
                return {"action": "HOLD"}

            trade_id = trade.get("id")
            pair = trade.get("pair")
            direction = str(trade.get("direction") or "").upper()

            entry_price = self._safe_float(trade.get("entry_price"))
            current_sl = self._safe_float(trade.get("stop_loss"))
            initial_sl = self._safe_float(trade.get("initial_stop_loss"), current_sl)
            current_tp = self._safe_float(trade.get("take_profit"))
            trade_state = trade.get("trade_state") or "OPEN"

            atr_trail_mult = self._safe_float(trade.get("trailing_atr_mult"), self.trailing_mult)
            activation_r = self._safe_float(trade.get("trail_activation_r"), self.trail_activation_r)

            high_p = self._safe_float(latest_candle.get("high"))
            low_p = self._safe_float(latest_candle.get("low"))
            close_p = self._safe_float(latest_candle.get("close"))
            atr = self._safe_float(latest_candle.get("atr"))

            if high_p <= 0: high_p = close_p
            if low_p <= 0: low_p = close_p

            if entry_price <= 0 or close_p <= 0:
                return {"action": "HOLD"}

            is_long = direction in ["BUY", "LONG"]
            from config import BYBIT_TESTNET
            is_paper_mode = os.getenv("ENABLE_PAPER_TRADING", "false").lower() == "true" or BYBIT_TESTNET

            # ---- Chandelier ATR Trailing Stop Management ----
            if atr > 0:
                initial_risk_dist = abs(entry_price - initial_sl) if initial_sl > 0 else (entry_price * 0.02)

                if is_long:
                    prev_highest = self._safe_float(trade.get("highest_price"), entry_price)
                    highest_price = max(prev_highest, high_p, close_p)
                    unrealized_dist = highest_price - entry_price

                    if unrealized_dist >= (initial_risk_dist * activation_r):
                        fee_aware_be = entry_price * (1.0 + self.fee_buffer_pct)
                        atr_trail_sl = highest_price - (atr * atr_trail_mult)
                        candidate_sl = max(fee_aware_be, atr_trail_sl)

                        if candidate_sl > current_sl:
                            # Update SL on Bybit via exchange API if executor is provided
                            if self.executor and not is_paper_mode:
                                sl_success = self.executor.set_position_trading_stop(
                                    symbol=pair,
                                    stop_loss=candidate_sl,
                                    direction=direction
                                )
                                if sl_success:
                                    logger.info(f"[{pair}] Chandelier Stop Loss updated on Bybit UTA to ${candidate_sl:.5f}")

                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "highest_price": highest_price,
                                "new_state": "TRAILING",
                                "msg": f"📈 Chandelier Trail active for #{trade_id} -> SL @ ${candidate_sl:.5f}",
                            }
                else:  # SHORT
                    prev_lowest = self._safe_float(trade.get("lowest_price"), entry_price)
                    lowest_price = min(prev_lowest, low_p, close_p) if prev_lowest > 0 else min(low_p, close_p)
                    unrealized_dist = entry_price - lowest_price

                    if unrealized_dist >= (initial_risk_dist * activation_r):
                        fee_aware_be = entry_price * (1.0 - self.fee_buffer_pct)
                        atr_trail_sl = lowest_price + (atr * atr_trail_mult)
                        candidate_sl = min(fee_aware_be, atr_trail_sl)

                        if current_sl == 0 or candidate_sl < current_sl:
                            if self.executor and not is_paper_mode:
                                sl_success = self.executor.set_position_trading_stop(
                                    symbol=pair,
                                    stop_loss=candidate_sl,
                                    direction=direction
                                )
                                if sl_success:
                                    logger.info(f"[{pair}] Chandelier Stop Loss updated on Bybit UTA to ${candidate_sl:.5f}")

                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "lowest_price": lowest_price,
                                "new_state": "TRAILING",
                                "msg": f"📉 Chandelier Trail active for #{trade_id} -> SL @ ${candidate_sl:.5f}",
                            }

            return {"action": "HOLD"}
        except Exception as err:
            logger.error(f"Error in DynamicTradeManager.process_trade: {err}")
            return {"action": "HOLD"}