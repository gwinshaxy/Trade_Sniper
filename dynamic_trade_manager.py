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
    Handles dynamic active position management with an enable/disable toggle.
    """

    def __init__(
        self,
        trailing_mult: float = 1.8,
        be_rr_trigger: float = 0.85,
        profit_lock_rr_trigger: float = 1.1,
        profit_lock_buffer: float = 0.4,
        trail_rr_trigger: float = 1.65,
        initial_trail_lock_rr: float = 1.2,
        tighten_rr_trigger: float = 1.8,
        tighten_mult: float = 1.2,
        fee_buffer_pct: float = 0.0012,
        enabled: bool = False,  # Solution 1: Default to False to prevent accidental activation
        executor: BybitFuturesLiveExecutor = None,
    ):
        self.trailing_mult = trailing_mult
        self.be_rr_trigger = be_rr_trigger
        self.profit_lock_rr_trigger = profit_lock_rr_trigger
        self.profit_lock_buffer = profit_lock_buffer
        self.trail_rr_trigger = trail_rr_trigger
        self.initial_trail_lock_rr = initial_trail_lock_rr
        self.tighten_rr_trigger = tighten_rr_trigger
        self.tighten_mult = tighten_mult
        self.fee_buffer_pct = fee_buffer_pct
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
        # Solution 2: Absolute hard guard at the top of process_trade
        if not self.enabled:
            return {"action": "HOLD"}

        try:
            if not trade or not latest_candle:
                return {"action": "HOLD"}

            trade_id = trade.get("id")
            direction = str(trade.get("direction") or "").upper()

            entry_price = self._safe_float(trade.get("entry_price"))
            current_sl = self._safe_float(trade.get("stop_loss"))
            current_tp = self._safe_float(trade.get("take_profit"))
            trade_state = trade.get("trade_state") or "OPEN"

            close_price = self._safe_float(latest_candle.get("close"))
            atr = self._safe_float(latest_candle.get("atr"))

            if entry_price <= 0 or close_price <= 0:
                return {"action": "HOLD"}

            is_long = direction in ["BUY", "LONG"]
            from config import BYBIT_TESTNET
            
            # Solution 3: Strict separation of Paper Mode vs. Live/Testnet Execution
            # In live or Bybit testnet execution, rely on exchange-managed stop-losses rather than virtual candle closes.
            is_paper_mode = os.getenv("ENABLE_PAPER_TRADING", "false").lower() == "true"

            # ---- 1. Intra-candle SL/TP Breach Check (Paper Trading Only) ----
            if is_paper_mode and current_sl > 0:
                hit_sl = close_price <= current_sl if is_long else close_price >= current_sl
                hit_tp = close_price >= current_tp if (is_long and current_tp > 0) else (close_price <= current_tp if (not is_long and current_tp > 0) else False)

                if hit_sl or hit_tp:
                    exit_price = current_sl if hit_sl else current_tp
                    position_size = self._safe_float(trade.get("position_size"))
                    account_balance = self._safe_float(trade.get("account_balance"), 100.0)
                    
                    pnl_usd, pnl_pct, outcome = calculate_pnl(
                        direction=direction,
                        entry_price=entry_price,
                        current_price=exit_price,
                        quantity=position_size,
                        account_balance=account_balance,
                        total_fees=(entry_price * position_size * 0.00055) + (exit_price * position_size * 0.00055),
                    )
                    finalize_trade_in_db(
                        trade_id=trade_id,
                        exit_price=exit_price,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        outcome="WIN" if pnl_usd > 0 else "LOSS",
                        fee_usd=(entry_price * position_size * 0.00055) + (exit_price * position_size * 0.00055),
                    )
                    return {
                        "action": "FINALIZED_CLOSED_TRADE",
                        "msg": f"✅ Virtual Trade #{trade_id} closed @ ${exit_price:.5f}"
                    }

            # ---- 2. Dynamic Trailing & Fee-Aware Breakeven Logic ----
            if atr > 0:
                initial_risk_dist = abs(entry_price - current_sl) if current_sl > 0 else (entry_price * 0.02)

                if is_long:
                    unrealized_profit = close_price - entry_price

                    if trade_state == "OPEN" and unrealized_profit >= (initial_risk_dist * self.be_rr_trigger):
                        fee_aware_sl = entry_price * (1.0 + self.fee_buffer_pct)
                        if fee_aware_sl > current_sl:
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": fee_aware_sl,
                                "new_state": "BE_LOCKED",
                                "msg": f"🔒 Trade #{trade_id} moved to Fee-Covered Breakeven @ ${fee_aware_sl:.5f}"
                            }

                    if trade_state in ["BE_LOCKED", "PROFIT_LOCKED", "TRAILING", "OPEN"] and unrealized_profit >= (initial_risk_dist * self.trail_rr_trigger):
                        locked_1r_sl = entry_price + (initial_risk_dist * self.initial_trail_lock_rr)
                        atr_candidate_sl = close_price - (atr * self.trailing_mult)
                        candidate_sl = max(locked_1r_sl, atr_candidate_sl)

                        if candidate_sl > current_sl:
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "new_state": "TRAILING",
                                "msg": f"📈 Scalp Trail activated for #{trade_id} → SL locked @ ${candidate_sl:.5f}"
                            }
                else:  # SHORT
                    unrealized_profit = entry_price - close_price

                    if trade_state == "OPEN" and unrealized_profit >= (initial_risk_dist * self.be_rr_trigger):
                        fee_aware_sl = entry_price * (1.0 - self.fee_buffer_pct)
                        if current_sl == 0 or fee_aware_sl < current_sl:
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": fee_aware_sl,
                                "new_state": "BE_LOCKED",
                                "msg": f"🔒 Trade #{trade_id} moved to Fee-Covered Breakeven @ ${fee_aware_sl:.5f}"
                            }

                    if trade_state in ["BE_LOCKED", "PROFIT_LOCKED", "TRAILING", "OPEN"] and unrealized_profit >= (initial_risk_dist * self.trail_rr_trigger):
                        locked_1r_sl = entry_price - (initial_risk_dist * self.initial_trail_lock_rr)
                        atr_candidate_sl = close_price + (atr * self.trailing_mult)
                        candidate_sl = min(locked_1r_sl, atr_candidate_sl)

                        if current_sl == 0 or candidate_sl < current_sl:
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "new_state": "TRAILING",
                                "msg": f"📉 Scalp Trail activated for #{trade_id} → SL locked @ ${candidate_sl:.5f}"
                            }

            return {"action": "HOLD"}
        except Exception as err:
            logger.error(f"Error in DynamicTradeManager.process_trade: {err}")
            return {"action": "HOLD"}