import logging
from typing import Dict, Any
from common import (
    get_db_connection,
    release_db_connection,
    finalize_trade_in_db,
    calculate_pnl,
)
from live_executor import BybitFuturesLiveExecutor, format_ccxt_futures_symbol

logger = logging.getLogger("dynamic_trade_manager")


class DynamicTradeManager:
    """
    Handles dynamic active position management featuring:
    - Breakeven lock (~0.6R)
    - 1.0R Profit-Securing Lock (~1.0R)
    - ATR Trailing Stop & Tightening Model (~1.4R+)
    """

    def __init__(
        self,
        trailing_mult: float = 1.8,
        be_rr_trigger: float = 0.6,
        profit_lock_rr_trigger: float = 1.0,  # Secures profit once 1.0R gain is reached
        profit_lock_buffer: float = 0.3,      # Locks in +0.3R guaranteed profit
        trail_rr_trigger: float = 1.4,
        tighten_rr_trigger: float = 1.8,
        tighten_mult: float = 1.2,
        be_buffer_atr: float = 0.1,
        hard_tp_enabled: bool = True,
        allow_partial_tp: bool = False,
        partial_tp_fraction: float = 0.5,
        executor: BybitFuturesLiveExecutor = None,
    ):
        self.trailing_mult = trailing_mult
        self.be_rr_trigger = be_rr_trigger
        self.profit_lock_rr_trigger = profit_lock_rr_trigger
        self.profit_lock_buffer = profit_lock_buffer
        self.trail_rr_trigger = trail_rr_trigger
        self.tighten_rr_trigger = tighten_rr_trigger
        self.tighten_mult = tighten_mult
        self.be_buffer_atr = be_buffer_atr
        self.hard_tp_enabled = hard_tp_enabled
        self.allow_partial_tp = allow_partial_tp
        self.partial_tp_fraction = partial_tp_fraction
        self.executor = executor

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
                    logger.info(
                        f"Persisted calculated SL (${stop_loss:.4f}) and TP (${take_profit:.4f}) "
                        f"to DB for Trade #{trade_id}"
                    )
            except Exception as e:
                logger.error(f"Failed to update DB SL/TP for trade #{trade_id}: {e}")
            finally:
                release_db_connection(conn)

    def _finalize_closed_trade(self, trade: Dict[str, Any], live_pos_info: Dict[str, Any]) -> bool:
        trade_id = trade.get("id")
        pair = trade.get("pair")
        direction = trade.get("direction")
        entry_price = self._safe_float(trade.get("entry_price"))
        position_size = self._safe_float(trade.get("position_size"))
        account_balance = self._safe_float(trade.get("account_balance"), 100.0)

        if not trade_id or not pair:
            logger.error(f"Cannot finalize closed trade: missing id/pair. Trade={trade}")
            return False

        real_closed_pnl = None
        real_fee = 0.0
        verified_exit = 0.0
        if self.executor is not None:
            try:
                real_closed_pnl, real_fee, verified_exit = self.executor.fetch_real_closed_pnl(pair)
            except Exception as e:
                logger.warning(f"[{pair}] closedPnl lookup failed during finalization: {e}")

        if verified_exit > 0:
            exit_price = verified_exit
        else:
            exit_price = self._safe_float(live_pos_info.get("mark_price"), 0.0) if live_pos_info else 0.0
            if exit_price <= 0:
                exit_price = entry_price

        if real_closed_pnl is not None:
            pnl_usd, pnl_pct, outcome = calculate_pnl(
                direction=direction,
                entry_price=entry_price,
                current_price=exit_price,
                quantity=position_size,
                account_balance=account_balance,
                total_fees=real_fee,
                exchange_closed_pnl=real_closed_pnl,
            )
        else:
            est_fee = (entry_price * position_size * 0.00055) + (exit_price * position_size * 0.00055)
            pnl_usd, pnl_pct, outcome = calculate_pnl(
                direction=direction,
                entry_price=entry_price,
                current_price=exit_price,
                quantity=position_size,
                account_balance=account_balance,
                total_fees=est_fee,
            )
            real_fee = est_fee
            if exit_price == entry_price:
                outcome = "UNKNOWN"

        try:
            finalize_trade_in_db(
                trade_id=trade_id,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                outcome=outcome,
                fee_usd=real_fee,
            )
            logger.info(
                f"[{pair}] Trade #{trade_id} FINALIZED via DynamicTradeManager "
                f"(exit=${exit_price:.5f}, PnL=${pnl_usd:.2f}, outcome={outcome})."
            )
            return True
        except Exception as e:
            logger.error(f"[{pair}] Failed to finalize trade #{trade_id}: {e}")
            return False

    def process_trade(
        self,
        trade: Dict[str, Any],
        latest_candle: Dict[str, Any],
        live_pos_info: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
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
            high_price = self._safe_float(latest_candle.get("high"), close_price)
            low_price = self._safe_float(latest_candle.get("low"), close_price)
            atr = self._safe_float(latest_candle.get("atr"))

            if entry_price <= 0 or close_price <= 0:
                return {"action": "HOLD"}

            is_long = direction in ["BUY", "LONG"]
            db_needs_update = False

            # ---- Sanity: SL direction guard -------------------------------
            if is_long:
                if current_sl >= entry_price and trade_state == "OPEN":
                    current_sl = entry_price * 0.98
                    db_needs_update = True
            elif not is_long:
                if current_sl <= entry_price and current_sl > 0 and trade_state == "OPEN":
                    current_sl = entry_price * 1.02
                    db_needs_update = True

            # ---- Sanity: TP direction guard -------------------------------
            if is_long and current_tp > 0 and current_tp <= entry_price:
                current_tp = entry_price + (abs(entry_price - current_sl) * 1.8)
                db_needs_update = True
            elif (not is_long) and current_tp > 0 and current_tp >= entry_price:
                current_tp = entry_price - (abs(entry_price - current_sl) * 1.8)
                db_needs_update = True

            if db_needs_update and trade_id is not None:
                self._update_db_sl_tp(trade_id, current_sl, current_tp)

            # ---- 1. Intra-candle SL/TP Inspection (Paper Mode / Testnet) ----
            import os
            from config import BYBIT_TESTNET
            
            is_paper_mode = os.getenv("ENABLE_PAPER_TRADING", "false").lower() == "true" or BYBIT_TESTNET
            
            if current_sl > 0 or current_tp > 0:
                if is_long:
                    hit_sl = low_price <= current_sl if current_sl > 0 else False
                    hit_tp = high_price >= current_tp if current_tp > 0 else False
                else:
                    hit_sl = high_price >= current_sl if current_sl > 0 else False
                    hit_tp = low_price <= current_tp if current_tp > 0 else False

                if hit_sl or hit_tp:
                    outcome_reason = "STOP_LOSS" if hit_sl and not hit_tp else ("TAKE_PROFIT" if hit_tp and not hit_sl else "SL_TP_BREACH")
                    logger.warning(
                        f"[{trade.get('pair')}] Trade #{trade_id} intra-candle price breach detected "
                        f"(Low: ${low_price:.5f}, High: ${high_price:.5f}) hitting {outcome_reason}."
                    )
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
                        "msg": f"✅ Virtual Trade #{trade_id} closed via intra-candle {outcome_reason} @ ${exit_price:.5f}"
                    }

            # ---- 2. Progressive Breakeven, Profit Locking & Trailing --------
            if atr > 0:
                risk_dist = abs(entry_price - current_sl) if current_sl > 0 else (entry_price * 0.02)

                if is_long:
                    unrealized_profit = close_price - entry_price

                    # 1. Breakeven Trigger (~0.6R)
                    if trade_state == "OPEN" and unrealized_profit >= (risk_dist * self.be_rr_trigger):
                        candidate_sl = entry_price + (atr * self.be_buffer_atr)
                        if candidate_sl > entry_price and candidate_sl > current_sl:
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "new_state": "BE_LOCKED",
                                "msg": f"🔒 Trade #{trade_id} moved to Breakeven @ ${candidate_sl:.5f}"
                            }

                    # 2. 1.0R Profit-Securing Lock Trigger (~1.0R gain)
                    if trade_state in ["OPEN", "BE_LOCKED"] and unrealized_profit >= (risk_dist * self.profit_lock_rr_trigger):
                        candidate_sl = entry_price + (risk_dist * self.profit_lock_buffer)
                        if candidate_sl > current_sl:
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "new_state": "PROFIT_LOCKED",
                                "msg": f"💰 Trade #{trade_id} crossed 1.0R gain! Locked in profit @ ${candidate_sl:.5f}"
                            }

                    # 3. Trailing Stop Trigger (~1.4R)
                    if trade_state in ["BE_LOCKED", "PROFIT_LOCKED", "TRAILING"] and unrealized_profit >= (risk_dist * self.trail_rr_trigger):
                        candidate_sl = close_price - (atr * self.trailing_mult)
                        if candidate_sl > entry_price and candidate_sl > current_sl:
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "new_state": "TRAILING",
                                "msg": f"📈 Scalp Trail activated for #{trade_id} → SL @ ${candidate_sl:.5f}"
                            }

                    # 4. Tightening Trigger (~1.8R)
                    if trade_state == "TRAILING" and unrealized_profit >= (risk_dist * self.tighten_rr_trigger):
                        candidate_sl = close_price - (atr * self.tighten_mult)
                        if candidate_sl > entry_price and candidate_sl > current_sl:
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "new_state": "TRAILING",
                                "msg": f"📈 Tightened scalp trail for #{trade_id} → SL @ ${candidate_sl:.5f}"
                            }
                else:  # SHORT
                    unrealized_profit = entry_price - close_price

                    # 1. Breakeven Trigger
                    if trade_state == "OPEN" and unrealized_profit >= (risk_dist * self.be_rr_trigger):
                        candidate_sl = entry_price - (atr * self.be_buffer_atr)
                        if candidate_sl < entry_price and (current_sl == 0 or candidate_sl < current_sl):
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "new_state": "BE_LOCKED",
                                "msg": f"🔒 Trade #{trade_id} moved to Breakeven @ ${candidate_sl:.5f}"
                            }

                    # 2. 1.0R Profit-Securing Lock Trigger (~1.0R gain)
                    if trade_state in ["OPEN", "BE_LOCKED"] and unrealized_profit >= (risk_dist * self.profit_lock_rr_trigger):
                        candidate_sl = entry_price - (risk_dist * self.profit_lock_buffer)
                        if current_sl == 0 or candidate_sl < current_sl:
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "new_state": "PROFIT_LOCKED",
                                "msg": f"💰 Trade #{trade_id} crossed 1.0R gain! Locked in profit @ ${candidate_sl:.5f}"
                            }

                    # 3. Trailing Stop Trigger
                    if trade_state in ["BE_LOCKED", "PROFIT_LOCKED", "TRAILING"] and unrealized_profit >= (risk_dist * self.trail_rr_trigger):
                        candidate_sl = close_price + (atr * self.trailing_mult)
                        if candidate_sl < entry_price and (current_sl == 0 or candidate_sl < current_sl):
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "new_state": "TRAILING",
                                "msg": f"📉 Scalp Trail activated for #{trade_id} → SL @ ${candidate_sl:.5f}"
                            }

                    # 4. Tightening Trigger
                    if trade_state == "TRAILING" and unrealized_profit >= (risk_dist * self.tighten_rr_trigger):
                        candidate_sl = close_price + (atr * self.tighten_mult)
                        if candidate_sl < entry_price and (current_sl == 0 or candidate_sl < current_sl):
                            return {
                                "action": "UPDATE_SL",
                                "new_sl": candidate_sl,
                                "new_state": "TRAILING",
                                "msg": f"📉 Tightened scalp trail for #{trade_id} → SL @ ${candidate_sl:.5f}"
                            }

            return {"action": "HOLD"}
        except Exception as err:
            logger.error(f"Error in DynamicTradeManager.process_trade: {err}")
            return {"action": "HOLD"}