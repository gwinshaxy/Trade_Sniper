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
    Handles dynamic active position management with a progressive
    breakeven + trailing stop model.

    MODEL:
      - Before BE trigger (default 1.5R): SL stays at initial level
      - At BE trigger: SL moves to entry + small buffer
      - At trail trigger (default 2.0R): SL begins trailing behind price (wide, 3.0× ATR)
      - At tighten trigger (default 3.5R): SL trails tighter (1.8× ATR)
      - Hard TP only used as a SAFETY NET for catastrophic exhaustion; the
        primary exit is the trailing stop.

    FIX #P0 (NEW): When the exchange reports zero contracts for a tracked trade,
    this class now FINALIZES the trade directly (querying Bybit's closedPnl
    endpoint for the authoritative exit price and net PnL) rather than
    deferring to the reconciler. The reconciler is only triggered by private
    WS execution events, which are unreliable on testnet (see log analysis),
    so deferring caused permanent zombie DB records.
    """

    def __init__(
        self,
        trailing_mult: float = 3.0,
        be_rr_trigger: float = 1.5,
        trail_rr_trigger: float = 2.0,
        tighten_rr_trigger: float = 3.5,
        tighten_mult: float = 1.8,
        be_buffer_atr: float = 0.1,
        hard_tp_enabled: bool = True,
        allow_partial_tp: bool = False,
        partial_tp_fraction: float = 0.5,
        executor: BybitFuturesLiveExecutor = None,
    ):
        self.trailing_mult = trailing_mult
        self.be_rr_trigger = be_rr_trigger
        self.trail_rr_trigger = trail_rr_trigger
        self.tighten_rr_trigger = tighten_rr_trigger
        self.tighten_mult = tighten_mult
        self.be_buffer_atr = be_buffer_atr
        self.hard_tp_enabled = hard_tp_enabled
        self.allow_partial_tp = allow_partial_tp
        self.partial_tp_fraction = partial_tp_fraction
        # FIX #P0: reference to the live executor so we can query closedPnl
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

    # ------------------------------------------------------------------
    # FIX #P0: Self-contained finalization for exchange-detected closes.
    # ------------------------------------------------------------------
    def _finalize_closed_trade(self, trade: Dict[str, Any], live_pos_info: Dict[str, Any]) -> bool:
        """
        Queries Bybit for the authoritative closedPnl, computes net PnL, and
        finalizes the DB record. Returns True if finalization succeeded.
        """
        trade_id = trade.get("id")
        pair = trade.get("pair")
        direction = trade.get("direction")
        entry_price = self._safe_float(trade.get("entry_price"))
        position_size = self._safe_float(trade.get("position_size"))
        account_balance = self._safe_float(trade.get("account_balance"), 100.0)

        if not trade_id or not pair:
            logger.error(f"Cannot finalize closed trade: missing id/pair. Trade={trade}")
            return False

        # Query Bybit for the real closed PnL and exit price
        real_closed_pnl = None
        real_fee = 0.0
        verified_exit = 0.0
        if self.executor is not None:
            try:
                real_closed_pnl, real_fee, verified_exit = self.executor.fetch_real_closed_pnl(pair)
            except Exception as e:
                logger.warning(f"[{pair}] closedPnl lookup failed during finalization: {e}")

        # Determine exit price: prefer exchange-verified, else live mark
        if verified_exit > 0:
            exit_price = verified_exit
        else:
            exit_price = self._safe_float(live_pos_info.get("mark_price"), 0.0) if live_pos_info else 0.0
            if exit_price <= 0:
                # Last resort: use the last known live position entry (will yield ~0 PnL)
                exit_price = entry_price
            logger.warning(
                f"[{pair}] Trade #{trade_id} closed but no verified exit price. "
                f"Using ${exit_price:.5f} for finalization."
            )

        # Compute PnL
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
            logger.error(
                f"🚨 CORRUPT TP for #{trade_id}: LONG TP (${current_tp}) <= entry (${entry_price})."
            )
            current_tp = entry_price + (abs(entry_price - current_sl) * 3.0)
            db_needs_update = True
        elif (not is_long) and current_tp > 0 and current_tp >= entry_price:
            logger.error(
                f"🚨 CORRUPT TP for #{trade_id}: SHORT TP (${current_tp}) >= entry (${entry_price})."
            )
            current_tp = entry_price - (abs(entry_price - current_sl) * 3.0)
            db_needs_update = True

        if current_sl == 0.0 and entry_price > 0:
            current_sl = entry_price * (1.0 - 0.02) if is_long else entry_price * (1.0 + 0.02)
            db_needs_update = True

        if current_tp == 0.0 and entry_price > 0:
            risk = abs(entry_price - current_sl)
            current_tp = entry_price + (risk * 4.0) if is_long else entry_price - (risk * 4.0)
            db_needs_update = True

        if db_needs_update and trade_id is not None:
            self._update_db_sl_tp(trade_id, current_sl, current_tp)

        # ---- 1. Exchange closure check --------------------------------
        # FIX #P0: Finalize DIRECTLY instead of deferring to reconciler.
        # The reconciler is triggered by private WS execution events, which
        # drop frequently on testnet (see 02:08–03:14 log evidence). Without
        # this fix, closed trades remain in 'OPEN' state indefinitely and
        # block future entries for the same symbol.
        if live_pos_info and not live_pos_info.get("error") and live_pos_info.get("contracts", 0.0) <= 0.001:
            logger.warning(
                f"[{trade.get('pair')}] Trade #{trade_id} closed on exchange. "
                f"Finalizing directly via DynamicTradeManager."
            )
            finalized = self._finalize_closed_trade(trade, live_pos_info)
            return {
                "action": "FINALIZED_CLOSED_TRADE" if finalized else "SYNC_CLOSED_FROM_EXCHANGE",
                "msg": (
                    f"✅ Trade #{trade_id} finalized (dynamic manager)."
                    if finalized
                    else f"⚠️ Trade #{trade_id} closed but finalization failed; reconciler will retry."
                ),
            }

        # ---- 2. Hard TP/SL checks -------------------------------------
        if is_long:
            if self.hard_tp_enabled and current_tp > 0 and trade_state not in ["TRAILING"] and high_price >= current_tp:
                return {
                    "action": "EXECUTE_CLOSE_TP",
                    "target_price": current_tp,
                    "msg": f"🎯 Hard TP hit for #{trade_id}. Executing market close on Bybit..."
                }
            if current_sl > 0 and low_price <= current_sl:
                return {
                    "action": "EXECUTE_CLOSE_SL",
                    "target_price": current_sl,
                    "msg": f"🛑 Target SL hit for #{trade_id}. Executing market close on Bybit..."
                }
        else:
            if self.hard_tp_enabled and current_tp > 0 and trade_state not in ["TRAILING"] and low_price <= current_tp:
                return {
                    "action": "EXECUTE_CLOSE_TP",
                    "target_price": current_tp,
                    "msg": f"🎯 Hard TP hit for #{trade_id}. Executing market close on Bybit..."
                }
            if current_sl > 0 and high_price >= current_sl:
                return {
                    "action": "EXECUTE_CLOSE_SL",
                    "target_price": current_sl,
                    "msg": f"🛑 Target SL hit for #{trade_id}. Executing market close on Bybit..."
                }

        # ---- 3. Progressive breakeven + trailing ---------------------
        if atr > 0:
            risk_dist = abs(entry_price - current_sl) if current_sl > 0 else (entry_price * 0.02)

            if is_long:
                unrealized_profit = close_price - entry_price

                # 3A: Move to Breakeven once +1.5R reached
                if trade_state == "OPEN" and unrealized_profit >= (risk_dist * self.be_rr_trigger):
                    candidate_sl = entry_price + (atr * self.be_buffer_atr)
                    if candidate_sl > entry_price and candidate_sl > current_sl:
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": candidate_sl,
                            "new_state": "BE_LOCKED",
                            "msg": f"🔒 Trade #{trade_id} moved to Breakeven @ ${candidate_sl:.5f}"
                        }

                # 3B: Start trailing once +2.0R reached
                if trade_state in ["BE_LOCKED", "TRAILING"] and unrealized_profit >= (risk_dist * self.trail_rr_trigger):
                    candidate_sl = close_price - (atr * self.trailing_mult)
                    if candidate_sl > entry_price and candidate_sl > current_sl:
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": candidate_sl,
                            "new_state": "TRAILING",
                            "msg": f"📈 Trail activated for #{trade_id} → SL @ ${candidate_sl:.5f}"
                        }

                # 3C: Tighten trail once +3.5R reached
                if trade_state == "TRAILING" and unrealized_profit >= (risk_dist * self.tighten_rr_trigger):
                    candidate_sl = close_price - (atr * self.tighten_mult)
                    if candidate_sl > entry_price and candidate_sl > current_sl:
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": candidate_sl,
                            "new_state": "TRAILING",
                            "msg": f"📈 Tightened trail for #{trade_id} → SL @ ${candidate_sl:.5f}"
                        }
            else:  # SHORT
                unrealized_profit = entry_price - close_price

                # 3A: Breakeven
                if trade_state == "OPEN" and unrealized_profit >= (risk_dist * self.be_rr_trigger):
                    candidate_sl = entry_price - (atr * self.be_buffer_atr)
                    if candidate_sl < entry_price and (current_sl == 0 or candidate_sl < current_sl):
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": candidate_sl,
                            "new_state": "BE_LOCKED",
                            "msg": f"🔒 Trade #{trade_id} moved to Breakeven @ ${candidate_sl:.5f}"
                        }

                # 3B: Start trailing
                if trade_state in ["BE_LOCKED", "TRAILING"] and unrealized_profit >= (risk_dist * self.trail_rr_trigger):
                    candidate_sl = close_price + (atr * self.trailing_mult)
                    if candidate_sl < entry_price and (current_sl == 0 or candidate_sl < current_sl):
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": candidate_sl,
                            "new_state": "TRAILING",
                            "msg": f"📉 Trail activated for #{trade_id} → SL @ ${candidate_sl:.5f}"
                        }

                # 3C: Tighten trail
                if trade_state == "TRAILING" and unrealized_profit >= (risk_dist * self.tighten_rr_trigger):
                    candidate_sl = close_price + (atr * self.tighten_mult)
                    if candidate_sl < entry_price and (current_sl == 0 or candidate_sl < current_sl):
                        return {
                            "action": "UPDATE_SL",
                            "new_sl": candidate_sl,
                            "new_state": "TRAILING",
                            "msg": f"📉 Tightened trail for #{trade_id} → SL @ ${candidate_sl:.5f}"
                        }

        return {"action": "HOLD"}