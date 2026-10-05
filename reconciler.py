import os
import sys
import time
import logging
import asyncio
import threading
from typing import Dict, Any, List, Set, Optional, Tuple

from common import (
    get_db_connection,
    release_db_connection,
    finalize_trade_in_db,
    calculate_pnl,
    send_telegram_notification,
    check_asset_cooldown,
    set_asset_cooldown,
    normalize_symbol,
    logger
)
from config import STRATEGY_CONFIG, BYBIT_TESTNET
from live_executor import BybitFuturesLiveExecutor, safe_float, format_ccxt_futures_symbol
from event_bus import event_bus

# Operational thresholds & parameters
DUST_THRESHOLD = 0.001
RETRY_THRESHOLD = 3
GRACE_PERIOD_SECONDS = 120.0  # Time allowed for API & order engines to settle
GHOST_MIN_SUSTAINED_ZERO_SECONDS = 900  # 15 minutes before confirming ghost state

# Thread safety locks & tracking caches
reconciler_lock = threading.Lock()
consecutive_zero_counts: Dict[int, int] = {}
first_zero_at: Dict[int, float] = {}

RECENTLY_OPENED_CACHE: Dict[str, float] = {}
ENTRY_ORDER_CACHE: Dict[int, Tuple[str, float]] = {}


def normalize_pair(pair_str: str) -> str:
    """Standardizes pairs (e.g., 'SOL/USDT:USDT' -> 'SOLUSDT') for reliable comparison."""
    if not pair_str:
        return ""
    return (
        pair_str.upper()
        .replace("/", "")
        .replace(":", "")
        .replace("-", "")
        .replace("_", "")
        .replace("USDTUSDT", "USDT")
    )


def mark_symbol_recently_opened(ccxt_symbol: str) -> None:
    """Registers a symbol in the grace period cache when a trade opens."""
    formatted = format_ccxt_futures_symbol(ccxt_symbol)
    with reconciler_lock:
        RECENTLY_OPENED_CACHE[formatted] = time.time()
    logger.debug(f"[{formatted}] Added to RECENTLY_OPENED_CACHE.")


def is_symbol_under_grace_period(ccxt_symbol: str) -> bool:
    """Checks if a symbol is within its post-execution grace period."""
    formatted = format_ccxt_futures_symbol(ccxt_symbol)
    with reconciler_lock:
        if formatted in RECENTLY_OPENED_CACHE:
            elapsed = time.time() - RECENTLY_OPENED_CACHE[formatted]
            if elapsed < GRACE_PERIOD_SECONDS:
                return True
            del RECENTLY_OPENED_CACHE[formatted]
    return False


def mark_trade_entry_order(trade_id: int, entry_order_id: str) -> None:
    """Registers the entry order ID for a trade to query status before declaring a ghost."""
    if trade_id and entry_order_id:
        with reconciler_lock:
            ENTRY_ORDER_CACHE[trade_id] = (str(entry_order_id), time.time())
        logger.debug(f"[Trade #{trade_id}] Registered entry_order_id={entry_order_id}.")


def check_active_or_pending_orders(executor: BybitFuturesLiveExecutor, ccxt_symbol: str) -> bool:
    """
    Filters out protective (SL/TP) orders.
    Only active entry orders suppress zero-contract counters.
    """
    try:
        open_orders = executor.exchange.fetch_open_orders(ccxt_symbol)
        for oo in open_orders:
            info = oo.get("info", {}) or {}
            stop_order_type = str(info.get("stopOrderType", "") or "").strip()
            if stop_order_type in ("StopLoss", "TakeProfit", "TrailingStop", "Stop"):
                continue
            return True

        since = int((time.time() - 300) * 1000)
        recent_closed = executor.exchange.fetch_closed_orders(ccxt_symbol, since=since, limit=10)
        for order in recent_closed:
            status = str(order.get("status", "")).lower()
            if status not in ["open", "untriggered", "new"]:
                continue
            info = order.get("info", {}) or {}
            stop_order_type = str(info.get("stopOrderType", "") or "").strip()
            if stop_order_type in ("StopLoss", "TakeProfit", "TrailingStop", "Stop"):
                continue
            return True
    except Exception as e:
        logger.debug(f"[{ccxt_symbol}] Exception checking pending orders: {e}")
        return False
    return False


def check_entry_order_never_filled(
    executor: BybitFuturesLiveExecutor,
    trade_id: int,
    ccxt_symbol: str,
) -> Optional[str]:
    """
    If the entry order was cancelled/rejected/expired without filling,
    returns the terminal status so the trade is marked CANCELLED instead of ghost-closed.
    """
    with reconciler_lock:
        cache_entry = ENTRY_ORDER_CACHE.get(trade_id)

    if not cache_entry:
        return None

    entry_order_id = cache_entry[0] if isinstance(cache_entry, tuple) else cache_entry

    try:
        order = executor.exchange.fetch_order(entry_order_id, ccxt_symbol)
        status = str(order.get("status", "")).lower()
        filled = safe_float(order.get("filled", 0))

        if status in ("canceled", "rejected", "expired") and filled < DUST_THRESHOLD:
            logger.warning(
                f"[Trade #{trade_id}] Entry order {entry_order_id} never filled "
                f"(status={status}, filled={filled}). Marking as CANCELLED."
            )
            return status.upper()
    except Exception as e:
        logger.debug(f"[Trade #{trade_id}] Entry order lookup failed: {e}")
    return None


def check_unprocessed_execution_close(
    executor: BybitFuturesLiveExecutor,
    symbol: str,
) -> bool:
    """
    Queries /v5/execution/list for recent close fills.
    If Bybit recorded a closing trade not yet processed in DB, prevents zero-contract counting.
    """
    try:
        execs = executor.fetch_recent_executions(symbol, lookback_ms=300_000)
        for e in execs:
            if str(e.get("execType", "")) != "Trade":
                continue
            closed_size = safe_float(e.get("closedSize", 0))
            if closed_size > DUST_THRESHOLD:
                logger.info(
                    f"[{symbol}] Unprocessed close fill detected in execution/list "
                    f"(closedSize={closed_size}, price={e.get('execPrice')}). "
                    f"Suppressing ghost counter."
                )
                return True
    except Exception as e:
        logger.debug(f"[{symbol}] execution/list check failed: {e}")
    return False


def fetch_all_live_exchange_positions(executor: BybitFuturesLiveExecutor) -> Dict[str, dict]:
    """
    Fetches active positions directly from Bybit.
    Filters out stale records (>60s old) and reads stopLoss/takeProfit directly from raw payloads.
    """
    active_positions = {}
    now_ms = int(time.time() * 1000)
    STALE_MS = 60_000

    try:
        positions = executor.exchange.fetch_positions()
        for p in positions:
            contracts = safe_float(p.get("contracts", 0))
            if contracts <= DUST_THRESHOLD:
                continue

            info = p.get("info", {}) or {}

            updated_ms = int(info.get("updatedTime", 0) or 0)
            if updated_ms > 0 and (now_ms - updated_ms) > STALE_MS:
                logger.debug(
                    f"[{p.get('symbol')}] Skipping stale position "
                    f"(updated {int((now_ms - updated_ms) / 1000)}s ago)."
                )
                continue

            raw_symbol = p.get("symbol", "")
            ccxt_symbol = format_ccxt_futures_symbol(raw_symbol)
            side = str(p.get("side", "")).lower()
            if not side or side == "none":
                side = "long" if safe_float(p.get("side", 0)) > 0 else "short"

            sl_raw = p.get("stopLoss")
            tp_raw = p.get("takeProfit")
            sl_val = safe_float(sl_raw) if sl_raw not in (None, "") else safe_float(info.get("stopLoss", 0))
            tp_val = safe_float(tp_raw) if tp_raw not in (None, "") else safe_float(info.get("takeProfit", 0))

            active_positions[ccxt_symbol] = {
                "symbol": ccxt_symbol,
                "raw_symbol": raw_symbol,
                "normalized_symbol": normalize_pair(raw_symbol),
                "side": side,
                "contracts": contracts,
                "entry_price": safe_float(p.get("entryPrice", 0)),
                "stop_loss": sl_val,
                "take_profit": tp_val,
                "unrealized_pnl": safe_float(p.get("unrealizedPnl", 0)),
                "leverage": safe_float(p.get("leverage", 1)),
                "position_idx": int(info.get("positionIdx", 0)),
            }
    except Exception as e:
        logger.error(f"[Reconciler] Error fetching bulk positions: {e}. Preserving DB state.")
        return {}

    return active_positions


class PositionReconciler:
    """
    Robust State Reconciliation Engine.
    Synchronizes local database trade records with live exchange position state.
    """

    def __init__(self, executor: Optional[BybitFuturesLiveExecutor] = None):
        self.executor = executor or BybitFuturesLiveExecutor()

    def fetch_open_db_trades(self) -> List[Dict[str, Any]]:
        """Retrieves active/open trade records from the database using row locking."""
        conn = get_db_connection()
        if not conn:
            logger.error("[Reconciler] Unable to establish DB connection for active trade query.")
            return []

        trades = []
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, pair, direction, entry_price, stop_loss, take_profit, 
                           position_size, account_balance, trade_state, status, created_at
                    FROM trade_setups
                    WHERE status IN ('EXECUTED', 'PENDING') AND trade_state IN ('OPEN', 'EXECUTED', 'BE_LOCKED', 'TRAILING')
                    FOR UPDATE SKIP LOCKED;
                """)
                columns = [desc[0] for desc in cur.description]
                for row in cur.fetchall():
                    trades.append(dict(zip(columns, row)))
        except Exception as e:
            logger.error(f"[Reconciler] Error querying open DB trades: {e}")
        finally:
            release_db_connection(conn)

        return trades

    def run_reconciliation_cycle(self) -> int:
        """Executes a complete reconciliation check across all database trades and exchange positions."""
        global consecutive_zero_counts, first_zero_at
        logger.info("🔍 Running Database <-> Bybit Futures Position Reconciliation & Auto-Healing...")

        db_trades = self.fetch_open_db_trades()
        if not db_trades:
            logger.info("[Reconciler] No active DB trades to reconcile.")
            return 0

        reconciled_count = 0
        current_trade_ids: Set[int] = {t["id"] for t in db_trades}

        # Purge tracking caches for closed trades
        with reconciler_lock:
            for cached_id in list(consecutive_zero_counts.keys()):
                if cached_id not in current_trade_ids:
                    del consecutive_zero_counts[cached_id]
            for cached_id in list(first_zero_at.keys()):
                if cached_id not in current_trade_ids:
                    del first_zero_at[cached_id]
            for cached_id in list(ENTRY_ORDER_CACHE.keys()):
                if cached_id not in current_trade_ids:
                    del ENTRY_ORDER_CACHE[cached_id]

        live_exchange_positions = fetch_all_live_exchange_positions(self.executor)
        tracked_ccxt_symbols: Set[str] = set()

        for trade in db_trades:
            trade_id = trade["id"]
            pair = trade["pair"]
            direction = str(trade["direction"]).upper()
            entry_price = safe_float(trade.get("entry_price"))
            position_size = safe_float(trade.get("position_size"))
            account_balance = safe_float(trade.get("account_balance"), 100.0)
            db_sl = safe_float(trade.get("stop_loss"))
            db_tp = safe_float(trade.get("take_profit"))

            ccxt_symbol = format_ccxt_futures_symbol(pair)
            normalized_db_pair = normalize_pair(pair)
            tracked_ccxt_symbols.add(ccxt_symbol)

            try:
                pos_info = self.executor.get_futures_position(ccxt_symbol)

                if not pos_info.get("error") and safe_float(pos_info.get("contracts", 0)) < DUST_THRESHOLD:
                    pos_info_fallback = self.executor.get_futures_position(pair)
                    if safe_float(pos_info_fallback.get("contracts", 0)) > DUST_THRESHOLD:
                        pos_info = pos_info_fallback

                if not pos_info or pos_info.get("error", True) or "contracts" not in pos_info:
                    logger.warning(f"[{pair}] API or connectivity glitch detected during check. Skipping trade #{trade_id}.")
                    continue

                live_contracts = safe_float(pos_info.get("contracts", 0))

                # Handle missing or zero position on exchange
                if live_contracts < DUST_THRESHOLD:
                    terminal_status = check_entry_order_never_filled(self.executor, trade_id, ccxt_symbol)
                    if terminal_status:
                        logger.warning(f"[{pair}] Trade #{trade_id} entry order terminated ({terminal_status}) without fill. Marking CANCELLED.")
                        try:
                            finalize_trade_in_db(
                                trade_id=trade_id,
                                exit_price=entry_price,
                                pnl_usd=0.0,
                                pnl_pct=0.0,
                                outcome="CANCELLED",
                                fee_usd=0.0,
                            )
                        except Exception as cancel_err:
                            logger.error(f"[{pair}] Failed to mark trade #{trade_id} CANCELLED: {cancel_err}")

                        with reconciler_lock:
                            consecutive_zero_counts.pop(trade_id, None)
                            first_zero_at.pop(trade_id, None)
                            ENTRY_ORDER_CACHE.pop(trade_id, None)

                        reconciled_count += 1
                        send_telegram_notification(
                            f"<b>🛑 ENTRY ORDER CANCELLED</b>\n\n"
                            f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                            f"<b>Pair:</b> <code>{pair}</code>\n"
                            f"<b>Reason:</b> <code>Entry order {terminal_status} without fill</code>"
                        )
                        continue

                    if check_unprocessed_execution_close(self.executor, pair):
                        with reconciler_lock:
                            consecutive_zero_counts[trade_id] = 0
                            first_zero_at.pop(trade_id, None)
                        continue

                    real_pnl_pre, real_fee_pre, real_exit_pre = self.executor.fetch_real_closed_pnl(pair)
                    if real_exit_pre > 0:
                        logger.info(f"[{pair}] Trade #{trade_id} confirmed closed via closed-pnl endpoint. Finalizing directly.")
                        pnl_usd_pre, pnl_pct_pre, outcome_pre = calculate_pnl(
                            direction=direction,
                            entry_price=entry_price,
                            current_price=real_exit_pre,
                            quantity=position_size,
                            account_balance=account_balance,
                            total_fees=real_fee_pre,
                            exchange_closed_pnl=real_pnl_pre,
                        )
                        if real_pnl_pre is None and real_exit_pre == entry_price:
                            outcome_pre = "UNKNOWN"

                        try:
                            finalize_trade_in_db(
                                trade_id=trade_id,
                                exit_price=real_exit_pre,
                                pnl_usd=pnl_usd_pre,
                                pnl_pct=pnl_pct_pre,
                                outcome=outcome_pre,
                                fee_usd=real_fee_pre,
                            )
                        except Exception as fin_err:
                            logger.error(f"[{pair}] Direct finalize failed for #{trade_id}: {fin_err}")
                            continue

                        set_asset_cooldown(pair)
                        try:
                            asyncio.run(event_bus.guard.release_trade_lock(pair))
                        except Exception:
                            pass

                        reconciled_count += 1
                        with reconciler_lock:
                            consecutive_zero_counts.pop(trade_id, None)
                            first_zero_at.pop(trade_id, None)
                            ENTRY_ORDER_CACHE.pop(trade_id, None)

                        send_telegram_notification(
                            f"<b>✅ TRADE FINALIZED VIA CLOSED-PNL</b>\n\n"
                            f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                            f"<b>Pair:</b> <code>{pair}</code>\n"
                            f"<b>Exit:</b> <code>${real_exit_pre:.5f}</code>\n"
                            f"<b>Net PnL:</b> <code>${pnl_usd_pre:.2f}</code>"
                        )
                        continue

                    if check_active_or_pending_orders(self.executor, ccxt_symbol):
                        with reconciler_lock:
                            consecutive_zero_counts[trade_id] = 0
                            first_zero_at.pop(trade_id, None)
                        continue

                    now_ts = time.time()
                    with reconciler_lock:
                        if trade_id not in first_zero_at:
                            first_zero_at[trade_id] = now_ts
                        zero_duration = now_ts - first_zero_at[trade_id]
                        consecutive_zero_counts[trade_id] = consecutive_zero_counts.get(trade_id, 0) + 1
                        current_zeros = consecutive_zero_counts[trade_id]

                    if current_zeros < RETRY_THRESHOLD or zero_duration < GHOST_MIN_SUSTAINED_ZERO_SECONDS:
                        logger.info(
                            f"[{pair}] Zero contracts detected "
                            f"({current_zeros}/{RETRY_THRESHOLD}, "
                            f"{zero_duration:.0f}s/{GHOST_MIN_SUSTAINED_ZERO_SECONDS}s). "
                            f"Awaiting confirmation."
                        )
                        continue

                    real_closed_pnl, real_fee, fetched_exit = self.executor.fetch_real_closed_pnl(pair)
                    if real_closed_pnl is None and fetched_exit == 0.0:
                        logger.warning(
                            f"[{pair}] Trade #{trade_id} reported 0 contracts, "
                            f"but Bybit closedPnl shows NO close record. Skipping ghost close."
                        )
                        continue

                    logger.warning(f"🚨 GHOST DB RECORD CONFIRMED: Trade #{trade_id} ({pair}) sustained zero contracts for {zero_duration:.0f}s. Auto-closing...")

                    exit_price = fetched_exit if fetched_exit > 0 else entry_price
                    if exit_price == entry_price:
                        logger.warning(f"[{pair}] No verified exit for ghost #{trade_id}. Defaulting exit price to entry (${entry_price}).")

                    pnl_usd, pnl_pct, outcome = calculate_pnl(
                        direction=direction,
                        entry_price=entry_price,
                        current_price=exit_price,
                        quantity=position_size,
                        account_balance=account_balance,
                        total_fees=real_fee,
                        exchange_closed_pnl=real_closed_pnl
                    )

                    if real_closed_pnl is None and exit_price == entry_price:
                        outcome = "GHOST_CLOSED"

                    finalize_trade_in_db(
                        trade_id=trade_id,
                        exit_price=exit_price,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        outcome=outcome,
                        fee_usd=real_fee
                    )

                    set_asset_cooldown(pair)
                    try:
                        asyncio.run(event_bus.guard.release_trade_lock(pair))
                    except Exception:
                        pass

                    reconciled_count += 1

                    with reconciler_lock:
                        consecutive_zero_counts.pop(trade_id, None)
                        first_zero_at.pop(trade_id, None)
                        ENTRY_ORDER_CACHE.pop(trade_id, None)

                    send_telegram_notification(
                        f"<b>⚠️ DB RECONCILIATION APPLIED</b>\n\n"
                        f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                        f"<b>Pair:</b> <code>{pair}</code>\n"
                        f"<b>Action:</b> <code>AUTO_CLOSED (Ghost Record)</code>\n"
                        f"<b>Exchange PnL:</b> <code>${real_closed_pnl}</code>"
                    )

                # Handle active open position on exchange
                else:
                    with reconciler_lock:
                        consecutive_zero_counts[trade_id] = 0
                        first_zero_at.pop(trade_id, None)

                    ex_pos = live_exchange_positions.get(ccxt_symbol)
                    if not ex_pos:
                        for pos_data in live_exchange_positions.values():
                            if pos_data.get("normalized_symbol") == normalized_db_pair:
                                ex_pos = pos_data
                                break

                    if ex_pos:
                        live_sl = ex_pos.get("stop_loss", 0.0)
                        live_tp = ex_pos.get("take_profit", 0.0)

                        if (live_sl > 0 and db_sl == 0.0) or (live_tp > 0 and db_tp == 0.0):
                            conn_sync = get_db_connection()
                            if conn_sync:
                                try:
                                    with conn_sync.cursor() as cursor:
                                        cursor.execute("""
                                            UPDATE trade_setups 
                                            SET stop_loss = COALESCE(NULLIF(%s, 0.0), stop_loss),
                                                take_profit = COALESCE(NULLIF(%s, 0.0), take_profit),
                                                updated_at = CURRENT_TIMESTAMP
                                            WHERE id = %s;
                                        """, (live_sl, live_tp, trade_id))
                                        conn_sync.commit()
                                        logger.info(f"[{pair}] Synced exchange SL/TP (${live_sl}/${live_tp}) -> DB for Trade #{trade_id}")
                                except Exception as sync_err:
                                    logger.error(f"Failed to sync exchange SL/TP to DB: {sync_err}")
                                finally:
                                    release_db_connection(conn_sync)

                        # Re-attach missing Stop Loss on exchange if present in DB
                        if db_sl > 0 and (live_sl == 0.0 or abs(db_sl - live_sl) / db_sl > 0.005):
                            logger.warning(f"[{pair}] DB Stop Loss (${db_sl:.5f}) misaligned with exchange (${live_sl:.5f}). Re-aligning...")
                            attached = self.executor.set_position_trading_stop(
                                symbol=pair,
                                stop_loss=db_sl,
                                take_profit=db_tp if db_tp > 0 else None,
                                position_idx=ex_pos.get("position_idx", 0),
                                direction=direction
                            )
                            if attached:
                                logger.info(f"[{pair}] Reconciler successfully aligned Stop Loss on exchange.")
                            else:
                                logger.error(f"[{pair}] Reconciler FAILED to align Stop Loss on exchange.")

            except Exception as trade_err:
                logger.error(f"Error during trade #{trade_id} reconciliation: {trade_err}", exc_info=True)

        # Check for untracked/orphan positions on exchange
        for ccxt_sym, pos in live_exchange_positions.items():
            if is_symbol_under_grace_period(ccxt_sym):
                logger.info(f"[{ccxt_sym}] Symbol is within grace period ({GRACE_PERIOD_SECONDS}s). Skipping orphan check.")
                continue

            normalized_sym = pos.get("normalized_symbol", "")
            db_has_symbol = any(
                normalize_pair(t["pair"]) == normalized_sym for t in db_trades
            )

            if not db_has_symbol and pos.get("contracts", 0.0) > DUST_THRESHOLD:
                logger.warning(
                    f"⚠️ ORPHAN POSITION DETECTED: {pos['raw_symbol']} has {pos['contracts']} "
                    f"contracts active on Bybit but 0 active setups in DB!"
                )
                send_telegram_notification(
                    f"<b>🚨 UNTRACKED/ORPHAN POSITION DETECTED</b>\n\n"
                    f"<b>Symbol:</b> <code>{pos['raw_symbol']}</code>\n"
                    f"<b>Side:</b> <code>{pos['side'].upper()}</code>\n"
                    f"<b>Size:</b> <code>{pos['contracts']}</code>\n"
                    f"<b>Entry Price:</b> <code>${pos['entry_price']:.5f}</code>\n\n"
                    f"<i>Manual verification recommended via exchange UI.</i>"
                )

        logger.info(f"[Reconciler] Reconciliation cycle finished. Reconciled count: {reconciled_count}")
        return reconciled_count


def reconcile_open_trades(executor: Optional[BybitFuturesLiveExecutor] = None) -> int:
    """Functional wrapper for scheduled reconciliation loops."""
    reconciler = PositionReconciler(executor=executor)
    return reconciler.run_reconciliation_cycle()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Reconciler test cycle...")
    reconciler = PositionReconciler()
    reconciler.run_reconciliation_cycle()