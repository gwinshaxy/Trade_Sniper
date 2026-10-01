import logging
import time
import threading
from typing import Dict, List, Set, Optional
from common import (
    get_db_connection,
    release_db_connection,
    send_telegram_notification,
    set_asset_cooldown,
    finalize_trade_in_db,
    calculate_pnl
)
from live_executor import BybitFuturesLiveExecutor, format_ccxt_futures_symbol

logger = logging.getLogger("reconciler")

DUST_THRESHOLD = 0.001
RETRY_THRESHOLD = 3

# SOLUTION 4: Enhance Reconciler Grace Period & Closed PnL Check
# Extended GRACE_PERIOD_SECONDS to allow Bybit testnet execution engines and order books to settle.
GRACE_PERIOD_SECONDS = 120.0  # Increased from 15.0s to mitigate testnet endpoint lag
GHOST_MIN_SUSTAINED_ZERO_SECONDS = 900  # Increased to 15 minutes for testnet stability

reconciler_lock = threading.Lock()
consecutive_zero_counts: Dict[int, int] = {}
first_zero_at: Dict[int, float] = {}

RECENTLY_OPENED_CACHE: Dict[str, float] = {}

ENTRY_ORDER_CACHE: Dict[int, str] = {}


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
    formatted = format_ccxt_futures_symbol(ccxt_symbol)
    with reconciler_lock:
        RECENTLY_OPENED_CACHE[formatted] = time.time()
    logger.debug(f"[{formatted}] Added to RECENTLY_OPENED_CACHE.")


def is_symbol_under_grace_period(ccxt_symbol: str) -> bool:
    formatted = format_ccxt_futures_symbol(ccxt_symbol)
    with reconciler_lock:
        if formatted in RECENTLY_OPENED_CACHE:
            elapsed = time.time() - RECENTLY_OPENED_CACHE[formatted]
            if elapsed < GRACE_PERIOD_SECONDS:
                return True
            del RECENTLY_OPENED_CACHE[formatted]
    return False


def mark_trade_entry_order(trade_id: int, entry_order_id: str) -> None:
    """Register the entry order ID for a trade so the reconciler can
    query its authoritative status before declaring the trade a ghost."""
    if trade_id and entry_order_id:
        with reconciler_lock:
            ENTRY_ORDER_CACHE[trade_id] = str(entry_order_id)
        logger.debug(f"[Trade #{trade_id}] Registered entry_order_id={entry_order_id}.")


def check_active_or_pending_orders(executor: BybitFuturesLiveExecutor, ccxt_symbol: str) -> bool:
    """
    Filter out protective (SL/TP) orders. Only entry orders should
    suppress the zero-contract counter. Bybit's `stopOrderType` field is
    non-empty for SL/TP conditional orders — we skip those.
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
    the trade should be marked CANCELLED (not ghost-closed). Returns the
    terminal status string if the order never filled, else None.
    """
    with reconciler_lock:
        entry_order_id = ENTRY_ORDER_CACHE.get(trade_id)

    if not entry_order_id:
        return None

    try:
        order = executor.exchange.fetch_order(entry_order_id, ccxt_symbol)
        status = str(order.get("status", "")).lower()
        filled = float(order.get("filled", 0) or 0)

        if status in ("canceled", "rejected", "expired") and filled < DUST_THRESHOLD:
            logger.warning(
                f"[Trade #{trade_id}] Entry order {entry_order_id} never filled "
                f"(status={status}, filled={filled}). Marking as CANCELLED, not ghost-closed."
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
    Query /v5/execution/list for a recent close fill. If Bybit has
    already recorded a closing trade (closedSize > 0) that the DB hasn't
    processed, we must NOT count this cycle as a zero-contract confirmation.
    """
    try:
        execs = executor.fetch_recent_executions(symbol, lookback_ms=300_000)
        for e in execs:
            if str(e.get("execType", "")) != "Trade":
                continue
            closed_size = float(e.get("closedSize", 0) or 0)
            if closed_size > DUST_THRESHOLD:
                logger.info(
                    f"[{symbol}] Unprocessed close fill detected in execution/list "
                    f"(closedSize={closed_size}, price={e.get('execPrice')}). "
                    f"Not counting as ghost."
                )
                return True
    except Exception as e:
        logger.debug(f"[{symbol}] execution/list check failed: {e}")
    return False


def fetch_all_live_exchange_positions(executor: BybitFuturesLiveExecutor) -> Dict[str, dict]:
    """
    Filters out positions with stale updatedTime (>60s old) that Bybit's
    bulk /v5/position/list endpoint may return for recently-closed trades.
    This prevents false-positive ORPHAN detection immediately after a close.

    Reads stopLoss/takeProfit from the raw Bybit `info` dict because
    CCXT does not reliably populate the normalized top-level keys on Bybit V5.
    """
    active_positions = {}
    now_ms = int(time.time() * 1000)
    STALE_MS = 60_000

    try:
        positions = executor.exchange.fetch_positions()
        for p in positions:
            contracts = float(p.get("contracts", 0) or 0)
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
                side = "long" if float(p.get("side", 0) or 0) > 0 else "short"

            sl_raw = p.get("stopLoss")
            tp_raw = p.get("takeProfit")
            sl_val = float(sl_raw) if sl_raw not in (None, "") else float(info.get("stopLoss", 0) or 0)
            tp_val = float(tp_raw) if tp_raw not in (None, "") else float(info.get("takeProfit", 0) or 0)

            active_positions[ccxt_symbol] = {
                "symbol": ccxt_symbol,
                "raw_symbol": raw_symbol,
                "normalized_symbol": normalize_pair(raw_symbol),
                "side": side,
                "contracts": contracts,
                "entry_price": float(p.get("entryPrice", 0) or 0),
                "stop_loss": sl_val,
                "take_profit": tp_val,
                "unrealized_pnl": float(p.get("unrealizedPnl", 0) or 0),
                "leverage": float(p.get("leverage", 1) or 1),
            }
    except Exception as e:
        logger.error(f"Reconciler: Network/API error fetching bulk positions: {e}. Preserving DB state.")
        return {}

    return active_positions


def reconcile_open_trades(executor: BybitFuturesLiveExecutor) -> int:
    global consecutive_zero_counts, first_zero_at
    logger.info("🔍 Running Database <-> Bybit Futures Position Reconciliation & Auto-Healing...")

    open_db_trades = []
    conn = get_db_connection()
    if not conn:
        return 0

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, pair, direction, entry_price, position_size, account_balance, stop_loss, take_profit 
                FROM trade_setups 
                WHERE trade_state IN ('OPEN', 'EXECUTED', 'BE_LOCKED', 'TRAILING');
            """)
            open_db_trades = cursor.fetchall()
    except Exception as e:
        logger.error(f"Error fetching open trades for reconciliation: {e}")
        return 0
    finally:
        release_db_connection(conn)

    reconciled_count = 0
    current_trade_ids: Set[int] = {trade[0] for trade in open_db_trades}

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

    live_exchange_positions = fetch_all_live_exchange_positions(executor)
    tracked_ccxt_symbols: Set[str] = set()

    for trade in open_db_trades:
        trade_id, pair, direction, entry_price, position_size, account_balance, db_sl, db_tp = trade
        ccxt_symbol = format_ccxt_futures_symbol(pair)
        normalized_db_pair = normalize_pair(pair)
        tracked_ccxt_symbols.add(ccxt_symbol)

        try:
            pos_info = executor.get_futures_position(ccxt_symbol)

            if not pos_info.get("error") and pos_info.get("contracts", 0.0) < DUST_THRESHOLD:
                pos_info_fallback = executor.get_futures_position(pair)
                if pos_info_fallback.get("contracts", 0.0) > DUST_THRESHOLD:
                    pos_info = pos_info_fallback

            if not pos_info or pos_info.get("error", True) or "contracts" not in pos_info:
                logger.warning(f"[{pair}] API or connectivity glitch detected during check. Preserving state & skipping cycle.")
                continue

            live_contracts = float(pos_info.get("contracts", 0.0))

            if live_contracts < DUST_THRESHOLD:
                terminal_status = check_entry_order_never_filled(executor, trade_id, ccxt_symbol)
                if terminal_status:
                    logger.warning(
                        f"[{pair}] Trade #{trade_id} entry order terminated "
                        f"({terminal_status}) without fill. Marking CANCELLED."
                    )
                    try:
                        finalize_trade_in_db(
                            trade_id=trade_id,
                            exit_price=float(entry_price),
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

                if check_unprocessed_execution_close(executor, pair):
                    with reconciler_lock:
                        consecutive_zero_counts[trade_id] = 0
                        first_zero_at.pop(trade_id, None)
                    continue

                real_pnl_pre, real_fee_pre, real_exit_pre = executor.fetch_real_closed_pnl(pair)
                if real_exit_pre > 0:
                    logger.info(
                        f"[{pair}] Trade #{trade_id} already closed per closed-pnl "
                        f"endpoint (exit=${real_exit_pre:.5f}). Finalizing directly."
                    )
                    bal_pre = float(account_balance or 100.0)
                    pnl_usd_pre, pnl_pct_pre, outcome_pre = calculate_pnl(
                        direction=direction,
                        entry_price=float(entry_price),
                        current_price=real_exit_pre,
                        quantity=float(position_size),
                        account_balance=bal_pre,
                        total_fees=real_fee_pre,
                        exchange_closed_pnl=real_pnl_pre,
                    )
                    if real_pnl_pre is None and real_exit_pre == float(entry_price):
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

                if check_active_or_pending_orders(executor, ccxt_symbol):
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
                        f"Waiting for confirmation."
                    )
                    continue

                real_closed_pnl, real_fee, fetched_exit = executor.fetch_real_closed_pnl(pair)
                if real_closed_pnl is None and fetched_exit == 0.0:
                    logger.warning(
                        f"[{pair}] Trade #{trade_id} reported 0 contracts, "
                        f"but Bybit closedPnl shows NO close record. Skipping ghost close to prevent false termination."
                    )
                    continue

                logger.warning(f"🚨 GHOST DB RECORD CONFIRMED: Trade #{trade_id} ({pair}) sustained zero contracts for {zero_duration:.0f}s. Auto-closing...")

                if fetched_exit > 0:
                    exit_price = fetched_exit
                else:
                    exit_price = float(entry_price)
                    logger.warning(
                        f"[{pair}] No verified exit for ghost #{trade_id}. "
                        f"Using entry price (${entry_price}); PnL will reflect fees only."
                    )

                bal = float(account_balance or 100.0)
                pnl_usd, pnl_pct, outcome = calculate_pnl(
                    direction=direction,
                    entry_price=float(entry_price),
                    current_price=exit_price,
                    quantity=float(position_size),
                    account_balance=bal,
                    total_fees=real_fee,
                    exchange_closed_pnl=real_closed_pnl
                )

                if real_closed_pnl is None and exit_price == float(entry_price):
                    outcome = "UNKNOWN"

                finalize_trade_in_db(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    pnl_usd=pnl_usd,
                    pnl_pct=pnl_pct,
                    outcome=outcome,
                    fee_usd=real_fee
                )

                set_asset_cooldown(pair)
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
                    live_sl = ex_pos.get('stop_loss', 0.0)
                    live_tp = ex_pos.get('take_profit', 0.0)

                    db_sl_val = float(db_sl or 0.0)
                    db_tp_val = float(db_tp or 0.0)

                    if (live_sl > 0 and db_sl_val == 0.0) or (live_tp > 0 and db_tp_val == 0.0):
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

                    # Re-attach missing exchange SL
                    if live_sl == 0.0 and db_sl_val > 0.0:
                        logger.warning(f"[{pair}] DB has SL ${db_sl_val:.5f} but exchange has NO SL. Attempting re-attachment...")
                        attached = executor.set_position_trading_stop(
                            symbol=pair,
                            stop_loss=db_sl_val,
                            take_profit=db_tp_val if db_tp_val > 0 else None,
                            direction=direction
                        )
                        if attached:
                            logger.info(f"[{pair}] Reconciler successfully re-attached SL ${db_sl_val:.5f} on exchange.")
                        else:
                            logger.error(f"[{pair}] Reconciler FAILED to re-attach SL on exchange.")

        except Exception as trade_err:
            logger.error(f"Error during trade #{trade_id} reconciliation: {trade_err}")

    # Check for orphaned positions on exchange that are missing in the DB
    for ccxt_sym, pos in live_exchange_positions.items():
        if is_symbol_under_grace_period(ccxt_sym):
            logger.info(f"[{ccxt_sym}] Symbol is within grace period ({GRACE_PERIOD_SECONDS}s). Skipping orphan check.")
            continue

        normalized_sym = pos.get("normalized_symbol", "")
        db_has_symbol = any(
            normalize_pair(t[1]) == normalized_sym for t in open_db_trades
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

    return reconciled_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Reconciler test loop...")