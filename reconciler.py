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

# Require BOTH 3 checks AND a minimum sustained-zero duration before
# declaring a ghost. Bybit testnet's position endpoint can lag 5-10 minutes
# after a close, so a pure check-count threshold produces false positives.
GHOST_MIN_SUSTAINED_ZERO_SECONDS = 600  # 10 minutes

reconciler_lock = threading.Lock()
consecutive_zero_counts: Dict[int, int] = {}
# Track the wall-clock time each trade first read zero contracts.
first_zero_at: Dict[int, float] = {}

RECENTLY_OPENED_CACHE: Dict[str, float] = {}
GRACE_PERIOD_SECONDS = 15.0

# In-memory cache of entry order IDs per trade_id so we can query
# the order status directly when the position endpoint reports zero.
ENTRY_ORDER_CACHE: Dict[int, str] = {}


# =====================================================================
# FIX 1: Unified Symbol Stripping in Position Matching
# =====================================================================
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
            # Skip protective SL/TP orders — they don't indicate an entry in flight
            if stop_order_type in ("StopLoss", "TakeProfit", "TrailingStop", "Stop"):
                continue
            # Any non-protective open order counts as pending entry activity
            return True

        since = int((time.time() - 300) * 1000)
        recent_closed = executor.exchange.fetch_closed_orders(ccxt_symbol, since=since, limit=10)
        for order in recent_closed:
            status = str(order.get("status", "")).lower()
            if status not in ["open", "untriggered", "new"]:
                continue
            info = order.get("info", {}) or {}
            stop_order_type = str(info.get("stopOrderType", "") or "").strip()
            # Same protective-order filter for closed-but-still-conditional orders
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
    STALE_MS = 60_000  # 60 seconds

    try:
        positions = executor.exchange.fetch_positions()
        for p in positions:
            contracts = float(p.get("contracts", 0) or 0)
            if contracts <= DUST_THRESHOLD:
                continue

            info = p.get("info", {}) or {}

            # Skip stale positions
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

            # Prefer raw Bybit fields; fall back to CCXT normalized keys.
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

    # Purge stale counters/caches for trades no longer open
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
                # Entry order never filled? Mark CANCELLED, not ghost.
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

                # Unprocessed execution/list check
                if check_unprocessed_execution_close(executor, pair):
                    with reconciler_lock:
                        consecutive_zero_counts[trade_id] = 0
                        first_zero_at.pop(trade_id, None)
                    continue

                # Authoritative closed-pnl check BEFORE counting zero.
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

                # Protective-order suppression
                if check_active_or_pending_orders(executor, ccxt_symbol):
                    with reconciler_lock:
                        consecutive_zero_counts[trade_id] = 0
                        first_zero_at.pop(trade_id, None)
                    continue

                # Time-based ghost detection counter
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

                # =====================================================================
                # FIX 2: Authoritative Bybit Closed PnL / Execution Verification
                # =====================================================================
                real_closed_pnl, real_fee, fetched_exit = executor.fetch_real_closed_pnl(pair)
                if real_closed_pnl is None and fetched_exit == 0.0:
                    logger.warning(
                        f"[{pair}] Trade #{trade_id} reported 0 contracts, "
                        f"but Bybit closedPnl shows NO close record. Skipping ghost close to prevent false termination."
                    )
                    continue  # Skip closing this trade

                logger.warning(f"🚨 GHOST DB RECORD CONFIRMED: Trade #{trade_id} ({pair}) sustained zero contracts for {zero_duration:.0f}s. Auto-closing...")

                if fetched_exit > 0:
                    exit_price = fetched_exit
                else:
                    exit_price = float(entry_price)  # Not current ticker — use entry
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

                # Look up live exchange position by ccxt_symbol or normalized pair matching (Fix 1)
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

                    # Exchange has SL/TP, DB missing → sync exchange → DB
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
                                    logger.info(f"Updated DB SL/TP for Trade #{trade_id} from exchange live parameters.")
                            except Exception as sync_err:
                                logger.error(f"Failed to sync exchange SL/TP to DB for trade #{trade_id}: {sync_err}")
                            finally:
                                release_db_connection(conn_sync)

                    # Exchange missing SL, DB has one → re-attach from DB
                    if live_sl == 0.0 and db_sl_val > 0:
                        logger.warning(
                            f"[{pair}] Exchange SL missing for Trade #{trade_id} "
                            f"(DB has ${db_sl_val:.5f}). Re-attaching from DB..."
                        )
                        try:
                            reattach_idx = executor._get_position_idx(
                                ccxt_symbol, direction
                            )
                            ok = executor.set_position_trading_stop(
                                symbol=pair,
                                stop_loss=db_sl_val,
                                take_profit=(db_tp_val if db_tp_val > 0 else None),
                                position_idx=reattach_idx,
                                direction=direction,
                            )
                            if ok:
                                logger.info(
                                    f"[{pair}] SL re-attached from DB for Trade #{trade_id}."
                                )
                                send_telegram_notification(
                                    f"<b>🔧 SL RE-ATTACHED FROM DB</b>\n\n"
                                    f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                                    f"<b>Pair:</b> <code>{pair}</code>\n"
                                    f"<b>SL:</b> <code>${db_sl_val:.5f}</code>"
                                )
                            else:
                                logger.error(
                                    f"[{pair}] SL re-attach from DB FAILED for Trade #{trade_id}."
                                )
                        except Exception as reattach_err:
                            logger.error(
                                f"[{pair}] SL re-attach exception for Trade #{trade_id}: {reattach_err}"
                            )

        except Exception as err:
            logger.error(f"Error reconciling trade ID #{trade_id}: {err}")

    # =====================================================================
    # FIX 3: Sync Adopted Trades with Stop Loss / Take Profit & Deduplication
    # =====================================================================
    if live_exchange_positions:
        for ex_symbol, ex_pos in live_exchange_positions.items():
            ex_normalized = ex_pos.get("normalized_symbol") or normalize_pair(ex_symbol)
            
            # Check if symbol is already tracked via standard symbol or normalized string
            is_tracked = any(
                normalize_pair(tracked_sym) == ex_normalized 
                for tracked_sym in tracked_ccxt_symbols
            )
            
            if not is_tracked:
                if is_symbol_under_grace_period(ex_symbol):
                    logger.info(f"[{ex_symbol}] Orphan check bypassed: Symbol is within recent order execution grace period window.")
                    continue

                conn_adopt = get_db_connection()
                if conn_adopt:
                    try:
                        with conn_adopt.cursor() as cursor:
                            # Verify whether an active setup already exists in trade_setups using normalized pair comparison
                            cursor.execute("""
                                SELECT id, stop_loss, take_profit FROM trade_setups 
                                WHERE trade_state IN ('OPEN', 'EXECUTED', 'BE_LOCKED', 'TRAILING');
                            """)
                            existing_open_trades = cursor.fetchall()
                            
                            existing_trade_id = None
                            existing_sl = 0.0
                            existing_tp = 0.0
                            
                            for row in existing_open_trades:
                                t_id, t_sl, t_tp = row[0], float(row[1] or 0.0), float(row[2] or 0.0)
                                cursor.execute("SELECT pair FROM trade_setups WHERE id = %s;", (t_id,))
                                p_row = cursor.fetchone()
                                if p_row and normalize_pair(p_row[0]) == ex_normalized:
                                    existing_trade_id = t_id
                                    existing_sl = t_sl
                                    existing_tp = t_tp
                                    break

                            if existing_trade_id is not None:
                                logger.info(
                                    f"[{ex_symbol}] Active trade setup #{existing_trade_id} already exists for normalized pair "
                                    f"'{ex_normalized}'. Skipping insertion of duplicate orphan record."
                                )
                                continue

                            logger.warning(f"🚨 ORPHAN POSITION DETECTED: {ex_symbol}. Safely inserting/updating position in DB...")

                            db_pair = ex_symbol
                            direction = "BUY" if ex_pos['side'].lower() in ["long", "buy"] else "SELL"
                            entry_price = float(ex_pos.get('entry_price', 0.0))
                            position_size = float(ex_pos.get('contracts', 0.0))
                            adopt_sl = float(ex_pos.get('stop_loss', 0.0))
                            adopt_tp = float(ex_pos.get('take_profit', 0.0))

                            cursor.execute("""
                                INSERT INTO trade_setups (
                                    pair, direction, entry_price, position_size, 
                                    stop_loss, take_profit, status, trade_state, created_at, updated_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN', NOW(), NOW())
                                ON CONFLICT (pair) WHERE trade_state IN ('OPEN', 'EXECUTED', 'BE_LOCKED', 'TRAILING')
                                DO UPDATE SET
                                    position_size = EXCLUDED.position_size,
                                    entry_price = EXCLUDED.entry_price,
                                    stop_loss = COALESCE(NULLIF(EXCLUDED.stop_loss, 0.0), trade_setups.stop_loss),
                                    take_profit = COALESCE(NULLIF(EXCLUDED.take_profit, 0.0), trade_setups.take_profit),
                                    updated_at = CURRENT_TIMESTAMP
                                RETURNING id;
                            """, (
                                db_pair, direction, entry_price, position_size,
                                adopt_sl, adopt_tp
                            ))

                            row = cursor.fetchone()
                            if row:
                                new_id = row[0]
                                conn_adopt.commit()
                                reconciled_count += 1

                                send_telegram_notification(
                                    f"<b>✅ AUTO-ADOPTED EXCHANGE POSITION</b>\n\n"
                                    f"<b>Trade ID:</b> <code>#{new_id}</code>\n"
                                    f"<b>Pair:</b> <code>{db_pair}</code>\n"
                                    f"<b>Side:</b> <code>{direction}</code>\n"
                                    f"<b>Contracts:</b> <code>{position_size}</code>\n"
                                    f"<b>SL:</b> <code>${adopt_sl}</code> \vert{} <b>TP:</b> <code>${adopt_tp}</code>"
                                )
                    except Exception as db_err:
                        logger.error(f"Failed to auto-insert/upsert orphan position for {ex_symbol}: {db_err}")
                        conn_adopt.rollback()
                    finally:
                        release_db_connection(conn_adopt)

    return reconciled_count