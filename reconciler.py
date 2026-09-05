import logging
import time
from typing import Dict, List, Set
from common import (
    get_db_connection,
    release_db_connection,
    send_telegram_notification,
    set_asset_cooldown,
    finalize_trade_in_db
)
from live_executor import BybitFuturesLiveExecutor, format_ccxt_futures_symbol

logger = logging.getLogger("reconciler")

DUST_THRESHOLD = 0.001
RETRY_THRESHOLD = 3

consecutive_zero_counts: Dict[int, int] = {}


def check_active_or_pending_orders(executor: BybitFuturesLiveExecutor, ccxt_symbol: str) -> bool:
    try:
        open_orders = executor.exchange.fetch_open_orders(ccxt_symbol)
        if open_orders:
            return True

        since = int((time.time() - 300) * 1000)
        recent_closed = executor.exchange.fetch_closed_orders(ccxt_symbol, since=since, limit=5)
        for order in recent_closed:
            status = str(order.get("status", "")).lower()
            if status in ["open", "untriggered", "new"]:
                return True
    except Exception:
        return False
    return False


def fetch_all_live_exchange_positions(executor: BybitFuturesLiveExecutor) -> Dict[str, dict]:
    active_positions = {}
    try:
        positions = executor.exchange.fetch_positions()
        for p in positions:
            contracts = float(p.get("contracts", 0) or 0)
            if contracts > DUST_THRESHOLD:
                raw_symbol = p.get("symbol", "")
                ccxt_symbol = format_ccxt_futures_symbol(raw_symbol)
                side = str(p.get("side", "")).lower()
                if not side or side == "none":
                    side = "long" if float(p.get("side", 0) or 0) > 0 else "short"

                active_positions[ccxt_symbol] = {
                    "symbol": ccxt_symbol,
                    "side": side,
                    "contracts": contracts,
                    "entry_price": float(p.get("entryPrice", 0) or 0),
                    "stop_loss": float(p.get("stopLoss", 0) or 0),       # Fetch live SL from Bybit
                    "take_profit": float(p.get("takeProfit", 0) or 0),   # Fetch live TP from Bybit
                    "unrealized_pnl": float(p.get("unrealizedPnl", 0) or 0),
                    "leverage": float(p.get("leverage", 1) or 1)
                }
    except Exception as e:
        logger.error(f"Reconciler: Failed to fetch live exchange positions: {e}")
    return active_positions


def reconcile_open_trades(executor: BybitFuturesLiveExecutor) -> int:
    global consecutive_zero_counts
    logger.info("🔍 Running Database <-> Bybit Futures Position Reconciliation & Auto-Healing...")

    open_db_trades = []
    conn = get_db_connection()
    if not conn:
        return 0

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, pair, direction, entry_price, position_size, account_balance, stop_loss, take_profit 
                FROM trade_setups 
                WHERE trade_state IN ('OPEN', 'EXECUTED', 'BE_LOCKED', 'TRAILING');
                """
            )
            open_db_trades = cursor.fetchall()
    except Exception as e:
        logger.error(f"Error fetching open trades for reconciliation: {e}")
        return 0
    finally:
        release_db_connection(conn)

    reconciled_count = 0
    current_trade_ids: Set[int] = {trade[0] for trade in open_db_trades}

    for cached_id in list(consecutive_zero_counts.keys()):
        if cached_id not in current_trade_ids:
            del consecutive_zero_counts[cached_id]

    live_exchange_positions = fetch_all_live_exchange_positions(executor)
    tracked_ccxt_symbols: Set[str] = set()

    for trade in open_db_trades:
        trade_id, pair, direction, entry_price, position_size, account_balance, db_sl, db_tp = trade
        ccxt_symbol = format_ccxt_futures_symbol(pair)
        tracked_ccxt_symbols.add(ccxt_symbol)

        try:
            pos_info = executor.get_futures_position(ccxt_symbol)

            # Skip cycle if fetching positions resulted in an API error
            if pos_info.get("error", False):
                logger.warning(f"[{pair}] Skipping reconciliation check due to API fetch error.")
                continue

            live_contracts = pos_info.get("contracts", 0.0)

            if live_contracts < DUST_THRESHOLD:
                if check_active_or_pending_orders(executor, ccxt_symbol):
                    consecutive_zero_counts[trade_id] = 0
                    continue

                consecutive_zero_counts[trade_id] = consecutive_zero_counts.get(trade_id, 0) + 1
                current_zeros = consecutive_zero_counts[trade_id]

                if current_zeros < RETRY_THRESHOLD:
                    continue

                logger.warning(f"🚨 GHOST DB RECORD CONFIRMED: Trade #{trade_id} ({pair}) reached zero checks limit. Auto-closing...")
                
                exit_price = float(entry_price)
                try:
                    ticker = executor.exchange.fetch_ticker(ccxt_symbol)
                    exit_price = float(ticker.get('last') or ticker.get('close') or entry_price)
                except Exception:
                    pass

                bal = float(account_balance or 100.0)
                from common import calculate_pnl
                pnl_usd, pnl_pct, outcome = calculate_pnl(direction, float(entry_price), exit_price, float(position_size), bal)

                finalize_trade_in_db(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    pnl_usd=pnl_usd,
                    pnl_pct=pnl_pct,
                    outcome=outcome
                )

                set_asset_cooldown(pair, hours=2)
                reconciled_count += 1
                del consecutive_zero_counts[trade_id]

                send_telegram_notification(
                    f"<b>⚠️ DB RECONCILIATION APPLIED</b>\n\n"
                    f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                    f"<b>Pair:</b> <code>{pair}</code>\n"
                    f"<b>Action:</b> <code>AUTO_CLOSED (Ghost Record)</code>\n"
                    f"<b>Reason:</b> Position closed on exchange over {RETRY_THRESHOLD} checks."
                )
            else:
                consecutive_zero_counts[trade_id] = 0

                # SYNC SL/TP TO DB IF POSITIONS EXIST ON BYBIT BUT DB HAS NONE/0
                ex_pos = live_exchange_positions.get(ccxt_symbol)
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
                                            take_profit = COALESCE(NULLIF(%s, 0.0), take_profit)
                                        WHERE id = %s;
                                    """, (live_sl, live_tp, trade_id))
                                    conn_sync.commit()
                                    logger.info(f"Updated DB SL/TP for Trade #{trade_id} from exchange live parameters.")
                            except Exception as sync_err:
                                logger.error(f"Failed to sync exchange SL/TP to DB for trade #{trade_id}: {sync_err}")
                            finally:
                                release_db_connection(conn_sync)

        except Exception as err:
            logger.error(f"Error reconciling trade ID #{trade_id}: {err}")

    # Auto-adopt un-tracked exchange positions into DB
    if live_exchange_positions:
        for ex_symbol, ex_pos in live_exchange_positions.items():
            if ex_symbol not in tracked_ccxt_symbols:
                logger.warning(f"🚨 ORPHAN POSITION DETECTED: {ex_symbol}. Inserting missing position into DB...")
                
                conn_adopt = get_db_connection()
                if conn_adopt:
                    try:
                        with conn_adopt.cursor() as cursor:
                            db_pair = ex_symbol
                            direction = "BUY" if ex_pos['side'].lower() in ["long", "buy"] else "SELL"
                            entry_price = float(ex_pos.get('entry_price', 0.0))
                            position_size = float(ex_pos.get('contracts', 0.0))
                            
                            cursor.execute("""
                                INSERT INTO trade_setups (
                                    pair, direction, entry_price, position_size, 
                                    stop_loss, take_profit, status, trade_state, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN', NOW())
                                RETURNING id;
                            """, (
                                db_pair, 
                                direction, 
                                entry_price, 
                                position_size,
                                ex_pos.get('stop_loss', 0.0),
                                ex_pos.get('take_profit', 0.0)
                            ))
                            
                            new_id = cursor.fetchone()[0]
                            conn_adopt.commit()
                            reconciled_count += 1
                            
                            send_telegram_notification(
                                f"<b>✅ AUTO-ADOPTED EXCHANGE POSITION</b>\n\n"
                                f"<b>Trade ID:</b> <code>#{new_id}</code>\n"
                                f"<b>Pair:</b> <code>{db_pair}</code>\n"
                                f"<b>Side:</b> <code>{direction}</code>\n"
                                f"<b>Contracts:</b> <code>{position_size}</code>\n"
                                f"<b>SL:</b> <code>${ex_pos.get('stop_loss', 0.0)}</code> | <b>TP:</b> <code>${ex_pos.get('take_profit', 0.0)}</code>"
                            )
                    except Exception as db_err:
                        logger.error(f"Failed to auto-insert orphan position for {ex_symbol}: {db_err}")
                    finally:
                        release_db_connection(conn_adopt)

    return reconciled_count