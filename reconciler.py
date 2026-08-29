import logging
from common import (
    get_db_connection,
    release_db_connection,
    send_telegram_notification,
    set_asset_cooldown
)
from live_executor import MEXCLiveExecutor

logger = logging.getLogger("reconciler")

# Minimum asset threshold below which a token is considered sold/absent
DUST_THRESHOLD = 0.001


def reconcile_open_trades(executor: MEXCLiveExecutor) -> int:
    """
    Scans database and exchange state to perform auto-healing and orphan reconciliation.
    - If DB is OPEN but physical balance is missing/dust (< DUST_THRESHOLD): Cancels open orders on MEXC and marks DB CLOSED.
    - If physical balance exists on MEXC (> DUST_THRESHOLD) but DB lacks active trade: Auto-heals DB record to OPEN.
    
    Returns count of updated/reconciled trades.
    """
    logger.info("🔍 Running Database <-> Exchange Position Reconciliation & Auto-Healing...")

    conn = get_db_connection()
    if not conn:
        logger.error("Reconciliation skipped: Unable to acquire database connection.")
        return 0

    reconciled_count = 0

    try:
        cursor = conn.cursor()
        
        # 1. Fetch all trades currently marked as OPEN in DB
        cursor.execute(
            """
            SELECT id, pair, direction, entry_price, position_size 
            FROM trade_setups 
            WHERE trade_state IN ('OPEN', 'BE_LOCKED', 'TRAILING');
            """
        )
        open_db_trades = cursor.fetchall()
        open_db_pairs = {trade[1]: trade for trade in open_db_trades}

        # 2. Iterate through open database records
        for trade in open_db_trades:
            trade_id, pair, direction, entry_price, position_size = trade
            base_asset = pair.split('/')[0].upper()

            # Query real-time spot balance directly from MEXC
            live_balance = executor.get_spot_balance(pair)

            # Case A: Database claims OPEN, but physical tokens are missing on MEXC
            if live_balance < DUST_THRESHOLD:
                logger.warning(
                    f"⚠️ ORPHAN DETECTED: Trade #{trade_id} ({pair}) is OPEN in DB, "
                    f"but MEXC balance is {live_balance:.5f} (Below limit {DUST_THRESHOLD}). "
                    f"Auto-closing DB record..."
                )

                # Cancel any hanging open limit orders on MEXC before closing DB record
                try:
                    executor.cancel_all_open_orders(pair)
                    logger.info(f"[{pair}] Cancelled hanging exchange orders during reconciliation.")
                except Exception as cancel_err:
                    logger.error(f"[{pair}] Failed to cancel open orders for {pair}: {cancel_err}")

                # Update database trade status to CLOSED
                cursor.execute(
                    """
                    UPDATE trade_setups 
                    SET trade_state = 'CLOSED', status = 'AUTO_RECONCILED' 
                    WHERE id = %s;
                    """,
                    (trade_id,)
                )
                conn.commit()

                # Apply 2-hour cooldown for auto-reconciled asset
                set_asset_cooldown(pair, hours=2)

                reconciled_count += 1

                send_telegram_notification(
                    f"<b>⚠️ DB RECONCILIATION APPLIED</b>\n\n"
                    f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                    f"<b>Pair:</b> <code>{pair}</code>\n"
                    f"<b>Action:</b> <code>AUTO_CLOSED (Orphan Record)</code>\n"
                    f"<b>Reason:</b> Token balance missing on MEXC spot wallet ({live_balance:.4f} {base_asset})"
                )

        # 3. Check Auto-Healing Case: Physical tokens present on MEXC, but missing active DB record
        # Example watchlist symbols check or query spot balances
        try:
            balances = executor.get_all_spot_balances() if hasattr(executor, 'get_all_spot_balances') else {}
            for pair, balance in balances.items():
                if balance >= DUST_THRESHOLD and pair not in open_db_pairs:
                    logger.warning(f"🛠️ AUTO-HEAL TRIGGERED: {pair} has physical balance ({balance:.4f}) but no active DB trade.")
                    
                    # Fetch recent execution trade price from MEXC API
                    recent_trades = executor.get_recent_trades(pair) if hasattr(executor, 'get_recent_trades') else []
                    entry_p = float(recent_trades[0]['price']) if recent_trades else executor.get_current_price(pair)
                    
                    initial_sl = entry_p * 0.98  # Default 2% protective SL
                    initial_tp = entry_p * 1.04  # Default 4% TP

                    cursor.execute(
                        """
                        INSERT INTO trade_setups 
                        (pair, direction, entry_price, stop_loss, take_profit, position_size, 
                         trade_state, status, highest_price, trailing_stop_price) 
                        VALUES (%s, 'BUY', %s, %s, %s, %s, 'OPEN', 'AUTO_HEALED', %s, %s);
                        """,
                        (pair, entry_p, initial_sl, initial_tp, balance, entry_p, initial_sl)
                    )
                    conn.commit()
                    reconciled_count += 1

                    send_telegram_notification(
                        f"<b>🛠️ DB AUTO-HEALED POSITION RESTORED</b>\n\n"
                        f"<b>Pair:</b> <code>{pair}</code>\n"
                        f"<b>Balance:</b> {balance:.4f}\n"
                        f"<b>Est. Entry Price:</b> ${entry_p:.5f}\n"
                        f"<b>Status:</b> Re-adopted into active trade tracking engine."
                    )
        except Exception as heal_err:
            logger.error(f"Error during balance auto-healing scan: {heal_err}")

    except Exception as e:
        logger.error(f"Error occurred during reconciliation cycle: {e}")
        if conn:
            conn.rollback()
    finally:
        release_db_connection(conn)

    logger.info(f"Reconciliation cycle complete. Total records updated: {reconciled_count}")
    return reconciled_count