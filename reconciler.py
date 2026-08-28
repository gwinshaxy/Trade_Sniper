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
    Scans the database for 'OPEN' trades, compares them against real-time
    MEXC spot holdings, and auto-closes any database records where physical tokens 
    are missing on the exchange. Applies a 2-hour cooldown to auto-closed assets.
    
    Returns the count of reconciled/closed orphan trades.
    """
    logger.info("🔍 Running Database <-> Exchange Position Reconciliation...")

    conn = get_db_connection()
    if not conn:
        logger.error("Reconciliation skipped: Unable to acquire database connection.")
        return 0

    reconciled_count = 0

    try:
        cursor = conn.cursor()
        
        # 1. Fetch all trades currently marked as OPEN
        cursor.execute(
            """
            SELECT id, pair, direction, entry_price, position_size 
            FROM trade_setups 
            WHERE trade_state = 'OPEN';
            """
        )
        open_trades = cursor.fetchall()

        if not open_trades:
            logger.info("Reconciliation complete: No open database trades found.")
            return 0

        # 2. Iterate through open database trades
        for trade in open_trades:
            trade_id, pair, direction, entry_price, position_size = trade
            base_asset = pair.split('/')[0].upper()

            # Query real-time spot balance directly from MEXC
            live_balance = executor.get_spot_balance(pair)

            # 3. Check for Orphan State: Database claims OPEN, but live balance is missing/dust
            if live_balance < DUST_THRESHOLD:
                logger.warning(
                    f"⚠️ ORPHAN DETECTED: Trade #{trade_id} ({pair}) is OPEN in DB, "
                    f"but MEXC balance is {live_balance:.5f} (Below limit {DUST_THRESHOLD}). "
                    f"Auto-closing DB record..."
                )

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

                # Apply 2-hour cooldown for auto-reconciled orphan trade
                set_asset_cooldown(pair, hours=2)

                reconciled_count += 1

                # Send Telegram alert for manual audit transparency
                send_telegram_notification(
                    f"<b>⚠️ DB RECONCILIATION APPLIED</b>\n\n"
                    f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                    f"<b>Pair:</b> <code>{pair}</code>\n"
                    f"<b>Action:</b> <code>AUTO_CLOSED (Orphan Record)</code>\n"
                    f"<b>Reason:</b> Token balance missing on MEXC spot wallet ({live_balance:.4f} {base_asset})"
                )
            else:
                logger.info(
                    f"✅ Verified Trade #{trade_id} ({pair}): DB open position match found on exchange "
                    f"(Balance: {live_balance:.4f} {base_asset})."
                )

    except Exception as e:
        logger.error(f"Error occurred during reconciliation cycle: {e}")
        if conn:
            conn.rollback()
    finally:
        release_db_connection(conn)

    logger.info(f"Reconciliation cycle complete. Total orphan trades closed: {reconciled_count}")
    return reconciled_count