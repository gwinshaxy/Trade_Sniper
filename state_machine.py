import asyncio
import logging
from common import (
    get_db_connection, 
    release_db_connection, 
    send_telegram_notification,
    set_asset_cooldown,
    calculate_pnl
)
from event_bus import event_bus
from live_executor import LiveExecutionEngine

try:
    from config import STRATEGY_CONFIG
except ImportError:
    STRATEGY_CONFIG = {"risk_pct": 1.0}

logger = logging.getLogger("state_machine")


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class StateMachineEngine:
    """Single-threaded state machine manager that processes trade signals, syncs state, and dispatches executions."""

    def __init__(self, executor: LiveExecutionEngine):
        self.executor = executor

    async def run(self):
        logger.info("State Machine Engine started. Waiting for events...")
        while True:
            event = await event_bus.consume()
            event_type = event.get("type")
            payload = event.get("payload", {})

            try:
                if event_type == "TRADE_SIGNAL":
                    await self._handle_trade_signal(payload)
                elif event_type == "TICKER_UPDATE":
                    await self._handle_ticker_update(payload)
                elif event_type == "EXECUTION_REPORT":
                    await self._handle_execution_report(payload)
            except Exception as e:
                logger.error(f"Error processing event {event_type}: {e}")

    async def _handle_trade_signal(self, payload: dict):
        symbol = payload["symbol"]
        
        # 1. Acquire lock FIRST before running DB or API pre-checks
        acquired = await event_bus.guard.try_acquire_trade_lock(symbol)
        if not acquired:
            logger.warning(f"[{symbol}] Active execution in-flight. Blocking duplicate order.")
            return

        try:
            # DB active trade count check
            conn_check = get_db_connection()
            if conn_check:
                try:
                    with conn_check.cursor() as cur:
                        cur.execute("""
                            SELECT COUNT(*) FROM trade_setups 
                            WHERE pair = %s AND status = 'EXECUTED' AND trade_state != 'CLOSED';
                        """, (symbol,))
                        if cur.fetchone()[0] > 0:
                            logger.warning(f"[{symbol}] Active trade already present in DB. Skipping.")
                            return
                finally:
                    release_db_connection(conn_check)

            direction = payload["direction"].upper()
            entry_price = safe_float(payload.get("entry_price"), 0.0)
            stop_loss = safe_float(payload.get("stop_loss"), 0.0)
            take_profit = safe_float(payload.get("take_profit"), 0.0)
            amount_usd = safe_float(payload.get("amount_usd"), 25.0)
            account_balance = safe_float(payload.get("account_balance"), 100.0)
            leverage = int(safe_float(payload.get("leverage"), 10))

            default_risk = safe_float(STRATEGY_CONFIG.get("risk_pct"), 1.0) if isinstance(STRATEGY_CONFIG, dict) else 1.0
            risk_pct_value = safe_float(payload.get("risk_pct"), default=default_risk)

            conn = get_db_connection()
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT COUNT(*) FROM trade_setups 
                            WHERE status = 'EXECUTED' AND trade_state != 'CLOSED';
                        """)
                        active_positions_count = cur.fetchone()[0]
                        if active_positions_count >= 4:
                            logger.warning(f"[{symbol}] Execution Blocked: Max open positions cap (4) reached.")
                            return
                finally:
                    release_db_connection(conn)

            loop = asyncio.get_running_loop()

            pos_info = await loop.run_in_executor(None, self.executor.get_futures_position, symbol)
            if pos_info.get("contracts", 0.0) > 0.001:
                logger.warning(
                    f"[{symbol}] Exchange Guard Triggered: Active {pos_info['side']} position "
                    f"({pos_info['contracts']} contracts) already exists on Bybit. Skipping execution."
                )
                return

            logger.info(
                f"[{symbol}] Processing {direction} trade signal via State Machine "
                f"(Ref Entry Price: ${entry_price:.5f} | Risk: {risk_pct_value}%)..."
            )

            exec_result = await loop.run_in_executor(
                None,
                self.executor.order_futures_bybit,
                symbol,
                direction,
                amount_usd,
                entry_price,
                stop_loss,
                take_profit,
                leverage,
                account_balance,
                risk_pct_value
            )

            if exec_result.get("status") == "SUCCESS":
                # Register symbol in grace-period cache IMMEDIATELY, before
                # the DB insert, so the reconciler skips adoption if it fires
                # between the WS execution event and our INSERT completing.
                try:
                    from reconciler import mark_symbol_recently_opened
                    mark_symbol_recently_opened(symbol)
                except Exception as grace_err:
                    logger.debug(f"[{symbol}] Grace registration failed: {grace_err}")

                executed_qty = exec_result["executed_qty"]
                fill_price = exec_result["fill_price"]
                sl_attached = exec_result.get("stop_loss_attached", False)

                conn_db = get_db_connection()
                if conn_db:
                    try:
                        with conn_db.cursor() as cur:
                            # -----------------------------------------------------------------
                            # FIX #P1: Use an ON CONFLICT upsert that matches the partial
                            # unique index (idx_unique_open_pair) which permits only ONE
                            # open trade per pair. This eliminates the race-condition
                            # duplicate-key error when the reconciler's ORPHAN-adoption
                            # path wins the insert race. We also RETURN id so the
                            # trade_id is always populated for notifications.
                            #
                            # NOTE: The WHERE clause in ON CONFLICT must match the
                            # partial index predicate exactly.
                            # -----------------------------------------------------------------
                            cur.execute("""
                                INSERT INTO trade_setups 
                                (pair, direction, entry_price, stop_loss, take_profit, position_size, account_balance, risk_pct, status, trade_state, updated_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN', NOW())
                                ON CONFLICT (pair) WHERE trade_state IN ('OPEN', 'EXECUTED', 'BE_LOCKED', 'TRAILING')
                                DO UPDATE SET
                                    direction = EXCLUDED.direction,
                                    entry_price = EXCLUDED.entry_price,
                                    stop_loss = COALESCE(NULLIF(EXCLUDED.stop_loss, 0.0), trade_setups.stop_loss),
                                    take_profit = COALESCE(NULLIF(EXCLUDED.take_profit, 0.0), trade_setups.take_profit),
                                    position_size = EXCLUDED.position_size,
                                    account_balance = EXCLUDED.account_balance,
                                    risk_pct = EXCLUDED.risk_pct,
                                    updated_at = NOW()
                                RETURNING id;
                            """, (symbol, direction, fill_price, stop_loss, take_profit, executed_qty, account_balance, risk_pct_value))
                            row = cur.fetchone()
                            trade_id = row[0] if row else None
                            conn_db.commit()

                        if trade_id is not None:
                            emoji = "🟢" if direction in ["BUY", "LONG"] else "🔴"
                            sl_status = "✅ Attached" if sl_attached else "⚠️ Fallback Active"
                            send_telegram_notification(
                                f"<b>{emoji} LIVE BYBIT FUTURES ORDER EXECUTED ({direction})</b>\n\n"
                                f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                                f"<b>Pair:</b> <code>{symbol}</code>\n"
                                f"<b>Entry:</b> ${fill_price:.5f}\n"
                                f"<b>Qty:</b> {executed_qty}\n"
                                f"<b>Risk %:</b> {risk_pct_value}%\n"
                                f"<b>SL:</b> ${stop_loss:.5f} ({sl_status}) | <b>TP:</b> ${take_profit:.5f}"
                            )
                        else:
                            logger.error(f"[{symbol}] Upsert returned no trade_id — DB state may be inconsistent.")
                    finally:
                        release_db_connection(conn_db)
            elif exec_result.get("status") == "FAILED" and "SL failed to attach" in str(exec_result.get("error", "")):
                # SL attach failed but emergency close likely succeeded.
                # Do NOT raise — the position is clean; just log and continue.
                logger.warning(
                    f"[{symbol}] Order rejected post-entry: {exec_result.get('error')}. "
                    f"Position expected to be clean."
                )
            else:
                logger.error(f"[{symbol}] Futures Trade Execution failed: {exec_result.get('error')}")

        finally:
            await event_bus.guard.release_trade_lock(symbol)

    async def _handle_ticker_update(self, payload: dict):
        symbol = payload["symbol"]
        price = payload["price"]
        clean_symbol = symbol.replace("/", "").replace("_", "").upper()

        if clean_symbol in event_bus.active_local_sl_guards:
            guard = event_bus.active_local_sl_guards[clean_symbol]
            direction = guard["direction"]
            sl_price = guard["stop_loss"]

            triggered = (direction in ["BUY", "LONG"] and price <= sl_price) or \
                        (direction in ["SELL", "SHORT"] and price >= sl_price)

            if triggered:
                logger.critical(f"[{clean_symbol}] EMERGENCY LOCAL SL TRIGGERED ({direction}) @ ${price:.5f}")
                event_bus.disarm_local_sl_guard(clean_symbol)

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    self.executor.close_live_position_bybit,
                    symbol,
                    guard["quantity"],
                    price,
                    "EMERGENCY_LOCAL_SL"
                )

    async def _handle_execution_report(self, report: dict):
        symbol = report.get("symbol", "").replace("_", "")
        order_status = report.get("orderStatus")
        price = safe_float(report.get("avgPrice"), 0.0)

        if order_status in ["Filled", "Cancelled"]:
            logger.info(f"[{symbol}] Bybit execution report event received: Status={order_status}, Price={price}")