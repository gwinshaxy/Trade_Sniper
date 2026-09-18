import os
import sys

# =====================================================================
# PROXY BYPASS FOR WEBSOCKETS & REST APIs AT APPLICATION ENTRY POINT
# Define explicit domain bypasses BEFORE any HTTP/network clients initialize
# =====================================================================
NO_PROXY_DOMAINS = (
    "api.binance.com,api.mexc.com,api.telegram.org,.supabase.co,"
    "stream.bybit.com,stream-testnet.bybit.com,localhost,127.0.0.1"
)
os.environ["NO_PROXY"] = NO_PROXY_DOMAINS
os.environ["no_proxy"] = NO_PROXY_DOMAINS

# Ensure global proxy environment variables are wiped to prevent ambient proxy leakage
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

from config import BYBIT_API_KEY, BYBIT_SECRET_KEY, BYBIT_TESTNET
from common import (
    get_db_connection,
    release_db_connection,
    calculate_pnl,
    send_telegram_notification,
    check_daily_circuit_breaker,
    ensure_schema_updated,
    finalize_trade_in_db
)
from event_bus import event_bus
from live_executor import LiveExecutionEngine
from reconciler import reconcile_open_trades
from state_machine import StateMachineEngine
from strategy import (
    fetch_klines,
    evaluate_signals,
    load_symbol_config,
    calculate_tema as calc_tema,
    calculate_atr as calc_atr
)
from dynamic_trade_manager import DynamicTradeManager

root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

raw_symbols = os.getenv("TRADING_SYMBOLS") or os.getenv("WATCHLIST") or "XRP/USDT,LINK/USDT,SOL/USDT,BNB/USDT"
WATCHLIST = [s.strip() for s in raw_symbols.split(",") if s.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "1h")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
ACCOUNT_RISK_PCT = float(os.getenv("ACCOUNT_RISK_PCT", "1.0"))
FALLBACK_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "100.0"))
MAX_CONCURRENT_POSITIONS = 4  # Locked max concurrent position threshold

logger = logging.getLogger("main_orchestrator")

executor = LiveExecutionEngine()
state_machine = StateMachineEngine(executor)
trade_manager = DynamicTradeManager()


async def dynamic_trade_management_loop():
    logger.info("Starting Dynamic Position Management & Trailing Stop Loop.")
    while True:
        try:
            conn = await asyncio.to_thread(get_db_connection)
            if not conn:
                await asyncio.sleep(10)
                continue

            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, pair, direction, entry_price, stop_loss, take_profit, trade_state, position_size, account_balance 
                        FROM trade_setups 
                        WHERE status = 'EXECUTED' AND trade_state != 'CLOSED';
                    """)
                    columns = [col[0] for col in cur.description]
                    active_trades = [dict(zip(columns, row)) for row in cur.fetchall()]

                for trade in active_trades:
                    trade_id = trade['id']
                    pair = trade['pair']
                    
                    df_active = await asyncio.to_thread(
                        fetch_klines,
                        symbol=pair,
                        interval="1h",
                        limit=300
                    )
                    
                    if df_active is None or df_active.empty or len(df_active) < 200:
                        logger.warning(f"[{pair}] K-line data insufficient/empty for active trade #{trade_id}. Skipping iteration.")
                        continue

                    cfg = load_symbol_config(pair)
                    df_active['tema'] = calc_tema(df_active['close'], period=int(cfg.get("tema_period", 200)))
                    df_active['atr'] = calc_atr(df_active, period=int(cfg.get("atr_period", 14)))

                    latest_candle = df_active.iloc[-1].to_dict()
                    
                    result = trade_manager.process_trade(trade, latest_candle)
                    action = result.get("action")

                    if action == "UPDATE_SL":
                        new_sl = result["new_sl"]
                        new_state = result["new_state"]
                        msg = result["msg"]

                        with conn.cursor() as cur:
                            cur.execute("""
                                UPDATE trade_setups
                                SET stop_loss = %s, trade_state = %s
                                WHERE id = %s AND status = 'EXECUTED';
                            """, (round(new_sl, 5), new_state, trade_id))
                            conn.commit()

                        send_telegram_notification(msg)

                    elif action in ["CLOSE_SL", "CLOSE_TP"]:
                        exit_price = result["exit_price"]
                        acct_bal = float(trade.get("account_balance") or FALLBACK_BALANCE)
                        pnl_usd, pnl_pct, outcome = calculate_pnl(
                            trade["direction"], 
                            float(trade["entry_price"]), 
                            exit_price, 
                            float(trade.get("position_size", 1.0)), 
                            acct_bal
                        )

                        finalize_trade_in_db(trade_id, exit_price, pnl_usd, pnl_pct, outcome)
                        send_telegram_notification(result.get("msg", f"Trade #{trade_id} closed at {exit_price}"))

            finally:
                release_db_connection(conn)

        except Exception as e:
            logger.error(f"Error in dynamic trade management loop: {e}")

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def strategy_evaluation_loop():
    logger.info("Bybit Futures Strategy Signal Evaluator initialized.")
    while True:
        try:
            live_balance = await asyncio.to_thread(executor.fetch_available_usdt_balance)
            active_usdt_balance = live_balance if live_balance > 1.0 else FALLBACK_BALANCE

            if check_daily_circuit_breaker(max_loss_pct=3.0, account_balance=active_usdt_balance):
                logger.warning("Daily Circuit Breaker Triggered. Pausing strategy evaluations.")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            # --- GLOBAL EQUAL-WEIGHT FRACTIONAL LOCK ---
            conn_global = await asyncio.to_thread(get_db_connection)
            global_active_count = 0
            if conn_global:
                try:
                    with conn_global.cursor() as cur:
                        cur.execute("""
                            SELECT COUNT(*) FROM trade_setups 
                            WHERE status = 'EXECUTED' AND trade_state != 'CLOSED';
                        """)
                        global_active_count = cur.fetchone()[0]
                finally:
                    release_db_connection(conn_global)

            if global_active_count >= MAX_CONCURRENT_POSITIONS:
                logger.info(f"Max concurrent positions reached ({global_active_count}/{MAX_CONCURRENT_POSITIONS}). Skipping trade evaluations.")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Calculate Equal-Weight Allocated Margin per Trade
            allocated_margin_per_trade = active_usdt_balance / MAX_CONCURRENT_POSITIONS

            for symbol in WATCHLIST:
                conn = await asyncio.to_thread(get_db_connection)
                if conn:
                    try:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT COUNT(*) FROM trade_setups 
                                WHERE pair = %s AND status = 'EXECUTED' AND trade_state != 'CLOSED';
                            """, (symbol,))
                            active_count = cur.fetchone()[0]

                        if active_count > 0:
                            logger.info(f"[{symbol}] Active trade currently open in DB. Skipping signal evaluation.")
                            continue
                    except Exception as db_err:
                        logger.error(f"[{symbol}] Error checking active trade count: {db_err}")
                    finally:
                        release_db_connection(conn)

                df_klines = await asyncio.to_thread(
                    fetch_klines,
                    symbol=symbol,
                    interval=TIMEFRAME,
                    limit=300
                )
                
                if df_klines is None or df_klines.empty:
                    logger.warning(f"[{symbol}] Kline data empty. Skipping evaluation.")
                    continue

                cfg = load_symbol_config(symbol)
                signal = evaluate_signals(
                    df=df_klines,
                    symbol=symbol,
                    account_balance=active_usdt_balance,
                    risk_pct=cfg.get("risk_pct", ACCOUNT_RISK_PCT),
                    tema_period=cfg.get("tema_period", 200),
                    rsi_period=cfg.get("rsi_period", 14),
                    rsi_thresh=cfg.get("rsi_thresh", 42.0),
                    adx_period=cfg.get("adx_period", 14),
                    adx_threshold=cfg.get("adx_threshold", 20.0),
                    use_adx_filter=cfg.get("use_adx_filter", True),
                    use_rsi_filter=cfg.get("use_rsi_filter", True),
                    use_candlestick_confirm=cfg.get("use_candlestick_confirm", True),
                    zone_tolerance=cfg.get("zone_tolerance", 0.0075),
                    max_sl_pct=cfg.get("max_sl_pct", 0.02),
                    min_sentiment=cfg.get("min_sentiment", 0.0),
                    min_rr=cfg.get("min_rr", 2.0),
                    atr_period=cfg.get("atr_period", 14),
                    atr_mult=cfg.get("atr_mult", 2.0),
                    use_atr_sl=cfg.get("use_atr_sl", True),
                    disable_htf=cfg.get("disable_htf", False),
                    sentiment_score=0.5
                )

                action = signal.get("action", "HOLD")
                reason = signal.get("reason", "No reason provided")

                logger.info(
                    f"[{symbol}] Live Close: {df_klines['close'].iloc[-1]:.4f} | "
                    f"Action: {action} | Reason: {reason}"
                )

                if action in ["BUY", "LONG", "SELL", "SHORT"]:
                    await event_bus.publish("TRADE_SIGNAL", {
                        "pair": symbol,
                        "symbol": symbol,
                        "direction": action,
                        "entry_price": float(signal.get("entry_price", 0.0)),
                        "stop_loss": float(signal.get("stop_loss", 0.0)),
                        "take_profit": float(signal.get("take_profit", 0.0)),
                        "amount_usd": allocated_margin_per_trade,
                        "account_balance": active_usdt_balance,
                        "leverage": cfg.get("leverage", 10)
                    })

        except Exception as e:
            logger.error(f"Error in strategy loop: {e}")

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def reconciler_background_task(executor):
    """
    Event-driven position reconciliation loop.
    Triggers instantly on WS execution events or runs every 1 hour (3600s) as a safety net.
    """
    exec_queue = asyncio.Queue()

    async def on_execution_event(payload):
        await exec_queue.put(payload)

    # Subscribe queue listener to execution events
    event_bus.subscribe("EXECUTION_EVENT", on_execution_event)

    # Fallback polling interval in seconds (1 hour)
    SAFETY_INTERVAL = 3600

    while True:
        try:
            # Wait for either an incoming WS execution event or the safety timer expiry
            try:
                event_payload = await asyncio.wait_for(exec_queue.get(), timeout=SAFETY_INTERVAL)
                logger.info(f"⚡ [Event-Driven] Triggering reconciliation via WS execution event ({event_payload.get('symbol')}).")
            except asyncio.TimeoutError:
                logger.info("⏰ [Safety Check] Running scheduled background position reconciliation...")

            # Execute position reconciliation in a thread to keep async loop non-blocking
            reconciled_count = await asyncio.to_thread(reconcile_open_trades, executor)
            if reconciled_count > 0:
                logger.info(f"Reconciliation finished: {reconciled_count} record(s) healed/updated.")

        except Exception as rec_err:
            logger.error(f"Reconciler task error: {rec_err}")
            await asyncio.sleep(10)  # Brief delay before retrying on loop errors


async def main():
    logger.info("Starting Centralized Bybit Futures Event-Driven Architecture...")
    
    ensure_schema_updated()

    from ws_engine import UnifiedWebSocketEngine

    ws_engine = UnifiedWebSocketEngine(
        symbols=WATCHLIST,
        is_testnet=BYBIT_TESTNET,
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_SECRET_KEY
    )

    await asyncio.gather(
        ws_engine.start(),
        state_machine.run(),
        strategy_evaluation_loop(),
        dynamic_trade_management_loop(),
        reconciler_background_task(executor)
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bybit Futures Trading Agent system shut down cleanly.")
        sys.exit(0)