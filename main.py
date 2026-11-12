import os
import sys

NO_PROXY_DOMAINS = (
    "://binance.com,://mexc.com,api.telegram.org,.supabase.co,"
    "://bybit.com,://bybit.com,localhost,127.0.0.1"
)
os.environ["NO_PROXY"] = NO_PROXY_DOMAINS
os.environ["no_proxy"] = NO_PROXY_DOMAINS

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

import asyncio
import logging
import json
from dotenv import load_dotenv

load_dotenv()

from config import BYBIT_API_KEY, BYBIT_SECRET_KEY, BYBIT_TESTNET, STRATEGY_CONFIG
from common import (
    get_db_connection,
    release_db_connection,
    calculate_pnl,
    send_telegram_notification,
    check_daily_circuit_breaker,
    ensure_schema_updated,
    finalize_trade_in_db,
    check_asset_cooldown
)
from event_bus import event_bus
from live_executor import BybitFuturesLiveExecutor as LiveExecutionEngine
from reconciler import reconcile_open_trades
from state_machine import StateMachineEngine
from strategy import (
    evaluate_signals,
    load_symbol_config,
    fetch_klines,
    EXECUTION_TIMEFRAME,
    ATR_TRAIL_MULTIPLIER,
    TRAIL_ACTIVATION_R,
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

logger = logging.getLogger("main_orchestrator")


def load_vetted_watchlist(json_file: str = "vetted_symbols.json") -> list:
    """Automatically loads vetted symbols from dynamic asset filter JSON output."""
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                symbols = data.get("vetted_symbols", [])
                if symbols:
                    return symbols
        except Exception as e:
            logger.warning(f"Failed to load vetted symbols from {json_file}: {e}")

    logger.warning("vetted_symbols.json not found or empty. Falling back to environment watchlist.")
    raw_symbols = (
        os.getenv("TRADING_SYMBOLS")
        or os.getenv("WATCHLIST")
        or "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,ADA/USDT,DOGE/USDT,BNB/USDT,LINK/USDT,NEAR/USDT,HYPE/USDT"
    )
    return [s.strip() for s in raw_symbols.split(",") if s.strip()]


TIMEFRAME = os.getenv("TIMEFRAME", EXECUTION_TIMEFRAME)
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
ACCOUNT_RISK_PCT = float(os.getenv("ACCOUNT_RISK_PCT", "1.0"))
FALLBACK_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "100.0"))
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "5"))

executor = LiveExecutionEngine()
state_machine = StateMachineEngine(executor)

enable_dyn_manager = STRATEGY_CONFIG.get("enable_dynamic_trade_manager", True)
trade_manager = DynamicTradeManager(
    trailing_mult=float(STRATEGY_CONFIG.get("atr_trail_multiplier", ATR_TRAIL_MULTIPLIER)),
    trail_activation_r=float(STRATEGY_CONFIG.get("trail_activation_r", TRAIL_ACTIVATION_R)),
    disable_static_tp=True,
    enabled=enable_dyn_manager,
    executor=executor
)


async def dynamic_trade_management_loop():
    logger.info(
        f"Starting Dynamic Chandelier Trailing Stop Loop "
        f"(Timeframe: {TIMEFRAME} | Activation: {trade_manager.trail_activation_r}R | Trail: {trade_manager.trailing_mult}x ATR)."
    )
    while True:
        try:
            conn = await asyncio.to_thread(get_db_connection)
            if not conn:
                await asyncio.sleep(10)
                continue

            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, pair, direction, entry_price, stop_loss, initial_stop_loss,
                               take_profit, trade_state, position_size, account_balance,
                               highest_price, lowest_price
                        FROM trade_setups 
                        WHERE status = 'EXECUTED' AND trade_state != 'CLOSED';
                    """)
                    columns = [col[0] for col in cur.description]
                    active_trades = [dict(zip(columns, row)) for row in cur.fetchall()]

                for trade in active_trades:
                    trade_id = trade['id']
                    pair = trade['pair']
                    pos_qty = float(trade.get("position_size", 0.0))

                    if trade.get("initial_stop_loss") is None and trade.get("stop_loss") is not None:
                        trade["initial_stop_loss"] = trade["stop_loss"]
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE trade_setups SET initial_stop_loss = %s WHERE id = %s;",
                                (trade["stop_loss"], trade_id)
                            )
                            conn.commit()

                    df_active = await asyncio.to_thread(fetch_klines, symbol=pair, interval=TIMEFRAME, limit=250)

                    if df_active is None or df_active.empty or len(df_active) < 50:
                        logger.warning(f"[{pair}] K-line data insufficient/empty for active trade #{trade_id}. Skipping iteration.")
                        continue

                    cfg = load_symbol_config(pair)
                    df_active['atr'] = calc_atr(df_active, period=int(cfg.get("atr_period", 14)))

                    latest_candle = df_active.iloc[-1].to_dict()
                    live_pos = await executor.get_futures_position_async(pair)

                    try:
                        result = trade_manager.process_trade(trade, latest_candle, live_pos_info=live_pos)
                    except Exception as pm_err:
                        logger.error(f"[{pair}] Error inside trade_manager.process_trade for #{trade_id}: {pm_err}")
                        result = {"action": "HOLD"}

                    action = result.get("action", "HOLD")

                    if action == "UPDATE_EXTREME":
                        new_high = result.get("highest_price")
                        new_low = result.get("lowest_price")
                        with conn.cursor() as cur:
                            cur.execute("""
                                UPDATE trade_setups
                                SET highest_price = COALESCE(%s, highest_price),
                                    lowest_price = COALESCE(%s, lowest_price),
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE id = %s AND status = 'EXECUTED';
                            """, (new_high, new_low, trade_id))
                            conn.commit()

                    elif action == "UPDATE_SL":
                        new_sl = result.get("new_sl")
                        if new_sl is not None and new_sl > 0:
                            new_state = result.get("new_state", "TRAILING")
                            new_high = result.get("highest_price")
                            new_low = result.get("lowest_price")

                            sl_updated = await asyncio.to_thread(
                                executor.set_position_trading_stop,
                                pair,
                                new_sl,
                                0,
                                None,
                                trade["direction"]
                            )

                            with conn.cursor() as cur:
                                cur.execute("""
                                    UPDATE trade_setups
                                    SET stop_loss = %s,
                                        trailing_stop_price = %s,
                                        highest_price = COALESCE(%s, highest_price),
                                        lowest_price = COALESCE(%s, lowest_price),
                                        trade_state = %s,
                                        updated_at = CURRENT_TIMESTAMP
                                    WHERE id = %s AND status = 'EXECUTED';
                                """, (round(new_sl, 6), round(new_sl, 6), new_high, new_low, new_state, trade_id))
                                conn.commit()

                            msg = result.get("msg", f"[{pair}] Chandelier SL updated to ${new_sl:.5f}")
                            if sl_updated:
                                msg += " (Bybit Position SL Updated)"
                            send_telegram_notification(msg)

                    elif action == "FINALIZED_CLOSED_TRADE":
                        msg = result.get("msg", "")
                        if msg:
                            send_telegram_notification(msg)
                        continue

                    elif action == "SYNC_CLOSED_FROM_EXCHANGE":
                        logger.info(
                            f"[{pair}] Trade #{trade_id} closed on exchange. "
                            f"Direct finalization failed; notifying reconciler."
                        )
                        await event_bus.publish("EXECUTION_EVENT", {
                            "symbol": pair, "order_status": "Filled", "exec_type": "Trade"
                        })
                        continue

                    elif action in ["EXECUTE_CLOSE_SL", "EXECUTE_CLOSE_TP"]:
                        target_exit = result.get("target_price", float(latest_candle.get("close")))
                        close_res = executor.close_live_position_bybit(
                            symbol=pair, position_size=pos_qty,
                            current_price=target_exit, outcome=action
                        )
                        actual_exit_price = float(close_res.get("exit_price", target_exit))
                        real_closed_pnl, real_fee, verified_exit = executor.fetch_real_closed_pnl(pair)
                        if verified_exit > 0:
                            actual_exit_price = verified_exit
                        est_fees = real_fee if real_fee > 0 else (
                            (float(trade["entry_price"]) * pos_qty * 0.00055) +
                            (actual_exit_price * pos_qty * 0.00055)
                        )
                        acct_bal = float(trade.get("account_balance") or FALLBACK_BALANCE)
                        pnl_usd, pnl_pct, outcome = calculate_pnl(
                            trade["direction"], float(trade["entry_price"]),
                            actual_exit_price, pos_qty, acct_bal,
                            total_fees=est_fees,
                            exchange_closed_pnl=real_closed_pnl,
                        )
                        finalize_trade_in_db(trade_id, actual_exit_price, pnl_usd, pnl_pct, outcome, fee_usd=est_fees)
                        send_telegram_notification(
                            f"🔴 POSITION CLOSED ({action})\n\n"
                            f"Trade ID: #{trade_id}\n"
                            f"Symbol: {pair}\n"
                            f"Actual Fill Exit: ${actual_exit_price:.5f}\n"
                            f"Net PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%) [Fees: ${est_fees:.2f}]"
                        )
            finally:
                release_db_connection(conn)
        except Exception as e:
            logger.error(f"Error in dynamic trade management loop: {e}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def strategy_evaluation_loop():
    logger.info(f"Bybit Futures 4H Donchian Breakout Evaluator initialized (Timeframe: {TIMEFRAME}).")
    while True:
        try:
            watchlist = load_vetted_watchlist()
            live_balance = await asyncio.to_thread(executor.fetch_available_usdt_balance)
            active_usdt_balance = live_balance if live_balance > 1.0 else FALLBACK_BALANCE
            if check_daily_circuit_breaker(max_loss_pct=3.0, account_balance=active_usdt_balance):
                logger.warning("Daily Circuit Breaker Triggered. Pausing strategy evaluations.")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue
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
                if getattr(strategy_evaluation_loop, "_last_log_state", None) != "MAX_REACHED":
                    logger.info(f"Max concurrent positions reached ({global_active_count}/{MAX_CONCURRENT_POSITIONS}). Skipping trade evaluations.")
                    strategy_evaluation_loop._last_log_state = "MAX_REACHED"
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue
            else:
                strategy_evaluation_loop._last_log_state = "OK"
            allocated_margin_per_trade = active_usdt_balance / MAX_CONCURRENT_POSITIONS
            for symbol in watchlist:
                if check_asset_cooldown(symbol):
                    logger.info(f"[{symbol}] Asset is in active cooldown. Skipping evaluation.")
                    continue
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
                df_klines = await asyncio.to_thread(fetch_klines, symbol=symbol, interval=TIMEFRAME, limit=300)
                if df_klines is None or df_klines.empty:
                    logger.warning(f"[{symbol}] Kline data empty. Skipping evaluation.")
                    continue
                closed_candle_df = df_klines.iloc[:-1]
                cfg = load_symbol_config(symbol)
                signal = evaluate_signals(
                    df=closed_candle_df,
                    symbol=symbol,
                    account_balance=active_usdt_balance,
                    risk_pct=cfg.get("risk_pct", ACCOUNT_RISK_PCT),
                    use_ema_200_filter=cfg.get("use_ema_200_filter", True),
                    custom_atr_mult=cfg.get("atr_sl_multiplier", 2.5)
                )
                action = signal.get("action", "HOLD")
                reason = signal.get("reason", "No reason provided")
                logger.info(
                    f"[{symbol}] Live Close: {df_klines['close'].iloc[-1]:.4f} | "
                    f"Action: {action} | Reason: {reason}"
                )
                if action in ["BUY", "LONG", "SELL", "SHORT"]:
                    raw_sl = signal.get("stop_loss") or signal.get("sl_price") or 0.0
                    raw_tp = 0.0
                    raw_symbol = signal.get("pair") or signal.get("symbol") or symbol
                    await event_bus.publish("TRADE_SIGNAL", {
                        "pair": raw_symbol,
                        "symbol": raw_symbol,
                        "direction": action,
                        "side": action,
                        "entry_price": float(signal.get("entry_price", 0.0)),
                        "stop_loss": float(raw_sl),
                        "sl_price": float(raw_sl),
                        "take_profit": raw_tp,
                        "tp_price": raw_tp,
                        "trailing_atr_mult": float(signal.get("trailing_atr_mult", ATR_TRAIL_MULTIPLIER)),
                        "trail_activation_r": float(TRAIL_ACTIVATION_R),
                        "amount_usd": allocated_margin_per_trade,
                        "account_balance": active_usdt_balance,
                        "leverage": cfg.get("leverage", 5),
                        "risk_pct": float(cfg.get("risk_pct", ACCOUNT_RISK_PCT)),
                    })
        except Exception as e:
            logger.error(f"Error in strategy loop: {e}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def reconciler_background_task(executor):
    exec_queue = asyncio.Queue()
    async def on_execution_event(payload):
        await exec_queue.put(payload)
    event_bus.subscribe("EXECUTION_EVENT", on_execution_event)
    SAFETY_INTERVAL = 300
    while True:
        try:
            try:
                event_payload = await asyncio.wait_for(exec_queue.get(), timeout=SAFETY_INTERVAL)
                logger.info(f"⚡ [Event-Driven] Triggering reconciliation via WS execution event ({event_payload.get('symbol')}).")
            except asyncio.TimeoutError:
                logger.info("⏰ [Safety Check] Running scheduled background position reconciliation...")
            reconciled_count = await asyncio.to_thread(reconcile_open_trades, executor)
            if reconciled_count > 0:
                logger.info(f"Reconciliation finished: {reconciled_count} record(s) healed/updated.")
        except Exception as rec_err:
            logger.error(f"Reconciler task error: {rec_err}")
        await asyncio.sleep(10)


async def main():
    logger.info("Starting Centralized Bybit Futures Event-Driven Architecture (4H Breakout + Chandelier Trail)...")
    ensure_schema_updated()
    from ws_engine import UnifiedWebSocketEngine
    initial_watchlist = load_vetted_watchlist()
    ws_engine = UnifiedWebSocketEngine(
        symbols=initial_watchlist,
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