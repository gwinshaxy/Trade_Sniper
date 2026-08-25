import os
import time
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
from dotenv import load_dotenv

from common import (
    ensure_schema_updated,
    get_db_connection,
    release_db_connection,
    check_daily_circuit_breaker,
    send_telegram_notification,
)
from live_executor import LiveExecutionEngine
from strategy import fetch_klines, evaluate_signals, safe_float
from trade_manager import TradeManager

load_dotenv()

raw_symbols = os.getenv("TRADING_SYMBOLS") or os.getenv("WATCHLIST") or "XRP/USDT"
WATCHLIST = [s.strip() for s in raw_symbols.split(",") if s.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "1h")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
ACCOUNT_RISK_PCT = float(os.getenv("ACCOUNT_RISK_PCT", "1.0"))
PORT = int(os.getenv("PORT", 10000))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main_engine")

execution_engine = LiveExecutionEngine()

class HealthCheckHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP Handler for Render port health checks."""
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "healthy", "service": "Trading Bot Engine"}')

    def log_message(self, format, *args):
        return

def run_health_server():
    """Runs a built-in lightweight HTTP health check server on $PORT in a background thread."""
    try:
        server = HTTPServer(("0.0.0.0", PORT), HealthCheckHandler)
        logger.info(f"Starting native background Health Check server on port {PORT}...")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start native Health Check HTTP server: {e}")

def process_symbol(symbol: str, tm: TradeManager):
    """Fetches data, evaluates strategy, checks open trades, and executes live spot buy orders."""
    logger.info(f"--- Processing {symbol} [{TIMEFRAME}] (LIVE SPOT MODE) ---")

    # 1. Fetch Latest Kline Data
    df = fetch_klines(symbol=symbol, timeframe=TIMEFRAME, limit=300)
    if df is None or df.empty:
        logger.warning(f"Could not retrieve kline data for {symbol}. Skipping cycle.")
        return

    latest_candle = df.iloc[-1]
    current_price = safe_float(latest_candle["close"])

    # 2. Process existing open trades for this pair in the Database
    db_symbol = symbol.replace("/", "").replace("-", "").strip().upper()

    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, pair, direction, entry_price, stop_loss, take_profit, position_size, account_balance, trade_state
                FROM trade_setups
                WHERE (pair = %s OR pair = %s) AND trade_state = 'OPEN';
                """,
                (symbol, db_symbol),
            )
            open_trades = cursor.fetchall()
            cursor.close()

            for trade_row in open_trades:
                trade_series = pd.Series(
                    {
                        "id": trade_row[0],
                        "pair": trade_row[1],
                        "direction": trade_row[2],
                        "entry_price": safe_float(trade_row[3]),
                        "stop_loss": safe_float(trade_row[4]) if trade_row[4] else None,
                        "take_profit": safe_float(trade_row[5]) if trade_row[5] else None,
                        "position_size": safe_float(trade_row[6]),
                        "account_balance": safe_float(trade_row[7], default=100.0),
                        "trade_state": trade_row[8],
                    }
                )

                action_result = tm.process_trade(trade_series, latest_candle)
                if action_result.get("action") in ["CLOSE_SL", "CLOSE_TP"]:
                    logger.info(
                        f"Trade #{trade_series['id']} condition met via {action_result['action']} at ${current_price:.5f}"
                    )
        except Exception as e:
            logger.error(f"Error querying/updating open trades for {symbol}: {e}")
        finally:
            release_db_connection(conn)

    # 3. Check Daily Circuit Breaker before opening new trades
    if check_daily_circuit_breaker(max_loss_pct=3.0, account_balance=100.0):
        logger.warning(
            f"Daily Circuit Breaker triggered (Max daily loss limit hit). Skipping new signal checks for {symbol}."
        )
        return

    # 4. Evaluate Strategy Signals for New Trade Entries
    signal, details = evaluate_signals(
        df, symbol=symbol, risk_pct=ACCOUNT_RISK_PCT
    )

    if signal in ["BUY", "SELL"] and isinstance(details, dict):
        if signal == "SELL":
            logger.info(f"⚡ SHORT SIGNAL DETECTED for {symbol}: Bypassed (Spot Mode Active).")
            return

        logger.info(f"⚡ LIVE BUY SIGNAL DETECTED: {signal} on {symbol}")

        # Ensure no existing open trade for this pair before placing live buy order
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM trade_setups WHERE (pair = %s OR pair = %s) AND trade_state = 'OPEN';",
                    (symbol, db_symbol),
                )
                existing = cursor.fetchone()

                if not existing:
                    target_stop_loss = details["stop_loss"]
                    target_take_profit = details["take_profit"]
                    position_size = details["position_size"]
                    account_balance = details.get("account_balance", 100.0)

                    amount_usd = details.get("position_size_usd") or (position_size * safe_float(details["entry_price"]))

                    # Execute Live Spot Buy Order
                    order_res = execution_engine.buy_spot_mexc(
                        symbol=symbol,
                        amount_usd=amount_usd
                    )

                    if order_res and (order_res.get("status") == "SUCCESS" or order_res.get("id")):
                        actual_entry = safe_float(order_res.get("fill_price") or order_res.get("price"), default=details["entry_price"])
                        actual_qty = safe_float(order_res.get("executed_qty") or order_res.get("amount"), default=position_size)

                        insert_query = """
                            INSERT INTO trade_setups (pair, direction, entry_price, stop_loss, take_profit, risk_pct, position_size, account_balance, status, trade_state)
                            VALUES (%s, 'LONG', %s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN')
                            RETURNING id;
                        """
                        cursor.execute(
                            insert_query,
                            (
                                symbol,
                                actual_entry,
                                target_stop_loss,
                                target_take_profit,
                                ACCOUNT_RISK_PCT,
                                actual_qty,
                                account_balance,
                            ),
                        )
                        trade_id = cursor.fetchone()[0]
                        conn.commit()

                        logger.info(
                            f"SUCCESS: Live Spot Buy Order Executed & Trade #{trade_id} Recorded [{symbol}]"
                        )

                        send_telegram_notification(
                            f"<b>🚀 NEW LIVE SPOT TRADE EXECUTED</b>\n\n"
                            f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                            f"<b>Pair:</b> <code>{symbol}</code>\n"
                            f"<b>Direction:</b> <code>LONG (BUY SPOT)</code>\n"
                            f"<b>Fill Price:</b> ${actual_entry:.5f}\n"
                            f"<b>Stop Loss:</b> ${target_stop_loss:.5f}\n"
                            f"<b>Take Profit:</b> ${target_take_profit:.5f}\n"
                            f"<b>Quantity:</b> {actual_qty:.4f} units"
                        )
                    else:
                        logger.error(f"Live Spot Order Execution failed for {symbol}: {order_res}")

                else:
                    logger.info(
                        f"Skipping trade execution: Trade #{existing[0]} is already OPEN for {symbol}."
                    )

                cursor.close()
            except Exception as e:
                logger.error(f"Failed to process live trade execution in DB: {e}")
            finally:
                release_db_connection(conn)
    else:
        logger.info(f"No entry signal for {symbol} ({details})")

def main():
    logger.info(f"Starting Live Trading Bot Engine (Spot Mode) | Active Watchlist: {WATCHLIST}...")
    
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    ensure_schema_updated()
    tm = TradeManager()

    send_telegram_notification(
        f"<b>🟢 BOT STARTED (LIVE SPOT MODE)</b>\nMonitoring pairs: <code>{', '.join(WATCHLIST)}</code>"
    )

    while True:
        try:
            for symbol in WATCHLIST:
                process_symbol(symbol, tm)
            logger.info(
                f"Cycle complete. Waiting {POLL_INTERVAL_SECONDS}s for next check..."
            )
            time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("Bot execution stopped manually by user.")
            send_telegram_notification(
                "<b>🔴 BOT STOPPED</b>\nExecution loop interrupted manually."
            )
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            time.sleep(15)

if __name__ == "__main__":
    main()