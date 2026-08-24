import os
import time
import logging
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

from common import (
    ensure_schema_updated,
    get_db_connection,
    release_db_connection,
    check_daily_circuit_breaker,
    send_telegram_notification,
)
from strategy import fetch_klines, evaluate_signals
from trade_manager import TradeManager

load_dotenv()

# Dynamic Symbol Loading from .env (e.g., TRADING_SYMBOLS="XRP/USDT,BTC/USDT")
raw_symbols = os.getenv("TRADING_SYMBOLS") or os.getenv("WATCHLIST") or "XRP/USDT"
WATCHLIST = [s.strip() for s in raw_symbols.split(",") if s.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "1h")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
ACCOUNT_RISK_PCT = float(os.getenv("ACCOUNT_RISK_PCT", "1.0"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main_engine")


def process_symbol(symbol: str, tm: TradeManager):
    """Fetches data, evaluates strategy, checks open trades, and processes new signals for a single symbol."""
    logger.info(f"--- Processing {symbol} [{TIMEFRAME}] (PAPER SPOT MODE) ---")

    # 1. Fetch Latest Kline Data
    df = fetch_klines(symbol=symbol, timeframe=TIMEFRAME, limit=300)
    if df is None or df.empty:
        logger.warning(f"Could not retrieve kline data for {symbol}. Skipping cycle.")
        return

    latest_candle = df.iloc[-1]
    current_price = float(latest_candle["close"])

    # 2. Process existing open trades for this pair in the Database
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, pair, direction, entry_price, stop_loss, take_profit, position_size, account_balance, trade_state
                FROM trade_setups
                WHERE pair = %s AND trade_state = 'OPEN';
            """,
                (symbol,),
            )
            open_trades = cursor.fetchall()
            cursor.close()

            for trade_row in open_trades:
                trade_series = pd.Series(
                    {
                        "id": trade_row[0],
                        "pair": trade_row[1],
                        "direction": trade_row[2],
                        "entry_price": float(trade_row[3]),
                        "stop_loss": float(trade_row[4]) if trade_row[4] else None,
                        "take_profit": (
                            float(trade_row[5]) if trade_row[5] else None
                        ),
                        "position_size": float(trade_row[6]),
                        "account_balance": float(trade_row[7] or 100.0),
                        "trade_state": trade_row[8],
                    }
                )

                # Process stop-loss / take-profit conditions via TradeManager
                action_result = tm.process_trade(trade_series, latest_candle)
                if action_result.get("action") in ["CLOSE_SL", "CLOSE_TP"]:
                    logger.info(
                        f"Trade #{trade_series['id']} closed via {action_result['action']} at ${current_price:.5f}"
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
        # In Spot Mode, SELL/SHORT signals are skipped for fresh entries
        if signal == "SELL":
            logger.info(f"⚡ SHORT SIGNAL DETECTED for {symbol}: Bypassed (Spot Mode Active).")
            return

        logger.info(f"⚡ BUY SIGNAL DETECTED: {signal} on {symbol}")

        # Ensure no existing open trade for this pair before inserting
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM trade_setups WHERE pair = %s AND trade_state = 'OPEN';",
                    (symbol,),
                )
                existing = cursor.fetchone()

                if not existing:
                    entry_price = details["entry_price"]
                    stop_loss = details["stop_loss"]
                    take_profit = details["take_profit"]
                    position_size = details["position_size"]
                    account_balance = details.get("account_balance", 100.0)

                    insert_query = """
                        INSERT INTO trade_setups (pair, direction, entry_price, stop_loss, take_profit, risk_pct, position_size, account_balance, status, trade_state)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN')
                        RETURNING id;
                    """
                    cursor.execute(
                        insert_query,
                        (
                            symbol,
                            "LONG",
                            entry_price,
                            stop_loss,
                            take_profit,
                            ACCOUNT_RISK_PCT,
                            position_size,
                            account_balance,
                        ),
                    )
                    trade_id = cursor.fetchone()[0]
                    conn.commit()

                    logger.info(
                        f"SUCCESS: Executed and recorded Paper Trade #{trade_id} [BUY {symbol}]"
                    )

                    send_telegram_notification(
                        f"<b>🚀 NEW PAPER TRADE RECORDED (SPOT)</b>\n\n"
                        f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                        f"<b>Pair:</b> <code>{symbol}</code>\n"
                        f"<b>Direction:</b> <code>LONG (BUY)</code>\n"
                        f"<b>Entry:</b> ${entry_price:.5f}\n"
                        f"<b>Stop Loss:</b> ${stop_loss:.5f}\n"
                        f"<b>Take Profit:</b> ${take_profit:.5f}\n"
                        f"<b>Size:</b> {position_size:.2f} units"
                    )
                else:
                    logger.info(
                        f"Skipping trade execution: Trade #{existing[0]} is already OPEN for {symbol}."
                    )

                cursor.close()
            except Exception as e:
                logger.error(f"Failed to process trade signal in DB: {e}")
            finally:
                release_db_connection(conn)
    else:
        logger.info(f"No entry signal for {symbol} ({details})")


def main():
    logger.info(f"Starting Paper Trading Bot Engine (Spot Mode) | Active Watchlist: {WATCHLIST}...")
    ensure_schema_updated()
    tm = TradeManager()

    send_telegram_notification(
        f"<b>🟢 BOT STARTED (PAPER SPOT MODE)</b>\nMonitoring pairs: <code>{', '.join(WATCHLIST)}</code>"
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