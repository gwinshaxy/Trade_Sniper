import time
import logging
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from strategy import fetch_klines, evaluate_signals, load_symbol_config, calc_tema, calc_atr
from agent import DynamicTradeManager
from common import get_db_connection, ensure_schema_updated, send_telegram_notification, calculate_pnl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ensure_schema_updated()

symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "ETH/USDT,BNB/USDT,SOL/USDT")
SYMBOLS = [s.strip().upper() for s in symbols_env.split(",")]
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", 60))
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", 10000.0))
DEFAULT_RISK_PCT = float(os.getenv("RISK_PCT", 1.0))


def check_equity_curve_filter(conn) -> float:
    """RECOMMENDATION 2: Equity Curve Filter
    Checks the last 10 closed trades PnL from PostgreSQL.
    If current average PnL is below its 10-trade moving average, returns a 0.3 risk scaling factor.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pnl_usd FROM trade_setups 
                WHERE status = 'CLOSED' 
                ORDER BY closed_at DESC LIMIT 10;
            """)
            rows = cur.fetchall()
            if not rows or len(rows) < 5:
                return 1.0

            pnl_list = [float(r[0]) for r in rows if r[0] is not None]
            if len(pnl_list) < 5:
                return 1.0

            recent_pnl = pnl_list[0]
            ma_10_pnl = sum(pnl_list) / len(pnl_list)

            if recent_pnl < ma_10_pnl:
                logging.info(f"[Equity Filter Active] Recent PnL (${recent_pnl:.2f}) < 10-MA (${ma_10_pnl:.2f}). Scaling risk to 30%.")
                return 0.3
    except Exception as e:
        logging.error(f"Equity curve filter query failed: {e}")
    return 1.0


def run_worker_loop():
    trade_manager = DynamicTradeManager()
    logging.info(f"Worker process active. Tracking symbols: {SYMBOLS}")

    while True:
        conn = None
        try:
            conn = get_db_connection()
            if conn:
                # --- PHASE 1: EVALUATE NEW TRADING SIGNALS ---
                risk_modifier = check_equity_curve_filter(conn)

                for symbol in SYMBOLS:
                    try:
                        df = fetch_klines(symbol=symbol, interval="1h", limit=300)
                        if not df.empty:
                            signal = evaluate_signals(df, symbol=symbol, account_balance=ACCOUNT_BALANCE)
                            if signal.get("action") in ["BUY", "SELL"]:
                                pair = signal["symbol"]
                                direction = signal["direction"]
                                entry = signal["entry_price"]
                                sl = signal["stop_loss"]
                                tp = signal["take_profit"]
                                rr = signal["risk_reward_ratio"]
                                
                                base_risk_pct = signal.get("risk_pct", DEFAULT_RISK_PCT)
                                adjusted_risk_pct = round(base_risk_pct * risk_modifier, 2)
                                risk_amt = ACCOUNT_BALANCE * (adjusted_risk_pct / 100.0)
                                risk_dist = abs(entry - sl)
                                pos_size = round(risk_amt / risk_dist, 4) if risk_dist > 0 else signal["position_size"]

                                with conn.cursor() as cur:
                                    cur.execute(
                                        "SELECT id FROM trade_setups WHERE pair = %s AND status = 'EXECUTED' AND trade_state != 'CLOSED';",
                                        (pair,)
                                    )
                                    existing = cur.fetchone()

                                    if not existing:
                                        cur.execute(
                                            """
                                            INSERT INTO trade_setups 
                                            (pair, direction, entry_price, stop_loss, take_profit, risk_reward_ratio, risk_pct, position_size, account_balance, status, trade_state)
                                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN');
                                            """,
                                            (pair, direction, entry, sl, tp, rr, adjusted_risk_pct, pos_size, ACCOUNT_BALANCE)
                                        )
                                        conn.commit()
                                        send_telegram_notification(
                                            f"<b>🚀 AUTOMATED TRADE EXECUTED</b>\n\n"
                                            f"<b>Pair:</b> <code>{pair}</code>\n"
                                            f"<b>Direction:</b> <code>{direction}</code>\n"
                                            f"<b>Entry:</b> ${entry:.5f}\n"
                                            f"<b>SL:</b> ${sl:.5f} | <b>TP:</b> ${tp:.5f}\n"
                                            f"<b>Risk:</b> {adjusted_risk_pct}% | <b>R:R:</b> 1:{rr}"
                                        )
                    except Exception as sig_err:
                        logging.error(f"Error evaluating signals for {symbol}: {sig_err}")

                # --- PHASE 2: DYNAMIC POSITION MANAGEMENT LOOP (COMPLETED) ---
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT id, pair, direction, entry_price, stop_loss, take_profit, position_size, account_balance, trade_state
                            FROM trade_setups
                            WHERE status = 'EXECUTED' AND trade_state != 'CLOSED';
                        """)
                        columns = [desc[0] for desc in cur.description]
                        active_rows = cur.fetchall()

                    for row in active_rows:
                        trade_data = dict(zip(columns, row))
                        symbol = trade_data["pair"]

                        df_latest = fetch_klines(symbol=symbol, interval="1h", limit=250)
                        if df_latest.empty:
                            continue

                        df_latest["tema"] = calc_tema(df_latest["close"], period=200)
                        df_latest["atr"] = calc_atr(df_latest, period=14)
                        latest_candle = df_latest.iloc[-1]

                        mgmt_result = trade_manager.process_trade(trade_data, latest_candle)
                        action = mgmt_result.get("action")

                        if action in ["CLOSE_SL", "CLOSE_TP"]:
                            exit_price = float(mgmt_result["exit_price"])
                            pnl_usd, pnl_pct, outcome = calculate_pnl(
                                trade_data["direction"],
                                float(trade_data["entry_price"]),
                                exit_price,
                                float(trade_data["position_size"]),
                                float(trade_data["account_balance"] or ACCOUNT_BALANCE)
                            )
                            with conn.cursor() as cur:
                                cur.execute("""
                                    UPDATE trade_setups
                                    SET exit_price = %s, pnl_usd = %s, pnl_pct = %s, outcome = %s,
                                        status = 'CLOSED', trade_state = 'CLOSED', closed_at = CURRENT_TIMESTAMP
                                    WHERE id = %s;
                                """, (exit_price, pnl_usd, pnl_pct, outcome, trade_data["id"]))
                                conn.commit()

                            send_telegram_notification(mgmt_result["msg"])

                        elif action == "UPDATE_SL":
                            new_sl = float(mgmt_result["new_sl"])
                            new_state = mgmt_result["new_state"]
                            with conn.cursor() as cur:
                                cur.execute("""
                                    UPDATE trade_setups
                                    SET stop_loss = %s, trade_state = %s
                                    WHERE id = %s;
                                """, (new_sl, new_state, trade_data["id"]))
                                conn.commit()

                            send_telegram_notification(mgmt_result["msg"])

                except Exception as mgmt_err:
                    logging.error(f"Position management loop error: {mgmt_err}")

        except Exception as conn_err:
            logging.error(f"Worker iteration exception: {conn_err}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_worker_loop()