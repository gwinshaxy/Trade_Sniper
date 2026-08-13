import time
import logging
import os
import gc
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from strategy import fetch_klines, evaluate_signals, load_symbol_config, calc_tema
from agent import DynamicTradeManager
from common import get_db_connection, ensure_schema_updated, send_telegram_notification, calculate_pnl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ensure_schema_updated()

symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "ETH/USDT,BNB/USDT,SOL/USDT")
SYMBOLS = [s.strip().upper() for s in symbols_env.split(",")]
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", 60))
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", 10000.0))
DEFAULT_RISK_PCT = float(os.getenv("RISK_PCT", 1.0))


def run_worker_loop():
    trade_manager = DynamicTradeManager()
    logging.info(f"Worker process active. Tracking symbols: {SYMBOLS}")

    while True:
        conn = None
        try:
            conn = get_db_connection()
            if conn:
                # --- PHASE 1: EVALUATE NEW TRADING SIGNALS ---
                for symbol in SYMBOLS:
                    try:
                        df = fetch_klines(symbol=symbol, interval="1h", limit=150)
                        if not df.empty:
                            signal = evaluate_signals(df, symbol=symbol, account_balance=ACCOUNT_BALANCE)
                            if signal.get("action") in ["BUY", "SELL"]:
                                pair = signal["symbol"]
                                direction = signal["direction"]
                                entry = signal["entry_price"]
                                sl = signal["stop_loss"]
                                tp = signal["take_profit"]
                                rr = signal["risk_reward_ratio"]
                                pos_size = signal["position_size"]
                                risk_pct = signal.get("risk_pct", DEFAULT_RISK_PCT)

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
                                            (pair, direction, entry, sl, tp, rr, risk_pct, pos_size, ACCOUNT_BALANCE)
                                        )
                                        conn.commit()
                                        send_telegram_notification(
                                            f"<b>🚀 AUTOMATED TRADE EXECUTED</b>\n\n"
                                            f"<b>Pair:</b> <code>{pair}</code>\n"
                                            f"<b>Direction:</b> <code>{direction}</code>\n"
                                            f"<b>Entry:</b> ${entry:.5f}\n"
                                            f"<b>SL:</b> ${sl:.5f} | <b>TP:</b> ${tp:.5f}\n"
                                            f"<b>Risk:</b> {risk_pct}%"
                                        )
                    except Exception as sym_err:
                        logging.error(f"[PHASE 1] Error evaluating signal for {symbol}: {sym_err}")

                # --- PHASE 2: MANAGE OPEN POSITIONS (TRAILING STOP / BE / CLOSES) ---
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT id, pair, direction, entry_price, stop_loss, take_profit, position_size, account_balance, trade_state FROM trade_setups WHERE status = 'EXECUTED' AND trade_state != 'CLOSED';")
                        open_trades = cur.fetchall()

                    for tr in open_trades:
                        trade_id, pair, direction, entry, sl, tp, pos_size, acct_bal, state = tr

                        df = fetch_klines(symbol=pair, interval="1h", limit=150)
                        if not df.empty:
                            config = load_symbol_config(pair)
                            tema_period = int(config.get("tema_period", 200))

                            df['tema'] = calc_tema(df['close'], period=min(tema_period, len(df)))
                            latest_candle = df.iloc[-1]

                            trade_row = pd.Series({
                                'id': trade_id,
                                'pair': pair,
                                'direction': direction,
                                'entry_price': float(entry),
                                'stop_loss': float(sl),
                                'take_profit': float(tp),
                                'position_size': float(pos_size),
                                'trade_state': state or 'OPEN'
                            })

                            action_res = trade_manager.process_trade(trade_row, latest_candle)
                            action = action_res.get("action")

                            if action in ["CLOSE_SL", "CLOSE_TP"]:
                                exit_price = action_res["exit_price"]
                                pnl_usd, pnl_pct, outcome = calculate_pnl(
                                    direction, float(entry), float(exit_price), float(pos_size), float(acct_bal or ACCOUNT_BALANCE)
                                )

                                with conn.cursor() as cur:
                                    cur.execute(
                                        """
                                        UPDATE trade_setups 
                                        SET exit_price = %s, pnl_usd = %s, pnl_pct = %s, outcome = %s, status = 'CLOSED', trade_state = 'CLOSED', closed_at = CURRENT_TIMESTAMP 
                                        WHERE id = %s;
                                        """,
                                        (exit_price, pnl_usd, pnl_pct, outcome, trade_id)
                                    )
                                    conn.commit()

                                send_telegram_notification(
                                    f"{action_res['msg']}\n\n"
                                    f"<b>PnL:</b> ${pnl_usd:,.2f} ({outcome})\n"
                                    f"<b>Exit Price:</b> ${exit_price:.5f}"
                                )

                            elif action == "UPDATE_SL":
                                with conn.cursor() as cur:
                                    cur.execute(
                                        "UPDATE trade_setups SET stop_loss = %s, trade_state = %s WHERE id = %s;", 
                                        (action_res['new_sl'], action_res['new_state'], trade_id)
                                    )
                                    conn.commit()
                                send_telegram_notification(action_res['msg'])

                except Exception as pos_err:
                    logging.error(f"[PHASE 2] Error managing open positions: {pos_err}")

        except Exception as e:
            logging.error(f"Worker loop connection/execution error: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            gc.collect()

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_worker_loop()