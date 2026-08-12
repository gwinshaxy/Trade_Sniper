import time
import logging
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from strategy import evaluate_signals, fetch_klines
from agent import DynamicTradeManager
import optimizer
from common import get_db_connection, ensure_schema_updated, send_telegram_notification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ensure_schema_updated()

symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "ETH/USDT,BNB/USDT,SOL/USDT")
SYMBOLS = [s.strip().upper() for s in symbols_env.split(",")]
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", 60))

def run_worker_loop():
    trade_manager = DynamicTradeManager()
    while True:
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT id, pair, direction, entry_price, stop_loss, take_profit, trade_state FROM trade_setups WHERE status = 'EXECUTED';")
                open_trades = cur.fetchall()
            
            for tr in open_trades:
                trade_id, pair, direction, entry, sl, tp, state = tr
                df = fetch_klines(symbol=pair, interval="1h", limit=100)
                if not df.empty:
                    action_res = trade_manager.process_trade(pd.Series({'id': trade_id, 'direction': direction, 'entry_price': entry, 'stop_loss': sl, 'take_profit': tp, 'trade_state': state}), df.iloc[-1])
                    if action_res.get("action") == "UPDATE_SL":
                        with conn.cursor() as cur:
                            cur.execute("UPDATE trade_setups SET stop_loss = %s, trade_state = %s WHERE id = %s;", (action_res['new_sl'], action_res['new_state'], trade_id))
                            conn.commit()
                        send_telegram_notification(action_res['msg'])
            conn.close()
        except Exception as e:
            logging.error(f"Worker loop error: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_worker_loop()