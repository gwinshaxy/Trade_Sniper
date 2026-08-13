import time
import logging
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from strategy import fetch_klines, evaluate_signals, load_symbol_config, calc_tema
from agent import DynamicTradeManager
from common import get_db_connection, ensure_schema_updated, send_telegram_notification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ensure_schema_updated()

symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "ETH/USDT,BNB/USDT,SOL/USDT")
SYMBOLS = [s.strip().upper() for s in symbols_env.split(",")]
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", 60))
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", 10000.0))
DEFAULT_RISK_PCT = float(os.getenv("RISK_PCT", 1.0))

def run_worker_loop():
    trade_manager = DynamicTradeManager()
    while True:
        try:
            conn = get_db_connection()
            
            # --- PHASE 1: EVALUATE NEW TRADING SIGNALS ---
            for symbol in SYMBOLS:
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
                        pos_size = signal["position_size"]
                        risk_pct = signal.get("risk_pct", DEFAULT_RISK_PCT)
                        
                        # Prevent duplicate active orders on the same pair
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT id FROM trade_setups WHERE pair = %s AND status = 'EXECUTED';",
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

            # --- PHASE 2: MANAGE OPEN POSITIONS (TRAILING STOP / BE) ---
            with conn.cursor() as cur:
                cur.execute("SELECT id, pair, direction, entry_price, stop_loss, take_profit, trade_state FROM trade_setups WHERE status = 'EXECUTED';")
                open_trades = cur.fetchall()
            
            for tr in open_trades:
                trade_id, pair, direction, entry, sl, tp, state = tr
                
                df = fetch_klines(symbol=pair, interval="1h", limit=300)
                if not df.empty:
                    config = load_symbol_config(pair)
                    tema_period = int(config.get("tema_period", 200))
                    
                    df['tema'] = calc_tema(df['close'], period=min(tema_period, len(df)))
                    latest_candle = df.iloc[-1]
                    
                    trade_row = pd.Series({
                        'id': trade_id,
                        'direction': direction,
                        'entry_price': entry,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'trade_state': state
                    })
                    
                    action_res = trade_manager.process_trade(trade_row, latest_candle)
                    
                    if action_res.get("action") == "UPDATE_SL":
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE trade_setups SET stop_loss = %s, trade_state = %s WHERE id = %s;", 
                                (action_res['new_sl'], action_res['new_state'], trade_id)
                            )
                            conn.commit()
                        send_telegram_notification(action_res['msg'])
            conn.close()
        except Exception as e:
            logging.error(f"Worker loop error: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_worker_loop()