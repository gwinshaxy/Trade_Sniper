import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import time
import logging
import os
import pandas as pd
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

from strategy import fetch_klines, calc_tema, evaluate_signals, load_symbol_config
from agent import DynamicTradeManager
import optimizer
from common import get_db_connection, ensure_schema_updated, send_telegram_notification, calculate_pnl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "ETHUSDT,BNBUSDT,SOLUSDT")
SYMBOLS = [s.strip().upper() for s in symbols_env.split(",")]

TIMEFRAME = os.getenv("TIMEFRAME", "1h")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", 300))
DEFAULT_ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", 100.0))

# Maximum allowed total combined account risk across all active trades (%)
MAX_TOTAL_PORTFOLIO_RISK_PCT = float(os.getenv("MAX_TOTAL_PORTFOLIO_RISK_PCT", 3.0))

SYMBOL_COOLDOWN = {}
COOLDOWN_PERIOD_SECONDS = int(os.getenv("COOLDOWN_PERIOD_SECONDS", 900))

def run_all_optimizations():
    """
    Executes the DEAP Genetic Optimizer sequentially across all configured symbols.
    """
    logging.info("🧬 Starting weekly scheduled DEAP Genetic Optimization for all symbols...")
    for sym in SYMBOLS:
        clean_sym = sym.replace("/", "").upper()
        try:
            logging.info(f"⏳ Optimizing parameters for {clean_sym}...")
            optimizer.run_optimization(symbol=clean_sym)
        except Exception as e:
            logging.error(f"Failed background optimization for {clean_sym}: {e}")

def start_background_optimizer():
    """
    Schedules the DEAP Genetic Optimizer to run across all symbols every Sunday at 00:00 UTC 
    in the background using APScheduler.
    """
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        func=run_all_optimizations,
        trigger="cron",
        day_of_week="sun",
        hour=0,
        minute=0,
        id="deap_optimizer_job",
        replace_existing=True
    )
    scheduler.start()
    logging.info("⏰ Background DEAP Optimizer Scheduler started (Runs every Sunday @ 00:00 UTC for all assets).")
    return scheduler

def get_total_open_risk_pct() -> float:
    """
    Calculates the aggregate percentage risk across all active/pending trades in the database.
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(SUM(risk_pct), 0) 
                FROM trade_setups 
                WHERE status IN ('PENDING', 'EXECUTED') AND trade_state != 'CLOSED';
            """)
            total_risk = cur.fetchone()[0]
        conn.close()
        return float(total_risk)
    except Exception as e:
        logging.error(f"Error calculating total open portfolio risk: {e}")
        return 0.0

def has_active_trade_for_symbol(pair: str) -> bool:
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM trade_setups 
                WHERE pair = %s AND status IN ('PENDING', 'EXECUTED');
            """, (pair,))
            count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception as e:
        logging.error(f"Error checking active trade for {pair}: {e}")
        return False

def fetch_active_trades():
    try:
        conn = get_db_connection()
        query = """
            SELECT id, pair, direction, entry_price, stop_loss, take_profit, 
                   risk_reward_ratio, risk_pct, position_size, status, trade_state, account_balance
            FROM trade_setups
            WHERE status IN ('PENDING', 'EXECUTED')
            ORDER BY created_at DESC;
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Error fetching active trades: {e}")
        return pd.DataFrame()

def update_trade_sl_and_state(trade_id: int, new_sl: float, new_state: str, new_status: str = None):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            if new_status:
                cur.execute(
                    "UPDATE trade_setups SET stop_loss = %s, trade_state = %s, status = %s WHERE id = %s;",
                    (new_sl, new_state, new_status, trade_id)
                )
            else:
                cur.execute(
                    "UPDATE trade_setups SET stop_loss = %s, trade_state = %s WHERE id = %s;",
                    (new_sl, new_state, trade_id)
                )
            conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to update trade #{trade_id}: {e}")

def close_trade_in_db(trade_id: int, exit_price: float, pnl_usd: float, pnl_pct: float, outcome: str):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trade_setups 
                SET status = 'CLOSED', trade_state = 'CLOSED', exit_price = %s, pnl_usd = %s, pnl_pct = %s, outcome = %s, closed_at = CURRENT_TIMESTAMP
                WHERE id = %s;
            """, (exit_price, pnl_usd, pnl_pct, outcome, trade_id))
            conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to close trade #{trade_id}: {e}")

def insert_new_signal(pair, direction, entry, sl, tp, rr, balance, risk_pct, size):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trade_setups 
                (pair, direction, entry_price, stop_loss, take_profit, risk_reward_ratio, account_balance, risk_pct, position_size, status, trade_state)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN')
                RETURNING id;
            """, (pair, direction, entry, sl, tp, rr, balance, risk_pct, size))
            trade_id = cur.fetchone()[0]
            conn.commit()
        conn.close()
        return trade_id
    except Exception as e:
        logging.error(f"Failed to insert signal: {e}")
        return None

def main():
    logging.info(f"⚡ Starting Multi-Asset Trading Bot Background Engine (Max Portfolio Risk Cap: {MAX_TOTAL_PORTFOLIO_RISK_PCT}%)...")
    ensure_schema_updated()
    
    # Initialize background cron optimizer
    scheduler = start_background_optimizer()
    
    trade_manager = DynamicTradeManager()

    try:
        while True:
            candle_cache = {}

            for symbol in SYMBOLS:
                logging.info(f"🔍 Polling {symbol} ({TIMEFRAME})...")
                formatted_symbol = symbol if "/" in symbol else f"{symbol[:-4]}/{symbol[-4:]}"

                if has_active_trade_for_symbol(formatted_symbol):
                    logging.info(f"🛡️ Skipping {formatted_symbol}: Active trade already open or pending in database.")
                    time.sleep(2.0)
                    continue

                last_signal_time = SYMBOL_COOLDOWN.get(formatted_symbol, 0)
                if time.time() - last_signal_time < COOLDOWN_PERIOD_SECONDS:
                    remaining_sec = int(COOLDOWN_PERIOD_SECONDS - (time.time() - last_signal_time))
                    logging.info(f"⏳ Skipping {formatted_symbol}: In cooldown for another {remaining_sec}s.")
                    time.sleep(2.0)
                    continue

                try:
                    df = fetch_klines(symbol=symbol, interval=TIMEFRAME, limit=500)
                    if df.empty:
                        logging.warning(f"Received empty dataframe for {symbol}. Skipping asset...")
                        time.sleep(2.0)
                        continue
                except Exception as e:
                    logging.error(f"Network error fetching klines for {symbol}: {e}. Skipping asset...")
                    time.sleep(2.0)
                    continue

                # Load asset-specific config to get optimized tema_period
                cfg = load_symbol_config(formatted_symbol)
                tema_period = int(cfg.get("tema_period", 200))

                df['200_TEMA'] = calc_tema(df['close'], tema_period)
                latest_candle = df.iloc[-1]
                logging.info(f"📊 {symbol} Latest Candle Close: ${latest_candle['close']:,.2f} (TEMA-{tema_period}: ${latest_candle['200_TEMA']:,.2f})")

                candle_cache[symbol.upper()] = latest_candle

                signal = evaluate_signals(df, symbol=formatted_symbol, account_balance=DEFAULT_ACCOUNT_BALANCE)
                
                if signal.get("action") in ["BUY", "SELL"]:
                    pair_name = signal.get("pair", formatted_symbol)
                    signal_risk = float(signal.get("risk_pct", 1.0))
                    
                    # Portfolio Risk Exposure Guardrail
                    current_portfolio_risk = get_total_open_risk_pct()
                    if (current_portfolio_risk + signal_risk) > MAX_TOTAL_PORTFOLIO_RISK_PCT:
                        logging.warning(
                            f"🛡️ Portfolio Risk Exceeded for {pair_name}: Current Open Risk ({current_portfolio_risk:.2f}%) + "
                            f"New Trade Risk ({signal_risk:.2f}%) > Max Limit ({MAX_TOTAL_PORTFOLIO_RISK_PCT:.2f}%). Execution blocked."
                        )
                        time.sleep(2.0)
                        continue

                    SYMBOL_COOLDOWN[formatted_symbol] = time.time()

                    trade_id = insert_new_signal(
                        pair=pair_name,
                        direction=signal["direction"],
                        entry=signal["entry"],
                        sl=signal["sl"],
                        tp=signal["tp"],
                        rr=signal["rr"],
                        balance=DEFAULT_ACCOUNT_BALANCE,
                        risk_pct=signal_risk,
                        size=signal["size"]
                    )
                    if trade_id:
                        alert_action = "BUY (LONG)" if signal["direction"] == "LONG" else "SELL (SHORT)"
                        alert_emoji = "🚀" if signal["direction"] == "LONG" else "📉"

                        alert_txt = (
                            f"<b>{alert_emoji} AUTOMATED {alert_action} SIGNAL EXECUTED</b>\n"
                            f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                            f"<b>Pair:</b> <code>{pair_name}</code>\n"
                            f"<b>Direction:</b> <code>{signal['direction']}</code>\n"
                            f"<b>Entry:</b> ${signal['entry']:,.2f}\n"
                            f"<b>SL:</b> ${signal['sl']:,.2f}\n"
                            f"<b>TP:</b> ${signal['tp']:,.2f}\n"
                            f"<b>R:R:</b> 1:{signal['rr']}\n"
                            f"<b>Risk:</b> {signal_risk}%\n"
                            f"<b>Total Portfolio Risk:</b> {current_portfolio_risk + signal_risk:.2f}% / {MAX_TOTAL_PORTFOLIO_RISK_PCT}%\n"
                            f"<b>Size:</b> {signal['size']} units"
                        )
                        logging.info(f"New Signal Executed: Trade #{trade_id} ({pair_name} - {signal['direction']})")
                        send_telegram_notification(alert_txt)
                else:
                    logging.info(f"Status for {symbol}: HOLD / No signal matched")

                time.sleep(2.0)

            active_trades = fetch_active_trades()
            if not active_trades.empty:
                for _, trade in active_trades.iterrows():
                    trade_pair_raw = str(trade['pair'])
                    cache_key = trade_pair_raw.replace("/", "").upper()
                    
                    if cache_key not in candle_cache:
                        try:
                            logging.info(f"🛡️ Fallback fetch: Retrieving missing candle data for active trade pair {trade_pair_raw}...")
                            clean_symbol = trade_pair_raw.replace("/", "")
                            fallback_df = fetch_klines(symbol=clean_symbol, interval=TIMEFRAME, limit=1000)
                            if not fallback_df.empty:
                                cfg = load_symbol_config(trade_pair_raw)
                                tema_period = int(cfg.get("tema_period", 200))
                                fallback_df['200_TEMA'] = calc_tema(fallback_df['close'], tema_period)
                                candle_cache[cache_key] = fallback_df.iloc[-1]
                            time.sleep(1.0)
                        except Exception as fe:
                            logging.error(f"Failed fallback candle fetch for {trade_pair_raw}: {fe}")

                executed_trades = active_trades[active_trades['status'] == 'EXECUTED']
                
                if not executed_trades.empty:
                    logging.info(f"📈 Monitoring {len(executed_trades)} active open trade(s)...")

                    for _, trade in executed_trades.iterrows():
                        trade_pair = str(trade['pair']).replace("/", "").upper()
                        matched_candle = candle_cache.get(trade_pair)
                        
                        if matched_candle is not None:
                            curr_low = float(matched_candle['low'])
                            curr_high = float(matched_candle['high'])
                            
                            sl = float(trade['stop_loss'])
                            tp = float(trade['take_profit'])
                            direction = trade['direction']
                            trade_id = int(trade['id'])
                            entry = float(trade['entry_price'])
                            size = float(trade['position_size'])
                            balance = float(trade['account_balance']) if pd.notnull(trade['account_balance']) else DEFAULT_ACCOUNT_BALANCE

                            hit_sl = (direction == 'LONG' and curr_low <= sl) or (direction == 'SHORT' and curr_high >= sl)
                            hit_tp = (direction == 'LONG' and curr_high >= tp) or (direction == 'SHORT' and curr_low <= tp)

                            if hit_sl or hit_tp:
                                exit_price = sl if hit_sl else tp
                                pnl_usd, pnl_pct, outcome = calculate_pnl(direction, entry, exit_price, size, balance)

                                close_trade_in_db(
                                    trade_id=trade_id,
                                    exit_price=exit_price,
                                    pnl_usd=pnl_usd,
                                    pnl_pct=pnl_pct,
                                    outcome=outcome
                                )

                                exit_label = "STOP LOSS (SL)" if hit_sl else "TAKE PROFIT (TP)"
                                exit_emoji = "🔴" if hit_sl else "🟢"
                                
                                alert_msg = (
                                    f"<b>{exit_emoji} TRADE #{trade_id} CLOSED BY {exit_label}</b>\n"
                                    f"<b>Pair:</b> <code>{trade['pair']}</code>\n"
                                    f"<b>Direction:</b> <code>{direction}</code>\n"
                                    f"<b>Exit Price:</b> ${exit_price:,.2f}\n"
                                    f"<b>PnL ($):</b> ${pnl_usd:,.2f}\n"
                                    f"<b>PnL (%):</b> {pnl_pct:.2f}%"
                                )
                                logging.info(f"🔒 Trade #{trade_id} closed by {exit_label} at price {exit_price}.")
                                send_telegram_notification(alert_msg)
                            else:
                                res = trade_manager.process_trade(trade, matched_candle)
                                if res.get("action") == "UPDATE_SL":
                                    update_trade_sl_and_state(
                                        trade_id=res["trade_id"],
                                        new_sl=res["new_sl"],
                                        new_state=res["new_state"]
                                    )
                                    logging.info(res["msg"])
                                    send_telegram_notification(f"<b>⚙️ RISK MANAGER UPDATE</b>\n{res['msg']}")

            time.sleep(POLL_INTERVAL_SECONDS)

    except (KeyboardInterrupt, SystemExit):
        logging.info("👋 Shutting down worker and background scheduler...")
        scheduler.shutdown()

if __name__ == "__main__":
    main()