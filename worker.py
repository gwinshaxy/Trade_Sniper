import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import time
import logging
import os
import urllib.request
import json
import pandas as pd
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

# ==========================================
# PROXY & NETWORK CONFIGURATION ROUTING
# ==========================================
HTTP_PROXY = os.getenv("HTTP_PROXY") or os.getenv("PROXY_URL")
HTTPS_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("PROXY_URL")

if HTTP_PROXY or HTTPS_PROXY:
    os.environ["HTTP_PROXY"] = HTTP_PROXY or HTTPS_PROXY
    os.environ["HTTPS_PROXY"] = HTTPS_PROXY or HTTP_PROXY
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,.supabase.co"

from strategy import calc_tema, evaluate_signals, load_symbol_config
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
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", 600))
DEFAULT_ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", 100.0))
MAX_TOTAL_PORTFOLIO_RISK_PCT = float(os.getenv("MAX_TOTAL_PORTFOLIO_RISK_PCT", 3.0))

SYMBOL_COOLDOWN = {}
COOLDOWN_PERIOD_SECONDS = int(os.getenv("COOLDOWN_PERIOD_SECONDS", 1800))

def normalize_symbol(symbol: str) -> str:
    """
    Ensures uniform formatting for symbols (e.g., 'ETH/USDT').
    """
    s = symbol.strip().upper()
    if "/" in s:
        return s
    # Convert compact formatting like ETHUSDT -> ETH/USDT
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/{s[-4:]}"
    return s

def get_cache_key(symbol: str) -> str:
    """
    Generates a uniform, slash-free uppercase key for candle dictionary mapping.
    """
    return normalize_symbol(symbol).replace("/", "").upper()

def fetch_market_data(symbol: str, interval_str: str):
    """
    Alternative public market data fetcher using CoinGecko REST API
    to bypass Yahoo Finance cloud IP rate-limiting (HTTP 429).
    Synchronized with dashboard.py implementation.
    """
    try:
        normalized = normalize_symbol(symbol)
        clean_base = normalized.split("/")[0] if "/" in normalized else normalized.replace("USDT", "")
        
        symbol_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "BNB": "binancecoin",
            "ADA": "cardano",
            "SOL": "solana",
            "XRP": "ripple",
            "DOGE": "dogecoin",
            "DOT": "polkadot",
            "AVAX": "avalanche-2"
        }
        coin_id = symbol_map.get(clean_base, "bitcoin")
        
        days_map = {"5m": 1, "15m": 7, "30m": 14, "1h": 30, "4h": 90, "1d": 365}
        days = days_map.get(interval_str, 30)
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            
        if not data:
            return None
            
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close'])
        df['time'] = pd.to_datetime(df['time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
        df['volume'] = 1000.0  # Synthetic placeholder volume for profile alignment if omitted
        
        return df[['time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logging.error(f"Error fetching alternative market data for {symbol}: {e}")
        return None

def run_all_optimizations():
    """Executes the DEAP Genetic Optimizer sequentially across all configured symbols."""
    logging.info("🧬 Starting weekly scheduled DEAP Genetic Optimization for all symbols...")
    for sym in SYMBOLS:
        clean_sym = normalize_symbol(sym).replace("/", "").upper()
        try:
            logging.info(f"⏳ Optimizing parameters for {clean_sym}...")
            optimizer.run_optimization(symbol=clean_sym)
        except Exception as e:
            logging.error(f"Failed background optimization for {clean_sym}: {e}")

def start_background_optimizer():
    """Schedules the DEAP Genetic Optimizer to run every Sunday at 00:00 UTC."""
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
    logging.info("⏰ Background DEAP Optimizer Scheduler started.")
    return scheduler

def get_total_open_risk_pct() -> float:
    """Calculates aggregate percentage risk across all active trades."""
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
    proxy_status = f"Active ({HTTP_PROXY or HTTPS_PROXY})" if (HTTP_PROXY or HTTPS_PROXY) else "Direct Connection"
    logging.info(f"⚡ Starting Multi-Asset Trading Bot Engine | Proxy: {proxy_status}")
    ensure_schema_updated()
    
    scheduler = start_background_optimizer()
    trade_manager = DynamicTradeManager()

    try:
        while True:
            candle_cache = {}

            for symbol in SYMBOLS:
                formatted_symbol = normalize_symbol(symbol)
                logging.info(f"🔍 Polling {formatted_symbol} ({TIMEFRAME})...")

                if has_active_trade_for_symbol(formatted_symbol):
                    logging.info(f"🛡️ Skipping {formatted_symbol}: Active trade already open.")
                    time.sleep(20.0)
                    continue

                last_signal_time = SYMBOL_COOLDOWN.get(formatted_symbol, 0)
                if time.time() - last_signal_time < COOLDOWN_PERIOD_SECONDS:
                    logging.info(f"⏳ Skipping {formatted_symbol}: In cooldown.")
                    time.sleep(20.0)
                    continue

                try:
                    df = fetch_market_data(symbol=formatted_symbol, interval_str=TIMEFRAME)
                    if df is None or df.empty:
                        logging.warning(f"Received empty dataframe for {formatted_symbol}. Skipping...")
                        time.sleep(20.0)
                        continue
                except Exception as e:
                    logging.error(f"Network error fetching klines for {formatted_symbol}: {e}. Skipping...")
                    time.sleep(20.0)
                    continue

                cfg = load_symbol_config(formatted_symbol)
                tema_period = int(cfg.get("tema_period", 200))

                df['200_TEMA'] = calc_tema(df['close'], tema_period)
                latest_candle = df.iloc[-1]
                logging.info(f"📊 {formatted_symbol} Close: ${latest_candle['close']:,.2f} (TEMA: ${latest_candle['200_TEMA']:,.2f})")

                cache_key = get_cache_key(formatted_symbol)
                candle_cache[cache_key] = latest_candle

                signal = evaluate_signals(df, symbol=formatted_symbol, account_balance=DEFAULT_ACCOUNT_BALANCE)
                
                if signal.get("action") in ["BUY", "SELL"]:
                    pair_name = normalize_symbol(signal.get("pair", formatted_symbol))
                    signal_risk = float(signal.get("risk_pct", 1.0))
                    
                    current_portfolio_risk = get_total_open_risk_pct()
                    if (current_portfolio_risk + signal_risk) > MAX_TOTAL_PORTFOLIO_RISK_PCT:
                        logging.warning(f"🛡️ Portfolio Risk Exceeded for {pair_name}. Execution blocked.")
                        time.sleep(20.0)
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
                        alert_txt = (
                            f"<b>🚀 AUTOMATED SIGNAL EXECUTED</b>\n"
                            f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                            f"<b>Pair:</b> <code>{pair_name}</code>\n"
                            f"<b>Direction:</b> <code>{signal['direction']}</code>\n"
                            f"<b>Entry:</b> ${signal['entry']:,.2f}"
                        )
                        logging.info(f"New Signal Executed: Trade #{trade_id} ({pair_name})")
                        send_telegram_notification(alert_txt)
                else:
                    logging.info(f"Status for {formatted_symbol}: HOLD")
                time.sleep(20.0)

            active_trades = fetch_active_trades()
            if not active_trades.empty:
                for _, trade in active_trades.iterrows():
                    trade_pair_raw = str(trade['pair'])
                    normalized_trade_pair = normalize_symbol(trade_pair_raw)
                    cache_key = get_cache_key(normalized_trade_pair)
                    
                    if cache_key not in candle_cache:
                        try:
                            fallback_df = fetch_market_data(symbol=normalized_trade_pair, interval_str=TIMEFRAME)
                            if fallback_df is not None and not fallback_df.empty:
                                cfg = load_symbol_config(normalized_trade_pair)
                                tema_period = int(cfg.get("tema_period", 200))
                                fallback_df['200_TEMA'] = calc_tema(fallback_df['close'], tema_period)
                                candle_cache[cache_key] = fallback_df.iloc[-1]
                            time.sleep(20.0)
                        except Exception as fe:
                            logging.error(f"Failed fallback candle fetch for {normalized_trade_pair}: {fe}")

                executed_trades = active_trades[active_trades['status'] == 'EXECUTED']
                if not executed_trades.empty:
                    for _, trade in executed_trades.iterrows():
                        trade_pair = normalize_symbol(str(trade['pair']))
                        matched_candle = candle_cache.get(get_cache_key(trade_pair))
                        
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
                                close_trade_in_db(trade_id, exit_price, pnl_usd, pnl_pct, outcome)
                                exit_label = "SL" if hit_sl else "TP"
                                send_telegram_notification(f"<b>🔒 TRADE #{trade_id} CLOSED BY {exit_label}</b>")
                            else:
                                res = trade_manager.process_trade(trade, matched_candle)
                                if res.get("action") == "UPDATE_SL":
                                    update_trade_sl_and_state(res["trade_id"], res["new_sl"], res["new_state"])
                                    send_telegram_notification(f"<b>⚙️ RISK MANAGER UPDATE</b>\n{res['msg']}")

            time.sleep(POLL_INTERVAL_SECONDS)

    except (KeyboardInterrupt, SystemExit):
        logging.info("👋 Shutting down worker and background scheduler...")
        scheduler.shutdown()

if __name__ == "__main__":
    main()