import os
import psycopg2
from psycopg2 import pool
import requests
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=env_path)

DB_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
if DB_URL:
    DB_URL = DB_URL.strip('"').strip("'")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Dynamic Connection Pool setup for threaded services
_db_pool = None

def get_db_pool():
    global _db_pool
    if _db_pool is None:
        if not DB_URL:
            raise ValueError("Neither DATABASE_URL nor DB_URL environment variable is set in .env.")
        _db_pool = pool.ThreadedConnectionPool(1, 20, DB_URL)
    return _db_pool

class ConnectionContext:
    def __init__(self, db_pool):
        self.pool = db_pool
        self.conn = None

    def __enter__(self):
        self.conn = self.pool.getconn()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is not None:
                self.conn.rollback()
            self.pool.putconn(self.conn)

def get_db_connection():
    """Establishes and returns a PostgreSQL database connection from the pool."""
    p = get_db_pool()
    conn = p.getconn()
    # Attach monkey-patched close method to return connection to pool cleanly
    original_close = conn.close
    def pooled_close():
        try:
            p.putconn(conn)
        except Exception:
            original_close()
    conn.close = pooled_close
    return conn

def ensure_schema_updated():
    """Ensures trade_setups and strategy_parameters tables exist with fully synchronized schemas."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_setups (
            id SERIAL PRIMARY KEY,
            pair VARCHAR(20) NOT NULL,
            direction VARCHAR(10) NOT NULL,
            entry_price NUMERIC(18, 8) NOT NULL,
            stop_loss NUMERIC(18, 8) NOT NULL,
            take_profit NUMERIC(18, 8) NOT NULL,
            risk_reward_ratio NUMERIC(18, 2),
            account_balance NUMERIC(18, 2) NOT NULL,
            risk_pct NUMERIC(18, 2) DEFAULT 1.0,
            position_size NUMERIC(18, 8) NOT NULL,
            status VARCHAR(20) DEFAULT 'PENDING',
            trade_state VARCHAR(20) DEFAULT 'OPEN',
            exit_price NUMERIC(18, 8),
            pnl_usd NUMERIC(18, 2),
            pnl_pct NUMERIC(18, 2),
            outcome VARCHAR(10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP
        );
    """)
    
    trade_columns = [
        "risk_reward_ratio NUMERIC(18, 2)",
        "risk_pct NUMERIC(18, 2) DEFAULT 1.0",
        "trade_state VARCHAR(20) DEFAULT 'OPEN'",
        "exit_price NUMERIC(18, 8)",
        "pnl_usd NUMERIC(18, 2)",
        "pnl_pct NUMERIC(18, 2)",
        "outcome VARCHAR(10)",
        "closed_at TIMESTAMP"
    ]
    for col in trade_columns:
        cursor.execute(f"ALTER TABLE trade_setups ADD COLUMN IF NOT EXISTS {col};")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_parameters (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) UNIQUE NOT NULL,
            tema_period INT NOT NULL DEFAULT 200,
            rsi_period INT NOT NULL DEFAULT 14,
            rsi_thresh FLOAT DEFAULT 42.0,
            adx_period INT DEFAULT 14,
            adx_threshold FLOAT DEFAULT 20.0,
            use_adx_filter BOOLEAN DEFAULT TRUE,
            max_sl_pct FLOAT DEFAULT 0.02,
            zone_tolerance NUMERIC NOT NULL DEFAULT 0.0075,
            min_sentiment NUMERIC NOT NULL DEFAULT 0.0,
            risk_pct NUMERIC NOT NULL DEFAULT 1.0,
            min_rr NUMERIC NOT NULL DEFAULT 2.0,
            vp_detection_pct NUMERIC NOT NULL DEFAULT 0.07,
            use_rsi_filter BOOLEAN DEFAULT TRUE,
            use_candlestick_confirm BOOLEAN DEFAULT TRUE,
            fitness_score NUMERIC DEFAULT 0.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    param_columns = [
        "adx_period INT DEFAULT 14",
        "adx_threshold FLOAT DEFAULT 20.0",
        "use_adx_filter BOOLEAN DEFAULT TRUE",
        "max_sl_pct FLOAT DEFAULT 0.02"
    ]
    for col in param_columns:
        cursor.execute(f"ALTER TABLE strategy_parameters ADD COLUMN IF NOT EXISTS {col};")
        
    conn.commit()
    cursor.close()
    conn.close()

def calculate_pnl(direction: str, entry_price: float, exit_price: float, position_size: float, account_balance: float):
    """Calculates realized PnL in USD and Percentage."""
    if direction.upper() == "LONG":
        pnl_usd = (exit_price - entry_price) * position_size
    else:
        pnl_usd = (entry_price - exit_price) * position_size

    pnl_pct = (pnl_usd / account_balance) * 100 if account_balance > 0 else 0.0
    outcome = "WIN" if pnl_usd >= 0 else "LOSS"
    return round(pnl_usd, 2), round(pnl_pct, 2), outcome

def send_telegram_notification(message: str) -> bool:
    """Sends an HTML formatted Telegram alert if credentials exist."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        return resp.status_code == 200
    except Exception as e:
        print(f"[Telegram Alert Error]: {e}")
        return False

def close_trade_manually(trade_id: int, exit_price: float, reason: str = "MANUAL_CLOSE"):
    """Manually closes an active trade, updates PnL, and notifies via Telegram."""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT direction, entry_price, position_size, account_balance, pair 
            FROM trade_setups WHERE id = %s AND status IN ('EXECUTED', 'PENDING');
        """, (trade_id,))
        row = cursor.fetchone()
        if not row:
            cursor.close()
            conn.close()
            return False

        direction, entry_price, position_size, account_balance, pair = row
        pnl_usd, pnl_pct, outcome = calculate_pnl(
            direction, float(entry_price), float(exit_price), float(position_size), float(account_balance or 10000.0)
        )

        cursor.execute("""
            UPDATE trade_setups
            SET exit_price = %s, pnl_usd = %s, pnl_pct = %s, outcome = %s,
                status = 'CLOSED', trade_state = 'CLOSED', closed_at = CURRENT_TIMESTAMP
            WHERE id = %s;
        """, (round(exit_price, 5), pnl_usd, pnl_pct, outcome, trade_id))
        conn.commit()
        cursor.close()
        conn.close()

        emoji = "🔴" if pnl_usd < 0 else "🟢"
        send_telegram_notification(
            f"<b>{emoji} MANUAL TRADE CLOSED ({reason})</b>\n\n"
            f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
            f"<b>Pair:</b> <code>{pair}</code>\n"
            f"<b>Exit Price:</b> ${exit_price:.5f}\n"
            f"<b>PnL:</b> ${pnl_usd:,.2f} ({pnl_pct:.2f}%)"
        )
        return True
    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        print(f"[Manual Close Error]: {e}")
        return False