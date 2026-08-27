import os
import logging
import psycopg2
from psycopg2 import pool
import requests
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger("trading_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DB_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
if DB_URL:
    DB_URL = DB_URL.strip('"').strip("'")

DB_HOST = os.getenv("DB_HOST", "db.vbczbralfqdwmtxqtzxk.supabase.co")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS")
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

_db_pool = None


def init_db_pool():
    global _db_pool
    if _db_pool is None:
        try:
            if DB_URL:
                _db_pool = pool.ThreadedConnectionPool(1, 20, DB_URL)
            elif DB_PASS:
                _db_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=20,
                    host=DB_HOST,
                    port=DB_PORT,
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASS,
                    sslmode=DB_SSLMODE,
                    keepalives=1,
                    keepalives_idle=30,
                    keepalives_interval=10,
                    keepalives_count=5
                )
            else:
                raise ValueError("Neither DATABASE_URL/DB_URL nor DB_PASS environment variables are defined!")
            
            logger.info("Database connection pool initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize DB connection pool: {e}")
            raise e


def get_db_pool():
    global _db_pool
    if _db_pool is None:
        init_db_pool()
    return _db_pool


class PooledConnectionWrapper:
    def __init__(self, conn, db_pool):
        self._conn = conn
        self._pool = db_pool

    def close(self):
        if self._conn and self._pool:
            try:
                self._pool.putconn(self._conn)
            except Exception:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            try:
                self._conn.rollback()
            except Exception:
                pass
        self.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def get_db_connection():
    p = get_db_pool()
    conn = p.getconn()
    return PooledConnectionWrapper(conn, p)


def release_db_connection(conn):
    global _db_pool
    if conn:
        if isinstance(conn, PooledConnectionWrapper):
            conn.close()
        elif _db_pool:
            try:
                _db_pool.putconn(conn)
            except Exception:
                pass


def execute_query(conn, query, params=None, fetch=False):
    cursor = conn.cursor()
    try:
        cursor.execute(query, params or ())
        if fetch:
            result = cursor.fetchall()
        else:
            conn.commit()
            result = None
        cursor.close()
        return result
    except Exception as e:
        conn.rollback()
        cursor.close()
        raise e


def ensure_schema_updated():
    """Ensures trade_setups and strategy_parameters tables exist with fully synchronized schemas including ATR parameters."""
    conn = get_db_connection()
    if not conn:
        logger.error("Database connection unavailable for schema check.")
        return

    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_setups (
                id SERIAL PRIMARY KEY,
                pair VARCHAR(20) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                entry_price NUMERIC(18, 8) NOT NULL,
                stop_loss NUMERIC(18, 8),
                take_profit NUMERIC(18, 8),
                risk_reward_ratio NUMERIC(18, 2),
                account_balance NUMERIC(18, 8),
                risk_pct NUMERIC(18, 2) DEFAULT 1.0,
                position_size NUMERIC(18, 8) NOT NULL,
                status VARCHAR(20) DEFAULT 'EXECUTED',
                trade_state VARCHAR(20) DEFAULT 'OPEN',
                exit_price NUMERIC(18, 8),
                pnl_usd NUMERIC(18, 8),
                pnl_pct NUMERIC(18, 8),
                outcome VARCHAR(20),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP WITH TIME ZONE
            );
        """)
        
        trade_columns = [
            "stop_loss NUMERIC(18, 8)",
            "take_profit NUMERIC(18, 8)",
            "risk_reward_ratio NUMERIC(18, 2)",
            "risk_pct NUMERIC(18, 2) DEFAULT 1.0",
            "trade_state VARCHAR(20) DEFAULT 'OPEN'",
            "exit_price NUMERIC(18, 8)",
            "pnl_usd NUMERIC(18, 8)",
            "pnl_pct NUMERIC(18, 8)",
            "outcome VARCHAR(20)",
            "closed_at TIMESTAMP WITH TIME ZONE"
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
                atr_period INT DEFAULT 14,
                atr_mult FLOAT DEFAULT 2.0,
                use_atr_sl BOOLEAN DEFAULT TRUE,
                disable_htf BOOLEAN DEFAULT FALSE,
                fitness_score NUMERIC DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        param_columns = [
            "adx_period INT DEFAULT 14",
            "adx_threshold FLOAT DEFAULT 20.0",
            "use_adx_filter BOOLEAN DEFAULT TRUE",
            "max_sl_pct FLOAT DEFAULT 0.02",
            "zone_tolerance NUMERIC DEFAULT 0.0075",
            "min_sentiment NUMERIC DEFAULT 0.0",
            "min_rr NUMERIC DEFAULT 2.0",
            "vp_detection_pct NUMERIC DEFAULT 0.07",
            "use_rsi_filter BOOLEAN DEFAULT TRUE",
            "use_candlestick_confirm BOOLEAN DEFAULT TRUE",
            "atr_period INT DEFAULT 14",
            "atr_mult FLOAT DEFAULT 2.0",
            "use_atr_sl BOOLEAN DEFAULT TRUE",
            "disable_htf BOOLEAN DEFAULT FALSE",
            "fitness_score NUMERIC DEFAULT 0.0"
        ]
        for col in param_columns:
            cursor.execute(f"ALTER TABLE strategy_parameters ADD COLUMN IF NOT EXISTS {col};")
            
        conn.commit()
        cursor.close()
        logger.info("Database schema verified and updated successfully.")
    except Exception as e:
        logger.error(f"Schema auto-update error: {e}")
    finally:
        release_db_connection(conn)


def calculate_pnl(direction: str, entry_price: float, current_price: float, quantity: float, account_balance: float = 100.0) -> tuple:
    dir_clean = str(direction).strip().upper()
    
    if dir_clean in ["BUY", "LONG"]:
        pnl_usd = (current_price - entry_price) * quantity
    elif dir_clean in ["SELL", "SHORT"]:
        pnl_usd = (entry_price - current_price) * quantity
    else:
        pnl_usd = 0.0

    pnl_pct = (pnl_usd / account_balance) * 100.0 if account_balance > 0 else 0.0
    outcome = "WIN" if pnl_usd > 0 else ("LOSS" if pnl_usd < 0 else "BREAKEVEN")
    return round(pnl_usd, 4), round(pnl_pct, 4), outcome


def send_telegram_notification(message: str) -> bool:
    bot_token = TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return False
        
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except requests.exceptions.Timeout:
        logger.debug("Telegram alert timed out (suppressed).")
        return False
    except Exception as e:
        logger.debug(f"Failed to send Telegram notification: {e}")
        return False


def close_trade_manually(trade_id: int, exit_price: float, reason: str = "MANUAL_CLOSE") -> bool:
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
            release_db_connection(conn)
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
        logger.error(f"[Manual Close Error]: {e}")
        return False
    finally:
        release_db_connection(conn)


def normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("-", "").upper()


def to_mexc_ws_symbol(symbol: str) -> str:
    clean = normalize_symbol(symbol)
    if clean.endswith("USDT"):
        return f"{clean[:-4]}_USDT"
    return clean


def check_daily_circuit_breaker(max_loss_pct: float = 3.0, account_balance: float = 100.0) -> bool:
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT SUM(pnl_usd) FROM trade_setups
            WHERE status = 'CLOSED' 
              AND closed_at >= CURRENT_DATE;
        """)
        row = cursor.fetchone()
        cursor.close()
        
        daily_pnl = float(row[0]) if row and row[0] is not None else 0.0
        max_loss_usd = -1 * abs(account_balance * (max_loss_pct / 100.0))
        return daily_pnl <= max_loss_usd
    except Exception as e:
        logger.error(f"[Circuit Breaker Error]: {e}")
        return False
    finally:
        release_db_connection(conn)