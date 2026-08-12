import asyncio
import json
import logging
import os
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
import websockets

from common import (
    calculate_pnl,
    ensure_schema_updated,
    get_db_connection,
    send_telegram_notification,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ensure_schema_updated()

def fetch_active_trades():
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, pair, direction, entry_price, stop_loss, take_profit, position_size, account_balance
                FROM trade_setups
                WHERE status = 'EXECUTED';
            """)
            return cur.fetchall()
    except Exception as e:
        logging.error(f"Error fetching active trades: {e}")
        return []
    finally:
        conn.close()

def execute_auto_settlement(trade_id, exit_price, reason):
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, pair, direction, entry_price, position_size, account_balance
                FROM trade_setups WHERE id = %s AND status = 'EXECUTED';
            """, (trade_id,))
            trade = cur.fetchone()
            if not trade:
                return

            pnl_usd, pnl_pct, outcome = calculate_pnl(
                trade["direction"], float(trade["entry_price"]), exit_price, float(trade["position_size"]), float(trade["account_balance"] or 10000.0)
            )

            cur.execute("""
                UPDATE trade_setups
                SET exit_price = %s, pnl_usd = %s, pnl_pct = %s, outcome = %s,
                    status = 'CLOSED', trade_state = 'CLOSED', closed_at = CURRENT_TIMESTAMP
                WHERE id = %s;
            """, (round(exit_price, 5), round(pnl_usd, 2), round(pnl_pct, 2), outcome, trade_id))
            conn.commit()

            emoji = "🟢" if outcome == "WIN" else "🔴"
            send_telegram_notification(f"<b>{emoji} REAL-TIME SETTLEMENT ({reason})</b>\n\n<b>Trade ID:</b> <code>#{trade_id}</code>\n<b>PnL:</b> ${pnl_usd:,.2f} ({outcome})")
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed auto-settlement for trade #{trade_id}: {e}")
    finally:
        conn.close()

async def monitor_prices():
    while True:
        trades = fetch_active_trades()
        if not trades:
            await asyncio.sleep(10)
            continue

        crypto_trades = [t for t in trades if t["pair"].replace("/", "").upper().endswith("USDT")]
        if not crypto_trades:
            await asyncio.sleep(10)
            continue

        active_pairs = list(set([t["pair"].replace("/", "").lower() for t in crypto_trades]))
        stream_names = "/".join([f"{p}@ticker" for p in active_pairs])
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_names}"

        try:
            async with websockets.connect(ws_url) as websocket:
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    data = json.loads(message)
                    symbol = data.get("s")
                    current_price = float(data.get("c"))

                    for trade in crypto_trades:
                        if trade["pair"].replace("/", "").upper() != symbol:
                            continue
                        if trade["direction"] == "LONG":
                            if current_price >= float(trade["take_profit"]):
                                execute_auto_settlement(trade["id"], float(trade["take_profit"]), "TAKE_PROFIT")
                                break
                            elif current_price <= float(trade["stop_loss"]):
                                execute_auto_settlement(trade["id"], float(trade["stop_loss"]), "STOP_LOSS")
                                break
                        elif trade["direction"] == "SHORT":
                            if current_price <= float(trade["take_profit"]):
                                execute_auto_settlement(trade["id"], float(trade["take_profit"]), "TAKE_PROFIT")
                                break
                            elif current_price >= float(trade["stop_loss"]):
                                execute_auto_settlement(trade["id"], float(trade["stop_loss"]), "STOP_LOSS")
                                break
        except Exception:
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(monitor_prices())