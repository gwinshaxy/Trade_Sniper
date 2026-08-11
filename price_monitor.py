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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

ensure_schema_updated()

def fetch_active_trades():
    """Fetch all open trades requiring real-time price monitoring."""
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
    """Settles the trade using common.calculate_pnl and dispatches Telegram HTML alerts."""
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

            pair = trade["pair"]
            direction = trade["direction"]
            entry = float(trade["entry_price"])
            pos_size = float(trade["position_size"])
            default_balance = float(os.getenv("ACCOUNT_BALANCE", 10000.0))
            initial_balance = float(trade["account_balance"]) if trade["account_balance"] else default_balance

            # Calculate Realized PnL via common module
            pnl_usd, pnl_pct, outcome = calculate_pnl(
                direction, entry, exit_price, pos_size, initial_balance
            )

            cur.execute("""
                UPDATE trade_setups
                SET exit_price = %s, pnl_usd = %s, pnl_pct = %s, outcome = %s,
                    status = 'CLOSED', trade_state = 'CLOSED', closed_at = CURRENT_TIMESTAMP
                WHERE id = %s;
            """, (round(exit_price, 5), round(pnl_usd, 2), round(pnl_pct, 2), outcome, trade_id))
            conn.commit()

            emoji = "🟢" if outcome == "WIN" else "🔴"
            msg = (
                f"<b>{emoji} REAL-TIME SETTLEMENT ({reason})</b>\n\n"
                f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                f"<b>Pair:</b> <code>{pair}</code>\n"
                f"<b>Exit Price:</b> ${exit_price:,.2f}\n"
                f"<b>PnL ($):</b> ${pnl_usd:,.2f}\n"
                f"<b>PnL (%):</b> {pnl_pct:.2f}%\n"
                f"<b>Outcome:</b> {outcome}"
            )
            send_telegram_notification(msg)
            logging.info(
                f"⚡ AUTO-SETTLEMENT ({reason}): Trade #{trade_id} closed at {exit_price} | PnL: ${pnl_usd:.2f} ({outcome})"
            )
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed auto-settlement for trade #{trade_id}: {e}")
    finally:
        conn.close()

async def monitor_prices():
    """Connects to Binance WebSockets and monitors prices against active trade SL/TP."""
    while True:
        trades = fetch_active_trades()
        if not trades:
            await asyncio.sleep(10)
            continue

        crypto_trades = [
            t for t in trades
            if t["pair"].replace("/", "").upper().endswith("USDT")
        ]
        if not crypto_trades:
            await asyncio.sleep(10)
            continue

        active_pairs = list(
            set([t["pair"].replace("/", "").lower() for t in crypto_trades])
        )
        stream_names = "/".join([f"{p}@ticker" for p in active_pairs])
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_names}"

        try:
            logging.info(f"📡 Connecting to Binance WebSocket stream for pairs: {active_pairs}")
            async with websockets.connect(ws_url) as websocket:
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    data = json.loads(message)

                    symbol = data.get("s")
                    current_price = float(data.get("c"))

                    for trade in crypto_trades:
                        trade_symbol = trade["pair"].replace("/", "").upper()
                        if trade_symbol != symbol:
                            continue

                        trade_id = trade["id"]
                        direction = trade["direction"]
                        sl = float(trade["stop_loss"])
                        tp = float(trade["take_profit"])

                        if direction == "LONG":
                            if current_price >= tp:
                                execute_auto_settlement(trade_id, tp, "TAKE_PROFIT")
                                break
                            elif current_price <= sl:
                                execute_auto_settlement(trade_id, sl, "STOP_LOSS")
                                break
                        elif direction == "SHORT":
                            if current_price <= tp:
                                execute_auto_settlement(trade_id, tp, "TAKE_PROFIT")
                                break
                            elif current_price >= sl:
                                execute_auto_settlement(trade_id, sl, "STOP_LOSS")
                                break

        except (asyncio.TimeoutError, websockets.ConnectionClosed):
            logging.warning("WebSocket connection reset. Reconnecting...")
            await asyncio.sleep(4)
        except Exception as e:
            logging.error(f"Stream error: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(monitor_prices())