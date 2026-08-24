import asyncio
import json
import os
import websockets
import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

from common import (
    calculate_pnl,
    ensure_schema_updated,
    get_db_connection,
    release_db_connection,
    send_telegram_notification,
    logger,
    to_mexc_ws_symbol
)
from live_executor import LiveExecutionEngine
from trade_manager import TradeManager
import strategy
from strategy import safe_float

load_dotenv()
ensure_schema_updated()
execution_engine = LiveExecutionEngine()
trade_manager = TradeManager()

def _sync_fetch_active_trades():
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, pair, direction, entry_price, stop_loss, take_profit, position_size, account_balance, trade_state
                FROM trade_setups
                WHERE status = 'EXECUTED' AND trade_state != 'CLOSED';
            """)
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Error fetching active trades: {e}")
        return []
    finally:
        release_db_connection(conn)

async def fetch_active_trades():
    return await asyncio.to_thread(_sync_fetch_active_trades)

def _sync_update_db_stop_loss(trade_id: int, new_sl: float, new_state: str):
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trade_setups
                SET stop_loss = %s, trade_state = %s
                WHERE id = %s;
            """, (safe_float(new_sl), new_state, trade_id))
            conn.commit()
            logger.info(f"Updated DB Trade #{trade_id} -> SL: ${safe_float(new_sl):.5f} | State: {new_state}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update stop loss in DB for trade #{trade_id}: {e}")
    finally:
        release_db_connection(conn)

async def update_db_stop_loss(trade_id: int, new_sl: float, new_state: str):
    await asyncio.to_thread(_sync_update_db_stop_loss, trade_id, new_sl, new_state)

def _sync_execute_auto_settlement(trade, trigger_price, reason):
    trade_id = trade["id"]
    pair = trade["pair"]
    direction = trade["direction"]
    position_size = safe_float(trade.get("position_size"))

    exit_res = execution_engine.close_live_position_mexc(pair, position_side="BOTH")
    raw_exit = exit_res.get("exit_price") if isinstance(exit_res, dict) else None
    actual_exit_price = safe_float(raw_exit) if raw_exit and safe_float(raw_exit) > 0 else safe_float(trigger_price)

    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            pnl_usd, pnl_pct, outcome = calculate_pnl(
                direction, safe_float(trade.get("entry_price")), actual_exit_price, position_size, safe_float(trade.get("account_balance"), default=100.0)
            )

            cur.execute("""
                UPDATE trade_setups
                SET exit_price = %s, pnl_usd = %s, pnl_pct = %s, outcome = %s,
                    status = 'CLOSED', trade_state = 'CLOSED', closed_at = CURRENT_TIMESTAMP
                WHERE id = %s;
            """, (round(actual_exit_price, 5), round(pnl_usd, 2), round(pnl_pct, 2), outcome, trade_id))
            conn.commit()

            emoji = "🟢" if outcome == "WIN" else "🔴"
            send_telegram_notification(
                f"<b>{emoji} MEXC LIVE SETTLEMENT ({reason})</b>\n\n"
                f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                f"<b>Pair:</b> <code>{pair}</code>\n"
                f"<b>Direction:</b> <code>{direction}</code>\n"
                f"<b>Entry:</b> ${safe_float(trade.get('entry_price')):.5f}\n"
                f"<b>Executed Exit:</b> ${actual_exit_price:.5f}\n"
                f"<b>PnL:</b> ${pnl_usd:,.2f} ({pnl_pct:.2f}%) | <b>Outcome:</b> {outcome}"
            )
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed auto-settlement DB update for trade #{trade_id}: {e}")
    finally:
        release_db_connection(conn)

async def execute_auto_settlement(trade, trigger_price, reason):
    await asyncio.to_thread(_sync_execute_auto_settlement, trade, trigger_price, reason)

async def send_mexc_ping(websocket):
    while True:
        try:
            await asyncio.sleep(20)
            await websocket.send(json.dumps({"method": "ping"}))
        except Exception:
            break

def _sync_get_technical_indicators(pair: str):
    try:
        if hasattr(strategy, "load_symbol_config"):
            cfg = strategy.load_symbol_config(pair)
        else:
            cfg = {"tema_period": 200}

        raw_df = strategy.fetch_klines(symbol=pair, timeframe="1h", limit=250)
        df = pd.DataFrame(raw_df) if not isinstance(raw_df, pd.DataFrame) else raw_df
        if not df.empty and len(df) >= 200:
            df["tema"] = strategy.calc_tema(df["close"], period=int(cfg.get("tema_period", 200)))
            df["atr"] = strategy.calc_atr(df, period=14)
            latest = df.iloc[-1]
            return safe_float(latest["tema"]), safe_float(latest["atr"])
    except Exception as e:
        logger.error(f"Error fetching indicator metrics for {pair}: {e}")
    return None, 0.0

async def get_technical_indicators(pair: str):
    return await asyncio.to_thread(_sync_get_technical_indicators, pair)

async def monitor_prices():
    MEXC_CONTRACT_WS_URL = "wss://contract.mexc.com/edge"
    
    while True:
        trades = await fetch_active_trades()
        if not trades:
            await asyncio.sleep(3)
            continue

        active_ws_symbols = list(set([to_mexc_ws_symbol(t["pair"]) for t in trades]))

        try:
            async with websockets.connect(MEXC_CONTRACT_WS_URL, ping_interval=None) as websocket:
                ping_task = asyncio.create_task(send_mexc_ping(websocket))

                for ws_sym in active_ws_symbols:
                    sub_payload = {
                        "method": "sub.ticker",
                        "param": {"symbol": ws_sym}
                    }
                    await websocket.send(json.dumps(sub_payload))
                
                logger.info(f"Subscribed to MEXC Futures WebSocket ticker feeds for: {active_ws_symbols}")

                last_trade_poll = asyncio.get_running_loop().time()

                while True:
                    if asyncio.get_running_loop().time() - last_trade_poll > 10:
                        trades = await fetch_active_trades()
                        new_ws_symbols = list(set([to_mexc_ws_symbol(t["pair"]) for t in trades]))
                        if set(new_ws_symbols) != set(active_ws_symbols):
                            logger.info("Active trade set changed! Re-subscribing Futures WebSocket feed...")
                            break
                        last_trade_poll = asyncio.get_running_loop().time()

                    msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    
                    try:
                        data = json.loads(msg)
                        
                        if data.get("channel") == "pong" or data.get("data") == "pong":
                            continue

                        channel = data.get("channel", "")
                        if channel == "push.ticker" and "data" in data and data["data"]:
                            ticker_data = data["data"]
                            
                            raw_price = ticker_data.get("p") or ticker_data.get("lastPrice") or ticker_data.get("close")
                            current_price = safe_float(raw_price)
                            if current_price <= 0:
                                continue

                            raw_ws_symbol = str(ticker_data.get("symbol", "")).upper()

                            for trade in trades:
                                trade_ws_format = to_mexc_ws_symbol(trade["pair"])
                                if trade_ws_format != raw_ws_symbol:
                                    continue

                                tema_val, atr_val = await get_technical_indicators(trade["pair"])
                                
                                # Safe fallback if TEMA fails to calculate
                                safe_tema = safe_float(tema_val, default=current_price) if tema_val is not None else current_price

                                candle_data = pd.Series({
                                    'close': safe_float(current_price), 
                                    'tema': safe_tema, 
                                    'atr': safe_float(atr_val)
                                })
                                
                                managed_res = trade_manager.process_trade(pd.Series(trade), candle_data)
                                action = managed_res.get("action")

                                if action == "CLOSE_SL":
                                    await execute_auto_settlement(trade, managed_res["exit_price"], "STOP_LOSS")
                                    trades = await fetch_active_trades()
                                    break
                                elif action == "CLOSE_TP":
                                    await execute_auto_settlement(trade, managed_res["exit_price"], "TAKE_PROFIT")
                                    trades = await fetch_active_trades()
                                    break
                                elif action == "UPDATE_SL":
                                    await update_db_stop_loss(trade["id"], managed_res["new_sl"], managed_res["new_state"])
                                    if "msg" in managed_res and managed_res["msg"]:
                                        send_telegram_notification(managed_res["msg"])
                                    trades = await fetch_active_trades()

                    except (ValueError, TypeError) as e:
                        logger.warning(f"Ignored invalid ticker frame: {e}")
                    except json.JSONDecodeError:
                        continue

        except Exception as err:
            logger.warning(f"MEXC Futures WebSocket Connection Interrupted ({err}). Reconnecting in 3 seconds...")
            await asyncio.sleep(3)
        finally:
            if 'ping_task' in locals() and not ping_task.done():
                ping_task.cancel()

if __name__ == "__main__":
    asyncio.run(monitor_prices())