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
    logger
)
from live_executor import LiveExecutionEngine
from trade_manager import TradeManager
import strategy
from strategy import safe_float

load_dotenv()
ensure_schema_updated()
execution_engine = LiveExecutionEngine()
trade_manager = TradeManager()

indicator_cache = {}

def _sync_fetch_active_trades():
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, pair, direction, entry_price, stop_loss, take_profit, position_size, account_balance, trade_state
                FROM trade_setups
                WHERE trade_state = 'OPEN';
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

    exit_res = execution_engine.sell_spot_mexc(pair, amount=position_size)
    
    raw_exit = exit_res.get("fill_price") or exit_res.get("price") if isinstance(exit_res, dict) else None
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
                f"<b>{emoji} MEXC LIVE SPOT SETTLEMENT ({reason})</b>\n\n"
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
    """Sends JSON PING frame and protocol ping every 10 seconds."""
    while True:
        try:
            await asyncio.sleep(10)
            await websocket.send(json.dumps({"method": "PING"}))
            await websocket.ping()
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            break
        except Exception as e:
            logger.debug(f"Application Ping error: {e}")
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

async def indicator_refresh_loop():
    global indicator_cache
    while True:
        try:
            trades = await fetch_active_trades()
            pairs = list(set([t["pair"] for t in trades]))
            for pair in pairs:
                tema, atr = await asyncio.to_thread(_sync_get_technical_indicators, pair)
                indicator_cache[pair] = {"tema": tema, "atr": atr}
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in indicator refresh background task: {e}")
        await asyncio.sleep(30)

async def manage_subscriptions(websocket, active_trades_ref, subscribed_symbols):
    msg_id = 100
    while True:
        try:
            await asyncio.sleep(10)
            trades = await fetch_active_trades()
            active_trades_ref["trades"] = trades
            current_symbols = set(t["pair"].replace("/", "").upper() for t in trades)
            
            new_symbols = current_symbols - subscribed_symbols
            for sym in new_symbols:
                msg_id += 1
                sub_msg = {
                    "method": "SUBSCRIPTION",
                    "params": [f"spot@public.deals.v3.api@{sym}"],
                    "id": msg_id
                }
                await websocket.send(json.dumps(sub_msg))
                subscribed_symbols.add(sym)
            
            removed_symbols = subscribed_symbols - current_symbols
            for sym in removed_symbols:
                msg_id += 1
                unsub_msg = {
                    "method": "UNSUBSCRIPTION",
                    "params": [f"spot@public.deals.v3.api@{sym}"],
                    "id": msg_id
                }
                await websocket.send(json.dumps(unsub_msg))
                subscribed_symbols.remove(sym)

        except (asyncio.CancelledError, websockets.ConnectionClosed):
            break
        except Exception as e:
            logger.warning(f"Error managing subscriptions: {e}")

async def monitor_prices():
    MEXC_SPOT_WS_URL = "wss://wbs.mexc.com/ws"
    
    while True:
        trades = await fetch_active_trades()
        if not trades:
            await asyncio.sleep(3)
            continue

        subscribed_symbols = set(t["pair"].replace("/", "").upper() for t in trades)
        active_trades_ref = {"trades": trades}

        ping_task = None
        sub_task = None

        try:
            async with websockets.connect(
                MEXC_SPOT_WS_URL, 
                ping_interval=None, 
                close_timeout=5
            ) as websocket:
                
                ping_task = asyncio.create_task(send_mexc_ping(websocket))
                sub_task = asyncio.create_task(manage_subscriptions(websocket, active_trades_ref, subscribed_symbols))

                for idx, symbol in enumerate(list(subscribed_symbols), start=1):
                    sub_param = {
                        "method": "SUBSCRIPTION",
                        "params": [f"spot@public.deals.v3.api@{symbol}"],
                        "id": idx
                    }
                    await websocket.send(json.dumps(sub_param))
                
                logger.info(f"Subscribed to MEXC Spot WS Streams for: {list(subscribed_symbols)}")

                while True:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=25.0)
                    
                    try:
                        data = json.loads(msg)
                        
                        if data.get("msg") == "PONG" or data.get("code") == "PONG":
                            continue

                        if "d" in data and "deals" in data["d"]:
                            raw_ws_symbol = str(data.get("s", "")).upper()

                            for deal in data["d"]["deals"]:
                                current_price = safe_float(deal.get("p"))
                                if current_price <= 0:
                                    continue

                                for trade in list(active_trades_ref["trades"]):
                                    db_sym = trade["pair"].replace("/", "").upper()
                                    if db_sym != raw_ws_symbol:
                                        continue

                                    cached_ind = indicator_cache.get(trade["pair"], {})
                                    tema_val = cached_ind.get("tema")
                                    atr_val = cached_ind.get("atr", 0.0)
                                    safe_tema = safe_float(tema_val, default=current_price) if tema_val is not None else current_price

                                    candle_data = pd.Series({
                                        'close': safe_float(current_price), 
                                        'tema': safe_tema, 
                                        'atr': safe_float(atr_val)
                                    })
                                    
                                    managed_res = trade_manager.process_trade(pd.Series(trade), candle_data)
                                    action = managed_res.get("action")

                                    if action in ("CLOSE_SL", "CLOSE_TP"):
                                        reason = "STOP_LOSS" if action == "CLOSE_SL" else "TAKE_PROFIT"
                                        await execute_auto_settlement(trade, managed_res["exit_price"], reason)
                                        active_trades_ref["trades"] = [t for t in active_trades_ref["trades"] if t["id"] != trade["id"]]
                                    elif action == "UPDATE_SL":
                                        await update_db_stop_loss(trade["id"], managed_res["new_sl"], managed_res["new_state"])
                                        if managed_res.get("msg"):
                                            send_telegram_notification(managed_res["msg"])

                    except (ValueError, TypeError) as e:
                        logger.warning(f"Ignored invalid ticker frame: {e}")
                    except json.JSONDecodeError:
                        continue

        except Exception as err:
            logger.warning(f"MEXC Spot WebSocket Interrupted ({err}). Reconnecting in 3 seconds...")
            await asyncio.sleep(3)
        finally:
            for task in (ping_task, sub_task):
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

async def main():
    refresh_task = asyncio.create_task(indicator_refresh_loop())
    try:
        await monitor_prices()
    finally:
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    asyncio.run(main())