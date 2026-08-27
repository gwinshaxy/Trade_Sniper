import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import ccxt.pro as ccxtpro
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

# Global Shared Thread Pool for Async DB Operations
executor = ThreadPoolExecutor(max_workers=5)

# Thread-safe / Event-loop local In-Memory Active Trades Cache
ACTIVE_TRADES_CACHE = []
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

async def refresh_active_trades_cache():
    """Background loop to periodically fetch open trades into the in-memory cache."""
    global ACTIVE_TRADES_CACHE
    loop = asyncio.get_running_loop()
    while True:
        try:
            trades = await loop.run_in_executor(executor, _sync_fetch_active_trades)
            ACTIVE_TRADES_CACHE = trades
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error updating active trade cache: {e}")
        await asyncio.sleep(5)

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
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, _sync_update_db_stop_loss, trade_id, new_sl, new_state)

def _sync_execute_auto_settlement(trade, trigger_price, reason):
    trade_id = trade["id"]
    pair = trade["pair"]
    direction = trade["direction"]
    position_size = safe_float(trade.get("position_size"))

    exit_res = execution_engine.close_live_position_mexc(pair, position_size=position_size, current_price=safe_float(trigger_price), outcome=reason)
    
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
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, _sync_execute_auto_settlement, trade, trigger_price, reason)

def _sync_get_technical_indicators(pair: str):
    try:
        cfg = strategy.load_symbol_config(pair) if hasattr(strategy, "load_symbol_config") else {"tema_period": 200}
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
    loop = asyncio.get_running_loop()
    while True:
        try:
            pairs = list(set([t["pair"] for t in ACTIVE_TRADES_CACHE]))
            for pair in pairs:
                tema, atr = await loop.run_in_executor(executor, _sync_get_technical_indicators, pair)
                indicator_cache[pair] = {"tema": tema, "atr": atr}
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in indicator refresh background task: {e}")
        await asyncio.sleep(30)

async def watch_single_ticker(exchange, symbol: str):
    """Worker task streaming a single MEXC spot ticker via WebSocket."""
    norm_sym = strategy.normalize_symbol(symbol)
    while True:
        try:
            ticker = await exchange.watch_ticker(norm_sym)
            current_price = safe_float(ticker.get('last'))
            if current_price <= 0:
                continue

            # Process in-memory cached trades for this symbol without DB overhead
            for trade in list(ACTIVE_TRADES_CACHE):
                if strategy.normalize_symbol(trade["pair"]) != norm_sym:
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
                elif action == "UPDATE_SL":
                    await update_db_stop_loss(trade["id"], managed_res["new_sl"], managed_res["new_state"])
                    if managed_res.get("msg"):
                        send_telegram_notification(managed_res["msg"])

        except ccxtpro.NetworkError as e:
            logger.warning(f"WebSocket network issue for {symbol} ({e}); retrying...")
            await asyncio.sleep(2)
        except ccxtpro.ExchangeError as e:
            logger.error(f"Exchange error on {symbol}: {e}")
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Unhandled WebSocket error on {symbol}: {e}")
            await asyncio.sleep(5)

async def watch_mexc_tickers_spot(watchlist: list):
    """ISSUE 2 FIX: Spawns per-symbol WebSocket loops compatible with MEXC Spot in CCXT Pro."""
    exchange = ccxtpro.mexc({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    tasks = [asyncio.create_task(watch_single_ticker(exchange, symbol)) for symbol in watchlist]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()
        await exchange.close()

async def main():
    raw_symbols = os.getenv("TRADING_SYMBOLS") or os.getenv("WATCHLIST") or "XRP/USDT"
    watchlist = [s.strip() for s in raw_symbols.split(",") if s.strip()]

    # Initial cache build
    loop = asyncio.get_running_loop()
    global ACTIVE_TRADES_CACHE
    ACTIVE_TRADES_CACHE = await loop.run_in_executor(executor, _sync_fetch_active_trades)

    cache_task = asyncio.create_task(refresh_active_trades_cache())
    refresh_task = asyncio.create_task(indicator_refresh_loop())
    ticker_task = asyncio.create_task(watch_mexc_tickers_spot(watchlist))

    try:
        await asyncio.gather(ticker_task)
    finally:
        cache_task.cancel()
        refresh_task.cancel()
        ticker_task.cancel()
        executor.shutdown(wait=False)

if __name__ == "__main__":
    asyncio.run(main())