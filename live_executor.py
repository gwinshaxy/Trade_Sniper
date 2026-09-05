import asyncio
import logging
import time
from typing import Dict, Any, Optional
import ccxt

from config import BYBIT_API_KEY, BYBIT_SECRET_KEY, BYBIT_TESTNET

from common import (
    get_db_connection,
    release_db_connection,
    send_telegram_notification,
    set_asset_cooldown,
    finalize_trade_in_db,
    logger
)
from event_bus import event_bus

MIN_DUST_THRESHOLD = 0.001
MAX_SLIPPAGE_PCT = 0.002       # 0.2% max slippage ceiling
MAX_ALLOWED_SPREAD_PCT = 0.003 # 0.3% max spread threshold


def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def format_ccxt_futures_symbol(symbol: str) -> str:
    """Consistently converts raw/dirty symbols to CCXT Bybit Linear Futures format (e.g., 'XRP/USDT:USDT')."""
    if ":" in symbol:
        return symbol
    raw = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
    if raw.endswith("USDTUSDT"):
        raw = raw[:-4]
    if raw.endswith("USDT"):
        base = raw[:-4]
        return f"{base}/USDT:USDT"
    return f"{raw}/USDT:USDT"


class BybitFuturesLiveExecutor:
    def __init__(self):
        self.exchange = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'recvWindow': 10000,  # Increase tolerance to 10 seconds for network/clock jitter
            }
        })
        if BYBIT_TESTNET:
            self.exchange.set_sandbox_mode(True)

        try:
            self.exchange.load_markets()
        except Exception as e:
            logger.warning(f"Could not refresh market structures on init: {e}")

    async def get_futures_position_async(self, symbol: str) -> Dict[str, Any]:
        """Non-blocking wrapper around get_futures_position."""
        return await asyncio.to_thread(self.get_futures_position, symbol)

    def get_futures_position(self, symbol: str) -> dict:
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        try:
            positions = self.exchange.fetch_positions([ccxt_symbol])
            for pos in positions:
                contracts = safe_float(pos.get('contracts', 0.0))
                if contracts > 0:
                    return {
                        "symbol": symbol,
                        "side": pos.get('side', '').upper(),
                        "contracts": contracts,
                        "entry_price": safe_float(pos.get('entryPrice', 0.0)),
                        "leverage": safe_float(pos.get('leverage', 1.0)),
                        "unrealized_pnl": safe_float(pos.get('unrealizedPnl', 0.0)),
                        "error": False
                    }
            return {
                "symbol": symbol,
                "side": "NONE",
                "contracts": 0.0,
                "entry_price": 0.0,
                "leverage": 1.0,
                "unrealized_pnl": 0.0,
                "error": False
            }
        except Exception as e:
            logger.error(f"Failed to fetch futures position for {ccxt_symbol}: {e}")
            return {
                "symbol": symbol,
                "side": "NONE",
                "contracts": 0.0,
                "entry_price": 0.0,
                "leverage": 1.0,
                "unrealized_pnl": 0.0,
                "error": True  # Flag error so reconciler does not count zero
            }

    def fetch_available_usdt_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance({'type': 'linear'})
            usdt_free = safe_float(balance.get('USDT', {}).get('free', 0.0))
            if usdt_free == 0.0:
                usdt_free = safe_float(balance.get('free', {}).get('USDT', 0.0))
            return usdt_free
        except Exception as e:
            logger.error(f"Failed to fetch live USDT Futures balance: {e}")
            return 0.0

    def get_available_usdt_balance(self) -> float:
        return self.fetch_available_usdt_balance()

    def fetch_ticker_data(self, symbol: str) -> dict:
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        try:
            ticker = self.exchange.fetch_ticker(ccxt_symbol)
            bid = safe_float(ticker.get('bid'))
            ask = safe_float(ticker.get('ask'))
            last = safe_float(ticker.get('last'))
            
            spread_pct = (ask - bid) / bid if (bid > 0 and ask > 0) else 0.0

            return {
                "bid": bid,
                "ask": ask,
                "last": last,
                "spread_pct": spread_pct
            }
        except Exception as e:
            logger.error(f"[{ccxt_symbol}] Failed to fetch ticker data: {e}")
            return {"bid": 0.0, "ask": 0.0, "last": 0.0, "spread_pct": 0.0}

    def fetch_ticker_price(self, symbol: str) -> float:
        data = self.fetch_ticker_data(symbol)
        if data["bid"] > 0 and data["ask"] > 0:
            return (data["bid"] + data["ask"]) / 2.0
        return data["last"]

    async def execute_order_async(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Executes market or limit futures orders asynchronously without blocking event loops."""
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        order_type = 'limit' if price else 'market'

        def _sync_create_order():
            return self.exchange.create_order(
                symbol=ccxt_symbol,
                type=order_type,
                side=side.lower(),
                amount=amount,
                price=price
            )

        try:
            logger.info(f"Submitting {order_type.upper()} {side.upper()} order for {amount} {ccxt_symbol}...")
            order_result = await asyncio.to_thread(_sync_create_order)
            logger.info(f"Order executed successfully: ID {order_result.get('id')}")
            return order_result
        except Exception as e:
            logger.error(f"Failed to execute CCXT order for {ccxt_symbol}: {e}")
            raise e

    def execute_live_order(
        self, 
        pair: str, 
        direction: str = "BUY", 
        entry_price: float = 0.0, 
        stop_loss: float = 0.0, 
        take_profit: float = 0.0,
        amount_usd: float = 25.0,
        leverage: int = 10
    ) -> bool:
        clean_direction = direction.upper().strip()
        
        if clean_direction not in ["BUY", "LONG", "SELL", "SHORT"]:
            logger.warning(f"[{pair}] Invalid order direction: {clean_direction}")
            return False

        res = self.order_futures_bybit(
            symbol=pair, 
            direction=clean_direction, 
            amount_usd=amount_usd, 
            stop_loss=stop_loss, 
            take_profit=take_profit,
            leverage=leverage
        )
        if res.get("status") == "SUCCESS":
            logger.info(f"[{pair}] Live Bybit futures order executed successfully: Order ID {res.get('order_id')}")
            return True
        else:
            logger.error(f"[{pair}] Live Bybit futures order execution failed: {res.get('error') or res.get('reason')}")
            return False

    def order_futures_bybit(
        self, 
        symbol: str, 
        direction: str, 
        amount_usd: float, 
        stop_loss: float = 0.0, 
        take_profit: float = 0.0,
        leverage: int = 10
    ) -> dict:
        ccxt_symbol = format_ccxt_futures_symbol(symbol)

        dir_clean = direction.upper().strip()
        is_long = dir_clean in ["BUY", "LONG"]
        side = 'buy' if is_long else 'sell'

        pos_info = self.get_futures_position(ccxt_symbol)
        if pos_info["contracts"] > MIN_DUST_THRESHOLD:
            logger.warning(f"[{symbol}] Guard Triggered: Active futures position exists ({pos_info['contracts']} contracts {pos_info['side']}). Blocking execution.")
            return {"status": "SKIPPED", "reason": "Active futures position detected on exchange"}

        available_usdt = self.fetch_available_usdt_balance()
        if available_usdt <= 1.0:
            return {"status": "FAILED", "error": f"Insufficient live USDT balance: ${available_usdt:.2f}"}

        ticker_data = self.fetch_ticker_data(ccxt_symbol)
        ref_price = ticker_data["ask"] if is_long else ticker_data["bid"]
        spread_pct = ticker_data["spread_pct"]

        if ref_price <= 0:
            return {"status": "FAILED", "error": "Invalid reference ticker price prior to execution"}

        if spread_pct > MAX_ALLOWED_SPREAD_PCT:
            logger.warning(f"[{symbol}] Order Rejected: High Spread detected ({spread_pct * 100:.3f}% > Max allowed {MAX_ALLOWED_SPREAD_PCT * 100:.2f}%).")
            return {"status": "FAILED", "error": f"Spread too high ({spread_pct * 100:.3f}%)"}

        try:
            self.exchange.set_leverage(leverage, ccxt_symbol)
        except Exception as lev_err:
            logger.debug(f"[{symbol}] Leverage setting notice: {lev_err}")

        trade_amount_usd = min(amount_usd, available_usdt * 0.98)
        notional_value = trade_amount_usd * leverage
        
        limit_price = ref_price * (1.0 + MAX_SLIPPAGE_PCT) if is_long else ref_price * (1.0 - MAX_SLIPPAGE_PCT)
        raw_qty = notional_value / ref_price
        
        try:
            formatted_qty = safe_float(self.exchange.amount_to_precision(ccxt_symbol, raw_qty))
            formatted_limit_price = safe_float(self.exchange.price_to_precision(ccxt_symbol, limit_price))
        except Exception:
            formatted_qty = round(raw_qty, 4)
            formatted_limit_price = round(limit_price, 6)

        logger.info(
            f"[{symbol}] Initiating Futures Limit {side.upper()} (IOC): Margin=${trade_amount_usd:.2f} @ {leverage}x "
            f"({formatted_qty} contracts @ Limit: ${formatted_limit_price:.6f})"
        )

        params = {'timeInForce': 'IOC'}
        if stop_loss > 0:
            params['stopLoss'] = safe_float(self.exchange.price_to_precision(ccxt_symbol, stop_loss))
        if take_profit > 0:
            params['takeProfit'] = safe_float(self.exchange.price_to_precision(ccxt_symbol, take_profit))

        try:
            order = self.exchange.create_order(
                symbol=ccxt_symbol,
                type='limit',
                side=side,
                amount=formatted_qty,
                price=formatted_limit_price,
                params=params
            )

            order_id = order.get("id")
            fill_price = safe_float(order.get("average") or order.get("price"))
            executed_qty = safe_float(order.get("filled") or order.get("amount"))

            if (fill_price <= 0 or executed_qty <= 0) and order_id:
                time.sleep(0.4)
                try:
                    fetched_order = self.exchange.fetch_order(order_id, ccxt_symbol, params={'acknowledged': True})
                    fill_price = safe_float(fetched_order.get("average") or fetched_order.get("price"))
                    executed_qty = safe_float(fetched_order.get("filled") or fetched_order.get("amount"))
                except Exception as fetch_err:
                    logger.warning(f"[{symbol}] Post-order lookup error: {fetch_err}")

                if executed_qty <= 0:
                    pos_check = self.get_futures_position(ccxt_symbol)
                    if pos_check["contracts"] > MIN_DUST_THRESHOLD:
                        executed_qty = pos_check["contracts"]
                        fill_price = pos_check["entry_price"] if pos_check["entry_price"] > 0 else ref_price
                        logger.info(f"[{symbol}] Confirmed IOC fill via live position check: {executed_qty} contracts @ ${fill_price:.6f}")

            if executed_qty <= 0:
                logger.warning(f"[{symbol}] IOC order unfilled/cancelled due to slippage ceiling.")
                return {"status": "FAILED", "error": "Limit IOC order unfilled due to slippage guard."}

            if fill_price <= 0:
                fill_price = ref_price

            logger.info(f"[{symbol}] Futures Order Executed: Side {side.upper()}, Price ${fill_price:.6f}, Qty {executed_qty}")

            if stop_loss > 0 and 'stopLoss' not in params:
                event_bus.arm_local_sl_guard(symbol, dir_clean, executed_qty, stop_loss)

            return {
                "status": "SUCCESS",
                "order_id": order_id,
                "fill_price": fill_price,
                "executed_qty": executed_qty,
                "cost_usd": trade_amount_usd,
                "raw_order": order
            }

        except Exception as e:
            logger.error(f"[{symbol}] Bybit Futures Order Exception: {e}")
            return {"status": "FAILED", "error": str(e)}

    def close_live_position_bybit(self, symbol: str, position_size: float, current_price: float, outcome: str = "CLOSE") -> dict:
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        pos_info = self.get_futures_position(ccxt_symbol)
        contracts = pos_info["contracts"]
        current_side = pos_info["side"]
        entry_price = pos_info["entry_price"]

        conn = get_db_connection()
        trade_id = None
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id FROM trade_setups 
                        WHERE UPPER(REPLACE(REPLACE(pair, '/', ''), '_', '')) = %s 
                          AND trade_state IN ('OPEN', 'EXECUTED', 'BE_LOCKED', 'TRAILING')
                        ORDER BY id DESC LIMIT 1;
                    """, (symbol.replace("/", "").upper(),))
                    row = cur.fetchone()
                    if row:
                        trade_id = row[0]
            finally:
                release_db_connection(conn)

        if contracts < MIN_DUST_THRESHOLD:
            set_asset_cooldown(symbol, hours=2)
            event_bus.disarm_local_sl_guard(symbol)

            if trade_id:
                finalize_trade_in_db(trade_id, current_price, 0.0, 0.0, "BREAKEVEN")

            return {
                "status": "FORCE_CLOSED_DB_ONLY",
                "order_id": "GHOST_POSITION_DB_CLOSE",
                "exit_price": current_price,
                "pnl_usd": 0.0,
                "pnl_pct": 0.0
            }

        close_side = 'sell' if current_side.upper() in ['BUY', 'LONG'] else 'buy'
        close_qty = safe_float(self.exchange.amount_to_precision(ccxt_symbol, min(position_size, contracts)))

        try:
            order = self.exchange.create_order(
                symbol=ccxt_symbol,
                type='market',
                side=close_side,
                amount=close_qty,
                params={'reduceOnly': True}
            )
            exit_price = safe_float(order.get("average") or order.get("price")) or current_price

            from common import calculate_pnl
            pnl_usd, pnl_pct, computed_outcome = calculate_pnl(current_side, entry_price, exit_price, close_qty)

            if trade_id:
                finalize_trade_in_db(trade_id, exit_price, pnl_usd, pnl_pct, computed_outcome)

            set_asset_cooldown(symbol, hours=2)
            event_bus.disarm_local_sl_guard(symbol)

            return {
                "status": "SUCCESS",
                "order_id": order.get("id"),
                "exit_price": exit_price,
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_pct,
                "raw_order": order
            }
        except Exception as e:
            logger.error(f"[{symbol}] Market close failed: {e}")
            return {"status": "FAILED", "error": str(e)}


LiveExecutionEngine = BybitFuturesLiveExecutor