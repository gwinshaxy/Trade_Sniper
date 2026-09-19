import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple, List
import ccxt.async_support as ccxt_async
import ccxt
import requests

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

# Clear environment proxy flags preventing 407 WebSocket & HTTP Auth errors
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(proxy_var, None)

# Set Cloudflare Worker Proxy URL
CLOUDFLARE_WORKER_URL = (
    os.getenv("CLOUDFLARE_WORKER_URL")
    or os.getenv("WORKER_URL")
    or "https://bybit-proxy.gspark4u.workers.dev"
)

if CLOUDFLARE_WORKER_URL:
    CLOUDFLARE_WORKER_URL = CLOUDFLARE_WORKER_URL.strip('"').strip("'").rstrip('/')

POLL_INTERVAL_SECONDS = 10  # Preserve Cloudflare Worker free tier limits
MIN_DUST_THRESHOLD = 0.001
MAX_SLIPPAGE_PCT = 0.002       # 0.2% max execution limit offset
MAX_ALLOWED_SPREAD_PCT = 0.003 # 0.3% max spread threshold

ENTRY_TOLERANCE_PCT_LIVE = 0.005    # 0.5% tolerance
ENTRY_TOLERANCE_PCT_TESTNET = 0.05  # 5.0% tolerance for testnet drift


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def format_ccxt_futures_symbol(symbol: str) -> str:
    """Consistently converts raw/dirty symbols to CCXT Bybit Linear Futures format (e.g., 'XRP/USDT:USDT')."""
    if not symbol:
        return ""
    if ":" in symbol:
        return symbol
    raw = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
    if raw.endswith("USDTUSDT"):
        raw = raw[:-4]
    if raw.endswith("USDT"):
        base = raw[:-4]
        return f"{base}/USDT:USDT"
    if raw.endswith("USDC"):
        base = raw[:-4]
        return f"{base}/USDC:USDC"
    return f"{raw}/USDT:USDT"


def force_ccxt_worker_urls(exchange_instance, worker_url: str):
    """
    Recursively overrides all REST API subdomains in a CCXT exchange instance 
    to route entirely through the Cloudflare Worker proxy.
    """
    if not worker_url:
        return
    
    clean_url = worker_url.rstrip('/')
    
    if not hasattr(exchange_instance, 'urls') or 'api' not in exchange_instance.urls:
        exchange_instance.urls['api'] = {
            'public': clean_url,
            'private': clean_url,
        }
        return

    # Direct dictionary assignments
    exchange_instance.urls['api'] = {
        'public': clean_url,
        'private': clean_url,
    }

    # Recursive check across nested sub-endpoint trees (e.g., v5 linear, spot, etc.)
    api_map = exchange_instance.urls.get('api', {})
    if isinstance(api_map, dict):
        for key in list(api_map.keys()):
            if isinstance(api_map[key], dict):
                for subkey in api_map[key]:
                    api_map[key][subkey] = clean_url
            else:
                api_map[key] = clean_url


def fetch_cryptocompare_fallback_kline(symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fallback market data engine routing requests correctly to CryptoCompare API v2."""
    try:
        clean_symbol = symbol.split(":")[0].replace("/", "").replace("_", "").replace("-", "").upper()
        if clean_symbol.endswith("USDT"):
            fsym = clean_symbol[:-4]
            tsym = "USDT"
        elif clean_symbol.endswith("USD"):
            fsym = clean_symbol[:-3]
            tsym = "USD"
        else:
            fsym = clean_symbol
            tsym = "USDT"

        url = "https://min-api.cryptocompare.com/data/v2/histominute"
        params = {
            "fsym": fsym,
            "tsym": tsym,
            "limit": limit,
            "e": "CCCAGG"
        }
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        
        if data.get("Response") == "Success":
            raw_candles = data.get("Data", {}).get("Data", [])
            formatted = []
            for c in raw_candles:
                formatted.append({
                    "timestamp": c.get("time") * 1000,
                    "open": float(c.get("open", 0.0)),
                    "high": float(c.get("high", 0.0)),
                    "low": float(c.get("low", 0.0)),
                    "close": float(c.get("close", 0.0)),
                    "volume": float(c.get("volumeto", 0.0))
                })
            return formatted
        else:
            logger.error(f"[{symbol}] CryptoCompare fallback error: {data.get('Message')}")
            return []
    except Exception as e:
        logger.error(f"[{symbol}] Exception in CryptoCompare fallback: {e}")
        return []


class BybitFuturesLiveExecutor:
    def __init__(self):
        # Initialize CCXT Bybit instance with options
        self.exchange = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',  # 'linear' for USDT Futures
                'recvWindow': 20000,
                'adjustForTimeDifference': True
            }
        })

        self.public_exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear'
            }
        })

        # Apply Fix 2: Force ALL CCXT sub-endpoints through Cloudflare Worker
        if CLOUDFLARE_WORKER_URL:
            logger.info(f"Routing CCXT Execution through Worker: {CLOUDFLARE_WORKER_URL}")
            force_ccxt_worker_urls(self.exchange, CLOUDFLARE_WORKER_URL)
            force_ccxt_worker_urls(self.public_exchange, CLOUDFLARE_WORKER_URL)
        else:
            logger.warning("No CLOUDFLARE_WORKER_URL specified. Operating on direct connection.")

        if BYBIT_TESTNET:
            self.exchange.set_sandbox_mode(True)
            self.public_exchange.set_sandbox_mode(True)

        try:
            self.public_exchange.load_markets()
        except Exception as e:
            logger.warning(f"Could not refresh public market structures on init via proxy: {e}")

    def format_ccxt_futures_symbol(self, raw_symbol: str) -> str:
        return format_ccxt_futures_symbol(raw_symbol)

    async def fetch_ticker_direct_async(self, symbol: str) -> float:
        """Asynchronously fetches current ticker price via Cloudflare Worker."""
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        try:
            async_config = {'enableRateLimit': True, 'options': {'defaultType': 'linear'}}
            async_public = ccxt_async.bybit(async_config)
            if CLOUDFLARE_WORKER_URL:
                force_ccxt_worker_urls(async_public, CLOUDFLARE_WORKER_URL)
            if BYBIT_TESTNET:
                async_public.set_sandbox_mode(True)
            ticker = await async_public.fetch_ticker(ccxt_symbol)
            await async_public.close()
            return safe_float(ticker.get('last') or ticker.get('close'))
        except Exception as e:
            logger.error(f"[{ccxt_symbol}] Direct async ticker fetch error: {e}")
            return 0.0

    async def fetch_tickers_batch_async(self, symbols: List[str]) -> Dict[str, float]:
        """Asynchronously fetches multiple tickers via Worker proxy."""
        formatted_symbols = [format_ccxt_futures_symbol(s) for s in symbols]
        price_map = {}
        try:
            async_config = {'enableRateLimit': True, 'options': {'defaultType': 'linear'}}
            async_public = ccxt_async.bybit(async_config)
            if CLOUDFLARE_WORKER_URL:
                force_ccxt_worker_urls(async_public, CLOUDFLARE_WORKER_URL)
            if BYBIT_TESTNET:
                async_public.set_sandbox_mode(True)
            
            tickers = await async_public.fetch_tickers(formatted_symbols)
            await async_public.close()

            for sym_raw, ccxt_sym in zip(symbols, formatted_symbols):
                ticker = tickers.get(ccxt_sym, {})
                last_price = safe_float(ticker.get('last') or ticker.get('close'))
                price_map[sym_raw] = last_price
            return price_map
        except Exception as e:
            logger.error(f"Batch async ticker fetch error: {e}")
            return {s: 0.0 for s in symbols}

    def fetch_ticker_data(self, symbol: str) -> Dict[str, Any]:
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        try:
            ticker = self.public_exchange.fetch_ticker(ccxt_symbol)
            bid = safe_float(ticker.get('bid'))
            ask = safe_float(ticker.get('ask'))
            last = safe_float(ticker.get('last'))
            info = ticker.get('info', {})
            
            mark_price = safe_float(ticker.get('markPrice') or info.get('markPrice') or info.get('indexPrice'))

            if last > 0:
                exec_price = last
            elif bid > 0 and ask > 0:
                exec_price = (bid + ask) / 2.0
            else:
                exec_price = mark_price

            spread_pct = (ask - bid) / bid if (bid > 0 and ask > 0) else 0.0

            return {
                "exec_price": exec_price,
                "bid": bid,
                "ask": ask,
                "last": last,
                "mark_price": mark_price,
                "spread_pct": spread_pct
            }
        except Exception as e:
            logger.error(f"[{ccxt_symbol}] Failed to fetch direct public ticker data: {e}")
            return {
                "exec_price": 0.0,
                "bid": 0.0,
                "ask": 0.0,
                "last": 0.0,
                "mark_price": 0.0,
                "spread_pct": 0.0
            }

    def fetch_ticker_price(self, symbol: str) -> float:
        data = self.fetch_ticker_data(symbol)
        return data["exec_price"]

    def validate_entry_price_deviation(self, symbol: str, strategy_entry_price: float, ticker_data: Dict[str, Any]) -> Tuple[bool, str]:
        if strategy_entry_price <= 0:
            return True, "No strategy entry price specified; skipping deviation check."

        exec_price = ticker_data.get("exec_price", 0.0)
        mark_price = ticker_data.get("mark_price", 0.0)

        if exec_price <= 0:
            return False, f"[{symbol}] Invalid live execution price: {exec_price}"

        tolerance_pct = ENTRY_TOLERANCE_PCT_TESTNET if BYBIT_TESTNET else ENTRY_TOLERANCE_PCT_LIVE
        price_dev = abs(exec_price - strategy_entry_price) / strategy_entry_price

        if price_dev > tolerance_pct:
            msg = (
                f"[{symbol}] Execution Rejected: Live market price (${exec_price:.5f}) "
                f"deviates from Strategy Entry (${strategy_entry_price:.5f}) by {price_dev * 100:.2f}% "
                f"(Max allowed: {tolerance_pct * 100:.2f}% | Testnet Mark Price: ${mark_price:.5f})."
            )
            logger.error(msg)
            return False, msg

        logger.info(
            f"[{symbol}] Price deviation check passed: {price_dev * 100:.2f}% deviation "
            f"(Live: ${exec_price:.5f} vs Strategy: ${strategy_entry_price:.5f})."
        )
        return True, "Price deviation within accepted tolerance."

    async def get_futures_position_async(self, symbol: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.get_futures_position, symbol)

    def get_futures_position(self, symbol: str) -> Dict[str, Any]:
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        clean_target = symbol.replace("/", "").replace(":", "").replace("_", "").replace("-", "").upper()
        try:
            positions = self.exchange.fetch_positions([ccxt_symbol])
            for pos in positions:
                pos_symbol = str(pos.get('symbol', '')).replace("/", "").replace(":", "").replace("_", "").replace("-", "").upper()
                contracts = safe_float(pos.get('contracts', 0.0))
                
                if clean_target in pos_symbol or pos_symbol in clean_target:
                    if contracts > 0:
                        return {
                            "symbol": symbol,
                            "side": str(pos.get('side', '')).upper(),
                            "contracts": contracts,
                            "entry_price": safe_float(pos.get('entryPrice', 0.0)),
                            "stop_loss": safe_float(pos.get('stopLoss', 0.0)),
                            "take_profit": safe_float(pos.get('takeProfit', 0.0)),
                            "leverage": safe_float(pos.get('leverage', 1.0)),
                            "unrealized_pnl": safe_float(pos.get('unrealizedPnl', 0.0)),
                            "error": False
                        }
            return {
                "symbol": symbol,
                "side": "NONE",
                "contracts": 0.0,
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "leverage": 1.0,
                "unrealized_pnl": 0.0,
                "error": False
            }
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            logger.error(f"Failed to fetch futures position for {ccxt_symbol}: {e}")
            return {
                "symbol": symbol,
                "side": "NONE",
                "contracts": 0.0,
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "leverage": 1.0,
                "unrealized_pnl": 0.0,
                "error": True
            }

    def fetch_available_usdt_balance(self) -> float:
        """Fetch wallet balance directly via proxied V5 account endpoint or CCXT balance query."""
        try:
            # First attempt: standard CCXT fetch_balance routed through Worker
            balance = self.exchange.fetch_balance({'type': 'linear'})
            usdt_free = safe_float(balance.get('USDT', {}).get('free', 0.0))
            if usdt_free == 0.0:
                usdt_free = safe_float(balance.get('free', {}).get('USDT', 0.0))
            if usdt_free > 0.0:
                return usdt_free
                
            # Fallback attempt: Query V5 UNIFIED account endpoint explicitly via private CCXT method
            res = self.exchange.privateGetV5AccountWalletBalance({'accountType': 'UNIFIED', 'coin': 'USDT'})
            list_data = res.get('result', {}).get('list', [])
            if list_data:
                coins = list_data[0].get('coin', [])
                for c in coins:
                    if c.get('coin') == 'USDT':
                        return safe_float(c.get('walletBalance') or c.get('equity') or 0.0)
            return 0.0
        except Exception as e:
            logger.error(f"Failed to fetch live USDT Futures balance: {e}")
            return 0.0

    def get_available_usdt_balance(self) -> float:
        return self.fetch_available_usdt_balance()

    async def poll_market_and_positions(self, symbols: list):
        """Asynchronous REST HTTP Polling loop routed through Worker."""
        logger.info(f"Starting REST HTTP Polling loop via Cloudflare Worker (Interval: {POLL_INTERVAL_SECONDS}s)...")
        while True:
            try:
                if not symbols:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                prices = await self.fetch_tickers_batch_async(symbols)
                
                for symbol in symbols:
                    current_price = prices.get(symbol, 0.0)
                    logger.debug(f"[{symbol}] Polled Market Price: ${current_price:.5f}")

                    position = await self.get_futures_position_async(symbol)
                    if position["contracts"] > MIN_DUST_THRESHOLD:
                        logger.info(f"[{symbol}] Polled Active Position: {position['contracts']} contracts {position['side']}")

            except Exception as e:
                logger.error(f"Error in REST market polling loop: {e}")

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def execute_live_order(
        self, 
        pair: str, 
        direction: str = "BUY", 
        entry_price: float = 0.0, 
        stop_loss: float = 0.0, 
        take_profit: float = 0.0,
        amount_usd: float = 25.0,
        leverage: int = 10,
        account_balance: float = 100.0,
        risk_pct: float = 1.0
    ) -> bool:
        clean_direction = direction.upper().strip()
        
        if clean_direction not in ["BUY", "LONG", "SELL", "SHORT"]:
            logger.warning(f"[{pair}] Invalid order direction: {clean_direction}")
            return False

        res = self.order_futures_bybit(
            symbol=pair, 
            direction=clean_direction, 
            amount_usd=amount_usd, 
            entry_price=entry_price,
            stop_loss=stop_loss, 
            take_profit=take_profit,
            leverage=leverage,
            account_balance=account_balance,
            risk_pct=risk_pct
        )
        if res.get("status") == "SUCCESS":
            return True
        else:
            logger.error(f"[{pair}] Live Bybit futures order execution failed: {res.get('error') or res.get('reason')}")
            return False

    def order_futures_bybit(
        self, 
        symbol: str, 
        direction: str, 
        amount_usd: float, 
        entry_price: float = 0.0,
        stop_loss: float = 0.0, 
        take_profit: float = 0.0,
        leverage: float = 5.0,
        account_balance: float = 100.0,
        risk_pct: float = 1.0
    ) -> Dict[str, Any]:
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        dir_clean = direction.upper().strip()
        is_long = dir_clean in ["BUY", "LONG"]
        side = 'buy' if is_long else 'sell'

        pos_info = self.get_futures_position(ccxt_symbol)
        if pos_info.get("error"):
            logger.warning(f"[{symbol}] Guard Triggered: Could not verify existing position due to API error. Aborting order.")
            return {"status": "FAILED", "error": "API error checking position before entry"}

        if pos_info["contracts"] > MIN_DUST_THRESHOLD:
            logger.warning(f"[{symbol}] Guard Triggered: Active futures position exists ({pos_info['contracts']} contracts {pos_info['side']}). Blocking execution.")
            return {"status": "SKIPPED", "reason": "Active futures position detected on exchange"}

        available_usdt = self.fetch_available_usdt_balance()
        if available_usdt <= 1.0:
            return {"status": "FAILED", "error": f"Insufficient live USDT balance: ${available_usdt:.2f}"}

        ticker_data = self.fetch_ticker_data(ccxt_symbol)
        ref_price = ticker_data["ask"] if (is_long and ticker_data["ask"] > 0) else ticker_data["bid"] if (not is_long and ticker_data["bid"] > 0) else ticker_data["exec_price"]
        spread_pct = ticker_data["spread_pct"]

        if ref_price <= 0:
            return {"status": "FAILED", "error": "Invalid reference ticker price prior to execution"}

        is_valid_entry, dev_reason = self.validate_entry_price_deviation(symbol, entry_price, ticker_data)
        if not is_valid_entry:
            return {"status": "FAILED", "error": dev_reason}

        if spread_pct > MAX_ALLOWED_SPREAD_PCT and not BYBIT_TESTNET:
            logger.warning(f"[{symbol}] Order Rejected: High Spread detected ({spread_pct * 100:.3f}% > Max allowed {MAX_ALLOWED_SPREAD_PCT * 100:.2f}%).")
            return {"status": "FAILED", "error": f"Spread too high ({spread_pct * 100:.3f}%)"}

        try:
            self.exchange.set_leverage(int(leverage), ccxt_symbol)
        except Exception as lev_err:
            logger.debug(f"[{symbol}] Leverage setting notice: {lev_err}")

        if stop_loss > 0 and abs(ref_price - stop_loss) > 0:
            risk_amount_usd = account_balance * (risk_pct / 100.0)
            sl_distance = abs(ref_price - stop_loss)
            target_contracts = risk_amount_usd / sl_distance
            
            required_margin = (target_contracts * ref_price) / leverage
            if required_margin > amount_usd:
                logger.info(f"[{symbol}] Fixed-Risk margin (${required_margin:.2f}) capped by Equal-Weight cap (${amount_usd:.2f}).")
                required_margin = amount_usd
                target_contracts = (required_margin * leverage) / ref_price
            
            raw_qty = target_contracts
            trade_amount_usd = required_margin
        else:
            trade_amount_usd = min(amount_usd, available_usdt * 0.98)
            notional_value = trade_amount_usd * leverage
            raw_qty = notional_value / ref_price

        limit_price = ref_price * (1.0 + MAX_SLIPPAGE_PCT) if is_long else ref_price * (1.0 - MAX_SLIPPAGE_PCT)

        try:
            formatted_qty = safe_float(self.exchange.amount_to_precision(ccxt_symbol, raw_qty))
            formatted_limit_price = safe_float(self.exchange.price_to_precision(ccxt_symbol, limit_price))
        except Exception:
            formatted_qty = round(raw_qty, 4)
            formatted_limit_price = round(limit_price, 6)

        params = {'timeInForce': 'IOC'}

        if stop_loss > 0:
            if is_long and stop_loss >= ref_price:
                logger.error(f"[{symbol}] Invalid Long Stop Loss (${stop_loss:.5f}) >= Reference Price (${ref_price:.5f}). Stripping SL parameter.")
                stop_loss = 0.0
            elif not is_long and stop_loss <= ref_price:
                logger.error(f"[{symbol}] Invalid Short Stop Loss (${stop_loss:.5f}) <= Reference Price (${ref_price:.5f}). Stripping SL parameter.")
                stop_loss = 0.0
            else:
                params['stopLoss'] = safe_float(self.exchange.price_to_precision(ccxt_symbol, stop_loss))

        if take_profit > 0:
            params['takeProfit'] = safe_float(self.exchange.price_to_precision(ccxt_symbol, take_profit))

        logger.info(
            f"[{symbol}] Initiating Futures Limit {side.upper()} (IOC): Margin=${trade_amount_usd:.2f} @ {leverage}x "
            f"({formatted_qty} contracts @ Limit: ${formatted_limit_price:.6f})"
        )

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
                    if not pos_check.get("error") and pos_check["contracts"] > MIN_DUST_THRESHOLD:
                        executed_qty = pos_check["contracts"]
                        fill_price = pos_check["entry_price"] if pos_check["entry_price"] > 0 else ref_price

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

    def close_live_position_bybit(self, symbol: str, position_size: float, current_price: float, outcome: str = "CLOSE") -> Dict[str, Any]:
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        pos_info = self.get_futures_position(ccxt_symbol)
        
        if pos_info.get("error"):
            logger.error(f"[{symbol}] Cannot attempt position close due to API/network error during verification.")
            return {"status": "FAILED", "error": "Network/API glitch preventing close verification"}

        contracts = pos_info["contracts"]
        current_side = pos_info["side"]

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
                "order_id": "GHOST_POSITION_DB_CLOSED",
                "exit_price": current_price,
                "executed_qty": position_size
            }

        close_side = 'sell' if current_side == 'BUY' else 'buy'
        try:
            order = self.exchange.create_order(
                symbol=ccxt_symbol,
                type='market',
                side=close_side,
                amount=contracts,
                params={'reduceOnly': True}
            )

            fill_price = safe_float(order.get("average") or order.get("price"))
            if fill_price <= 0:
                fill_price = current_price

            set_asset_cooldown(symbol, hours=2)
            event_bus.disarm_local_sl_guard(symbol)

            return {
                "status": "SUCCESS",
                "order_id": order.get("id"),
                "exit_price": fill_price,
                "executed_qty": contracts
            }
        except Exception as close_err:
            logger.error(f"[{symbol}] Error executing market close on Bybit: {close_err}")
            return {"status": "FAILED", "error": str(close_err)}

LiveExecutionEngine = BybitFuturesLiveExecutor