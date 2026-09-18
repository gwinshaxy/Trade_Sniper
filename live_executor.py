import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple
import ccxt.async_support as ccxt_async
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

# =====================================================================
# CLOUDFLARE WORKER REVERSE PROXY CONFIGURATION
# Bypasses local IP restrictions via Cloudflare Worker edge nodes.
# Clean environment of any legacy HTTP proxy variables.
# =====================================================================
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# Set your Cloudflare Worker URL here or pull from environment variables
CLOUDFLARE_WORKER_URL = (
    os.getenv("CLOUDFLARE_WORKER_URL")
    or os.getenv("WORKER_URL")
    or "https://bybit-proxy.gspark4u.workers.dev"  # Default worker domain
)

if CLOUDFLARE_WORKER_URL:
    CLOUDFLARE_WORKER_URL = CLOUDFLARE_WORKER_URL.strip('"').strip("'").rstrip('/')

# Throttling interval for private execution polling loop
POLL_INTERVAL_SECONDS = 1080  # 18 minutes

MIN_DUST_THRESHOLD = 0.001
MAX_SLIPPAGE_PCT = 0.002       # 0.2% max execution limit offset
MAX_ALLOWED_SPREAD_PCT = 0.003 # 0.3% max spread threshold

# Environment-aware entry deviation tolerances
ENTRY_TOLERANCE_PCT_LIVE = 0.005    # 0.5% tolerance for live production trading
ENTRY_TOLERANCE_PCT_TESTNET = 0.05  # 5.0% extended tolerance to absorb testnet drift


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


class BybitFuturesLiveExecutor:
    def __init__(self):
        # 1. Private Exchange Config (Routed via Cloudflare Worker Reverse Proxy)
        private_exchange_config = {
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'recvWindow': 20000,
                'adjustForTimeDifference': True
            }
        }

        if CLOUDFLARE_WORKER_URL:
            logger.info(f"Routing CCXT Private Execution through Cloudflare Worker: {CLOUDFLARE_WORKER_URL}")
            private_exchange_config['urls'] = {
                'api': {
                    'public': CLOUDFLARE_WORKER_URL,
                    'private': CLOUDFLARE_WORKER_URL,
                }
            }
        else:
            logger.warning("No CLOUDFLARE_WORKER_URL specified. Operating on direct connection.")

        self.exchange = ccxt.bybit(private_exchange_config)

        # 2. Public Exchange Config (Also routed through Worker to avoid regional blocks)
        public_exchange_config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear'
            }
        }

        if CLOUDFLARE_WORKER_URL:
            public_exchange_config['urls'] = {
                'api': {
                    'public': CLOUDFLARE_WORKER_URL,
                    'private': CLOUDFLARE_WORKER_URL,
                }
            }

        self.public_exchange = ccxt.bybit(public_exchange_config)

        if BYBIT_TESTNET:
            self.exchange.set_sandbox_mode(True)
            self.public_exchange.set_sandbox_mode(True)

        try:
            self.public_exchange.load_markets()
        except Exception as e:
            logger.warning(f"Could not refresh public market structures on init: {e}")

    def format_ccxt_futures_symbol(self, raw_symbol: str) -> str:
        return format_ccxt_futures_symbol(raw_symbol)

    async def fetch_ticker_direct_async(self, symbol: str) -> float:
        """Asynchronously fetches current ticker price directly via Cloudflare Worker."""
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        try:
            async_config = {'enableRateLimit': True, 'options': {'defaultType': 'linear'}}
            if CLOUDFLARE_WORKER_URL:
                async_config['urls'] = {
                    'api': {
                        'public': CLOUDFLARE_WORKER_URL,
                        'private': CLOUDFLARE_WORKER_URL,
                    }
                }
            async_public = ccxt_async.bybit(async_config)
            if BYBIT_TESTNET:
                async_public.set_sandbox_mode(True)
            ticker = await async_public.fetch_ticker(ccxt_symbol)
            await async_public.close()
            return safe_float(ticker.get('last') or ticker.get('close'))
        except Exception as e:
            logger.error(f"[{ccxt_symbol}] Direct async ticker fetch error: {e}")
            return 0.0

    def fetch_ticker_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches ticker using public exchange instance routed through Cloudflare Worker.
        Prioritizes actual market execution prices (last traded price and orderbook prices)
        over testnet mark price drift.
        """
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        try:
            ticker = self.public_exchange.fetch_ticker(ccxt_symbol)
            bid = safe_float(ticker.get('bid'))
            ask = safe_float(ticker.get('ask'))
            last = safe_float(ticker.get('last'))
            info = ticker.get('info', {})
            
            mark_price = safe_float(ticker.get('markPrice') or info.get('markPrice') or info.get('indexPrice'))

            # Prioritize last traded price, then orderbook midpoint, using mark_price purely as fallback
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
        """Fetches the primary market execution reference price for a given symbol."""
        data = self.fetch_ticker_data(symbol)
        return data["exec_price"]

    def validate_entry_price_deviation(self, symbol: str, strategy_entry_price: float, ticker_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates strategy entry price against current ticker prices using environment-specific tolerance.
        """
        if strategy_entry_price <= 0:
            return True, "No strategy entry price specified; skipping deviation check."

        exec_price = ticker_data.get("exec_price", 0.0)
        mark_price = ticker_data.get("mark_price", 0.0)

        if exec_price <= 0:
            return False, f"[{symbol}] Invalid live execution price: {exec_price}"

        # Set dynamic tolerance depending on testnet vs live environment
        tolerance_pct = ENTRY_TOLERANCE_PCT_TESTNET if BYBIT_TESTNET else ENTRY_TOLERANCE_PCT_LIVE

        # Calculate deviation relative to strategy entry price
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
        """Position lookup via Cloudflare Worker proxy."""
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

    async def throttled_execution_loop(self, symbols: list):
        """
        Execution loop monitoring active positions and prices via Cloudflare Worker proxy.
        """
        logger.info(f"Starting execution loop via Cloudflare Worker (Poll interval: {POLL_INTERVAL_SECONDS}s)...")
        while True:
            try:
                for symbol in symbols:
                    # 1. Market Price Fetch
                    current_price = self.fetch_ticker_price(symbol)
                    logger.debug(f"[{symbol}] Direct Market Price: ${current_price:.5f}")

                    # 2. Private Position Verification
                    position = await self.get_futures_position_async(symbol)
                    if position["contracts"] > MIN_DUST_THRESHOLD:
                        logger.info(f"[{symbol}] Active position detected: {position['contracts']} contracts {position['side']}")

            except Exception as loop_err:
                logger.error(f"Error in execution loop iteration: {loop_err}")

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

        # Validate entry price deviation using updated environment-aware method
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