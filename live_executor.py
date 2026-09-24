import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple, List
import ccxt.async_support as ccxt_async
import ccxt
import pandas as pd
import requests

from config import BYBIT_API_KEY, BYBIT_SECRET_KEY, BYBIT_TESTNET
from common import (
    get_db_connection,
    release_db_connection,
    send_telegram_notification,
    set_asset_cooldown,
    finalize_trade_in_db,
    calculate_pnl,
    logger
)
from event_bus import event_bus

BYBIT_WS_URL = (
    "wss://stream-testnet.bybit.com/v5/public/linear"
    if BYBIT_TESTNET
    else "wss://stream.bybit.com/v5/public/linear"
)

POLL_INTERVAL_SECONDS = 10
MIN_DUST_THRESHOLD = 0.001
MAX_SLIPPAGE_PCT = 0.002
MAX_ALLOWED_SPREAD_PCT = 0.003

ENTRY_TOLERANCE_PCT_LIVE = 0.005
ENTRY_TOLERANCE_PCT_TESTNET = 0.05

# FIX #8: Toggle maker orders to reduce fee drag
USE_MAKER_ORDERS = os.getenv("USE_MAKER_ORDERS", "true").lower() == "true"


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def format_ccxt_futures_symbol(symbol: str) -> str:
    """Consistently converts raw/dirty symbols to CCXT Bybit Linear Futures format."""
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


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 200) -> pd.DataFrame:
    """Primary kline fetcher prioritizing Bybit directly to eliminate basis risk."""
    ccxt_symbol = format_ccxt_futures_symbol(symbol)
    
    # 1. Tier 1 (Primary): Bybit Linear Futures REST
    try:
        public_ex = ccxt.bybit({'options': {'defaultType': 'linear'}})
        if BYBIT_TESTNET:
            public_ex.set_sandbox_mode(True)
        ohlcv = public_ex.fetch_ohlcv(ccxt_symbol, timeframe=interval, limit=limit)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
    except Exception as e:
        logger.warning(f"[{symbol}] Primary Bybit kline fetch failed ({e}). Falling back to Binance...")

    # 2. Tier 2 Fallback: Binance Futures
    try:
        binance_ex = ccxt.binanceusdm()
        ohlcv = binance_ex.fetch_ohlcv(symbol.split(':')[0], timeframe=interval, limit=limit)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
    except Exception as e:
        logger.warning(f"[{symbol}] Secondary Binance kline fetch failed ({e}). Falling back to CryptoCompare...")

    # 3. Tier 3 Fallback: CryptoCompare
    raw_candles = fetch_cryptocompare_fallback_kline(symbol, limit=limit)
    if raw_candles:
        df = pd.DataFrame(raw_candles)
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    return pd.DataFrame()


def fetch_cryptocompare_fallback_kline(symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        raw = symbol.split(":")[0]
        if "/" in raw:
            fsym, tsym = raw.split("/")
        else:
            fsym = raw.replace("USDT", "").replace("USD", "")
            tsym = "USDT"

        url = "https://min-api.cryptocompare.com/data/v2/histominute"
        params = {"fsym": fsym.upper(), "tsym": tsym.upper(), "limit": limit, "e": "CCCAGG"}
        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        if data.get("Response") == "Success":
            raw_candles = data.get("Data", {}).get("Data", [])
            return [
                {
                    "timestamp": c.get("time") * 1000,
                    "open": float(c.get("open", 0.0)),
                    "high": float(c.get("high", 0.0)),
                    "low": float(c.get("low", 0.0)),
                    "close": float(c.get("close", 0.0)),
                    "volume": float(c.get("volumeto", 0.0))
                }
                for c in raw_candles
            ]
        else:
            logger.error(f"[{symbol}] CryptoCompare fallback error: {data.get('Message')}")
            return []
    except Exception as e:
        logger.error(f"[{symbol}] Exception in CryptoCompare fallback: {e}")
        return []


class BybitFuturesLiveExecutor:
    def __init__(self):
        self.exchange = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
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

        if BYBIT_TESTNET:
            self.exchange.set_sandbox_mode(True)
            self.public_exchange.set_sandbox_mode(True)

        try:
            self.public_exchange.load_markets()
        except Exception as e:
            logger.warning(f"Could not load public market structures: {e}")

    def format_ccxt_futures_symbol(self, raw_symbol: str) -> str:
        return format_ccxt_futures_symbol(raw_symbol)

    def set_position_trading_stop(self, symbol: str, stop_loss: float, position_idx: int = 0) -> bool:
        """
        FIX #2: Auto-discovers the correct CCXT method name and verifies the SL is actually
        visible on the exchange after the update.
        """
        try:
            ccxt_symbol = self.format_ccxt_futures_symbol(symbol)
            formatted_sl = float(self.exchange.price_to_precision(ccxt_symbol, stop_loss))
            formatted_symbol = symbol.replace("/", "").replace(":USDT", "").replace("-", "").replace("_", "").upper()

            params = {
                "category": "linear",
                "symbol": formatted_symbol,
                "stopLoss": str(formatted_sl),
                "slTriggerBy": "LastPrice",
                "positionIdx": position_idx,
            }

            candidates = [
                "private_post_v5_position_trading_stop",
                "private_linear_post_v5_position_trading_stop",
                "privatePostV5PositionTradingStop",
            ]
            method = None
            for name in candidates:
                if hasattr(self.exchange, name):
                    method = getattr(self.exchange, name)
                    logger.info(f"[{symbol}] Using CCXT method '{name}' for trading_stop.")
                    break

            if method is None:
                logger.critical(
                    f"[{symbol}] FATAL: No CCXT trading_stop method found. "
                    f"Available: {[m for m in dir(self.exchange) if 'trading_stop' in m.lower()]}"
                )
                return False

            response = method(params)

            if not isinstance(response, dict) or response.get("retCode") != 0:
                logger.error(
                    f"[SL SET ERROR] Symbol: {formatted_symbol} | "
                    f"retCode: {response.get('retCode')} | retMsg: {response.get('retMsg')}"
                )
                return False

            time.sleep(0.3)
            try:
                positions = self.exchange.fetch_positions([ccxt_symbol])
                for pos in positions:
                    if safe_float(pos.get("contracts", 0)) > MIN_DUST_THRESHOLD:
                        live_sl = safe_float(pos.get("stopLoss", 0))
                        if live_sl > 0 and abs(live_sl - formatted_sl) / formatted_sl < 0.005:
                            logger.info(f"[SL VERIFIED] {formatted_symbol} SL=${live_sl} confirmed on exchange.")
                            return True
                        else:
                            logger.error(
                                f"[SL VERIFY FAILED] {formatted_symbol} expected ${formatted_sl}, "
                                f"exchange reports ${live_sl}"
                            )
                            return False
            except Exception as verify_err:
                logger.warning(f"[{symbol}] Post-SL verification fetch failed (treating as success): {verify_err}")

            logger.info(f"[SL SET SUCCESS] Symbol: {formatted_symbol} | SL Price: {formatted_sl}")
            return True
        except Exception as e:
            logger.exception(f"[SL SET EXCEPTION] Failed to set Stop Loss for {symbol}: {e}")
            return False

    async def fetch_ticker_direct_async(self, symbol: str) -> float:
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        try:
            async_config = {'enableRateLimit': True, 'options': {'defaultType': 'linear'}}
            async_public = ccxt_async.bybit(async_config)

            if BYBIT_TESTNET:
                async_public.set_sandbox_mode(True)

            ticker = await async_public.fetch_ticker(ccxt_symbol)
            await async_public.close()
            return safe_float(ticker.get('last') or ticker.get('close'))
        except Exception as e:
            logger.error(f"[{ccxt_symbol}] Direct async ticker fetch error: {e}")
            return 0.0

    async def fetch_tickers_batch_async(self, symbols: List[str]) -> Dict[str, float]:
        formatted_symbols = [format_ccxt_futures_symbol(s) for s in symbols]
        price_map = {}
        try:
            async_config = {'enableRateLimit': True, 'options': {'defaultType': 'linear'}}
            async_public = ccxt_async.bybit(async_config)

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
                "exec_price": 0.0, "bid": 0.0, "ask": 0.0,
                "last": 0.0, "mark_price": 0.0, "spread_pct": 0.0
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
                "symbol": symbol, "side": "NONE", "contracts": 0.0,
                "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
                "leverage": 1.0, "unrealized_pnl": 0.0, "error": False
            }
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            logger.error(f"Failed to fetch futures position for {ccxt_symbol}: {e}")
            return {
                "symbol": symbol, "side": "NONE", "contracts": 0.0,
                "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
                "leverage": 1.0, "unrealized_pnl": 0.0, "error": True
            }

    def fetch_available_usdt_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance({'type': 'linear'})
            usdt_free = safe_float(balance.get('USDT', {}).get('free', 0.0))
            if usdt_free == 0.0:
                usdt_free = safe_float(balance.get('free', {}).get('USDT', 0.0))
            if usdt_free > 0.0:
                return usdt_free

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
        logger.info(f"Starting REST HTTP Polling loop (Interval: {POLL_INTERVAL_SECONDS}s)...")
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

    def execute_live_order(self, pair: str, direction: str = "BUY", entry_price: float = 0.0,
                           stop_loss: float = 0.0, take_profit: float = 0.0, amount_usd: float = 25.0,
                           leverage: int = 5.0, account_balance: float = 100.0, risk_pct: float = 1.0) -> bool:
        clean_direction = direction.upper().strip()

        if clean_direction not in ["BUY", "LONG", "SELL", "SHORT"]:
            logger.warning(f"[{pair}] Invalid order direction: {clean_direction}")
            return False

        res = self.order_futures_bybit(
            symbol=pair, direction=clean_direction, amount_usd=amount_usd,
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            leverage=leverage, account_balance=account_balance, risk_pct=risk_pct
        )
        if res.get("status") == "SUCCESS":
            return True
        else:
            logger.error(f"[{pair}] Live Bybit futures order execution failed: {res.get('error') or res.get('reason')}")
            return False

    def execute_order_and_attach_sl(self, symbol: str, side: str, amount: float, target_sl: float) -> Dict[str, Any]:
        order = self.order_futures_bybit(
            symbol=symbol, direction=side, amount_usd=amount, stop_loss=target_sl
        )

        if order and order.get("status") == "SUCCESS":
            logger.info(f"Order filled for {symbol}. Verifying explicit post-execution Stop Loss at {target_sl}")

            pos_idx = 0

            if target_sl > 0:
                sl_success = self.set_position_trading_stop(
                    symbol=symbol, stop_loss=target_sl, position_idx=pos_idx
                )
                order["stop_loss_attached"] = sl_success

                if not sl_success:
                    logger.critical(
                        f"🚨 [EMERGENCY GUARD] Failed to set exchange Stop Loss for {symbol} at ${target_sl}. "
                        f"Closing position immediately to prevent naked risk exposure."
                    )
                    exec_qty = order.get("executed_qty", 0.0)
                    fill_price = order.get("fill_price", 0.0)
                    close_res = self.close_live_position_bybit(
                        symbol=symbol, position_size=exec_qty, current_price=fill_price,
                        outcome="EMERGENCY_SL_ATTACH_FAILURE"
                    )
                    send_telegram_notification(
                        f"🚨 <b>EMERGENCY POSITION CLOSE</b>\n\n"
                        f"<b>Symbol:</b> <code>{symbol}</code>\n"
                        f"<b>Reason:</b> Failed to attach Exchange Stop Loss ({target_sl})\n"
                        f"<b>Close Status:</b> {close_res.get('status')}"
                    )
                    return {
                        "status": "FAILED",
                        "error": "Emergency close triggered: exchange stop loss failed to attach."
                    }

        return order

    def order_futures_bybit(self, symbol: str, direction: str, amount_usd: float,
                            entry_price: float = 0.0, stop_loss: float = 0.0, take_profit: float = 0.0,
                            leverage: float = 5.0, account_balance: float = 100.0, risk_pct: float = 1.0) -> Dict[str, Any]:
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
            logger.warning(f"[{symbol}] Order Rejected: High Spread detected ({spread_pct * 100:.3f}%).")
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

        # FIX #8: Maker (PostOnly) preferred; fallback to IOC if disabled
        if USE_MAKER_ORDERS:
            book = self.fetch_ticker_data(ccxt_symbol)
            if is_long and book["bid"] > 0:
                limit_price = book["bid"]
            elif not is_long and book["ask"] > 0:
                limit_price = book["ask"]
            else:
                limit_price = ref_price
            params = {'timeInForce': 'PostOnly', 'reduceOnly': False}
        else:
            limit_price = ref_price * (1.0 + MAX_SLIPPAGE_PCT) if is_long else ref_price * (1.0 - MAX_SLIPPAGE_PCT)
            params = {'timeInForce': 'IOC'}

        try:
            formatted_qty = safe_float(self.exchange.amount_to_precision(ccxt_symbol, raw_qty))
            formatted_limit_price = safe_float(self.exchange.price_to_precision(ccxt_symbol, limit_price))
        except Exception:
            formatted_qty = round(raw_qty, 4)
            formatted_limit_price = round(limit_price, 6)

        if stop_loss > 0:
            if is_long and stop_loss >= ref_price:
                logger.error(f"[{symbol}] Invalid Long Stop Loss (${stop_loss:.5f}) >= Reference Price (${ref_price:.5f}). Stripping SL parameter.")
                target_sl = 0.0
            elif not is_long and stop_loss <= ref_price:
                logger.error(f"[{symbol}] Invalid Short Stop Loss (${stop_loss:.5f}) <= Reference Price (${ref_price:.5f}). Stripping SL parameter.")
                target_sl = 0.0
            else:
                target_sl = stop_loss
                params['stopLoss'] = safe_float(self.exchange.price_to_precision(ccxt_symbol, target_sl))
        else:
            target_sl = 0.0

        if take_profit > 0:
            params['takeProfit'] = safe_float(self.exchange.price_to_precision(ccxt_symbol, take_profit))

        logger.info(
            f"[{symbol}] Initiating Futures {side.upper()} (TIF={params.get('timeInForce', 'IOC')}): Margin=${trade_amount_usd:.2f} @ {leverage}x "
            f"({formatted_qty} contracts @ Limit: ${formatted_limit_price:.6f})"
        )

        try:
            try:
                order = self.exchange.create_order(
                    symbol=ccxt_symbol, type='limit', side=side,
                    amount=formatted_qty, price=formatted_limit_price, params=params
                )
            except ccxt.OrderImmediatelyFillable:
                logger.warning(f"[{symbol}] PostOnly crossed spread. Retrying as GTC limit.")
                params['timeInForce'] = 'GTC'
                order = self.exchange.create_order(
                    symbol=ccxt_symbol, type='limit', side=side,
                    amount=formatted_qty, price=formatted_limit_price, params=params
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
                logger.warning(f"[{symbol}] Order unfilled/cancelled due to slippage ceiling.")
                return {"status": "FAILED", "error": "Limit order unfilled due to slippage guard."}

            if fill_price <= 0:
                fill_price = ref_price

            logger.info(f"[{symbol}] Futures Order Executed: Side {side.upper()}, Price ${fill_price:.6f}, Qty {executed_qty}")

            # FIX #2: Emergency close on SL attach failure
            sl_attached = False
            if target_sl > 0:
                sl_attached = self.set_position_trading_stop(
                    symbol=symbol, stop_loss=target_sl, position_idx=0
                )
                if not sl_attached:
                    logger.critical(
                        f"🚨 [EMERGENCY GUARD] Exchange SL attachment FAILED for {symbol} @ ${target_sl}. "
                        f"Closing position immediately to prevent naked risk."
                    )
                    close_res = self.close_live_position_bybit(
                        symbol=symbol, position_size=executed_qty, current_price=fill_price,
                        outcome="EMERGENCY_SL_ATTACH_FAILURE"
                    )
                    send_telegram_notification(
                        f"🚨 <b>EMERGENCY POSITION CLOSE</b>\n\n"
                        f"<b>Symbol:</b> <code>{symbol}</code>\n"
                        f"<b>Reason:</b> Exchange SL failed to attach\n"
                        f"<b>Close Status:</b> {close_res.get('status')}"
                    )
                    return {
                        "status": "FAILED",
                        "error": "Emergency close triggered: exchange SL failed to attach."
                    }

            return {
                "status": "SUCCESS",
                "order_id": order_id,
                "fill_price": fill_price,
                "executed_qty": executed_qty,
                "cost_usd": trade_amount_usd,
                "stop_loss_attached": sl_attached,
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
        current_side = str(pos_info["side"]).upper()

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

        # FIX #1: Ghost-close branch now pulls real PnL from Bybit
        if contracts < MIN_DUST_THRESHOLD:
            set_asset_cooldown(symbol, hours=4)
            event_bus.disarm_local_sl_guard(symbol)

            if trade_id:
                real_closed_pnl = None
                real_fee = 0.0
                verified_exit = current_price
                try:
                    formatted_symbol = symbol.replace("/", "").replace(":USDT", "").replace("_", "").upper()
                    cpnl_resp = self.exchange.private_get_v5_position_closed_pnl({
                        "category": "linear",
                        "symbol": formatted_symbol,
                        "limit": 1
                    })
                    records = cpnl_resp.get("result", {}).get("list", [])
                    if records:
                        latest = records[0]
                        real_closed_pnl = safe_float(latest.get("closedPnl"), 0.0)
                        real_fee = abs(safe_float(latest.get("openFee"), 0.0)) + abs(safe_float(latest.get("closeFee"), 0.0))
                        verified_exit = safe_float(latest.get("avgExitPrice"), current_price)
                except Exception as pnl_err:
                    logger.warning(f"[{symbol}] Ghost-close PnL lookup failed: {pnl_err}")

                conn = get_db_connection()
                if conn:
                    try:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT direction, entry_price, position_size, account_balance FROM trade_setups WHERE id = %s;",
                                (trade_id,)
                            )
                            row = cur.fetchone()
                            if row:
                                direction, entry_p, pos_qty, acc_bal = row
                                pnl_usd, pnl_pct, outcome_verified = calculate_pnl(
                                    direction=direction,
                                    entry_price=safe_float(entry_p),
                                    current_price=verified_exit,
                                    quantity=safe_float(pos_qty),
                                    account_balance=safe_float(acc_bal, 100.0),
                                    total_fees=real_fee,
                                    exchange_closed_pnl=real_closed_pnl
                                )
                                finalize_trade_in_db(trade_id, verified_exit, pnl_usd, pnl_pct, outcome_verified, fee_usd=real_fee)
                    finally:
                        release_db_connection(conn)

            return {
                "status": "FORCE_CLOSED_DB_ONLY",
                "order_id": "GHOST_POSITION_DB_CLOSED",
                "exit_price": current_price,
                "executed_qty": position_size
            }

        close_side = 'sell' if current_side in ['BUY', 'LONG'] else 'buy'
        try:
            order = self.exchange.create_order(
                symbol=ccxt_symbol, type='market', side=close_side,
                amount=contracts, params={'reduceOnly': True}
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