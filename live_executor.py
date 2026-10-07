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

ENTRY_TOLERANCE_PCT_LIVE = 0.003
ENTRY_TOLERANCE_PCT_TESTNET = 0.03

USE_MAKER_ORDERS = os.getenv("USE_MAKER_ORDERS", "true").lower() == "true"

SL_VERIFY_MAX_ATTEMPTS_LIVE = 8
SL_VERIFY_MAX_ATTEMPTS_TESTNET = 20
SL_API_MAX_RETRIES = 3


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def resolve_and_verify_symbol(exchange: ccxt.Exchange, base_symbol: str, quote_symbol: str = "USDT") -> str:
    """
    Dynamically resolves CCXT unified symbol notation across Mainnet and Testnet environments,
    verifying active trading status before returning the symbol key.
    """
    if not exchange.markets:
        exchange.load_markets()
        
    candidates = [
        f"{base_symbol}/{quote_symbol}:{quote_symbol}",  # Mainnet UTA format: 'ARB/USDT:USDT'
        f"{base_symbol}/{quote_symbol}",                # Testnet format: 'ARB/USDT'
    ]
    
    for candidate in candidates:
        if candidate in exchange.markets:
            if exchange.markets[candidate].get('active', True):
                return candidate
                
    # Fallback lookup across all markets
    for key, market in exchange.markets.items():
        if key.startswith(f"{base_symbol}/{quote_symbol}") and market.get('active', True):
            return key
            
    raise ValueError(f"No active market found on Bybit for {base_symbol}/{quote_symbol}")


def format_ccxt_futures_symbol(symbol: str, exchange: Optional[ccxt.Exchange] = None) -> str:
    """
    Dynamically resolves raw/dirty symbols using CCXT market verification if exchange instance is provided,
    otherwise falls back to standard CCXT string normalization.
    """
    if not symbol:
        return ""

    # Clean raw string input
    raw = symbol.split(":")[0].replace("/", "").replace("_", "").replace("-", "").upper()
    if raw.endswith("USDTUSDT"):
        raw = raw[:-4]
    
    if raw.endswith("USDT"):
        base_symbol = raw[:-4]
        quote_symbol = "USDT"
    elif raw.endswith("USDC"):
        base_symbol = raw[:-4]
        quote_symbol = "USDC"
    else:
        base_symbol = raw
        quote_symbol = "USDT"

    if exchange is not None:
        try:
            return resolve_and_verify_symbol(exchange, base_symbol, quote_symbol)
        except Exception as e:
            logger.warning(f"Failed to resolve market via exchange: {e}. Falling back to default format.")

    if ":" in symbol:
        return symbol

    return f"{base_symbol}/{quote_symbol}:{quote_symbol}"


class BybitFuturesLiveExecutor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
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
            self.exchange.load_markets()
            self.public_exchange.load_markets()
        except Exception as e:
            logger.warning(f"Could not load market structures: {e}")

        self.position_mode: Optional[str] = None
        try:
            acct_info = self.exchange.private_get_v5_account_info()
            self.position_mode = (
                acct_info.get("result", {}).get("positionMode")
                if isinstance(acct_info, dict) else None
            )
            logger.info(f"Bybit account position mode detected: {self.position_mode}")
        except Exception as e:
            logger.warning(f"Could not determine account position mode at startup: {e}")

    def resolve_symbol(self, symbol: str) -> str:
        """Helper method using self.exchange to resolve symbol dynamically."""
        if not symbol:
            return ""
        
        # Check config symbols map first
        if hasattr(self, 'config') and isinstance(self.config, dict):
            symbols_map = self.config.get("symbols", {})
            if symbol in symbols_map:
                mapped = symbols_map[symbol]
                if isinstance(mapped, dict):
                    mapped = mapped.get("binance", mapped.get("bybit", symbol))
                if ":" in str(mapped):
                    return str(mapped)
                symbol = str(mapped)

        raw = symbol.split(":")[0].replace("/", "").replace("_", "").replace("-", "").upper()
        if raw.endswith("USDTUSDT"):
            raw = raw[:-4]
        
        base_symbol = raw[:-4] if raw.endswith("USDT") else raw
        quote_symbol = "USDT"
        
        try:
            return resolve_and_verify_symbol(self.exchange, base_symbol, quote_symbol)
        except Exception as e:
            logger.warning(f"[{symbol}] Symbol dynamic resolution fallback to format_ccxt_futures_symbol: {e}")
            return format_ccxt_futures_symbol(symbol)

    def format_ccxt_futures_symbol(self, symbol: str) -> str:
        """Instance wrapper directing to resolve_symbol for unified market verification."""
        return self.resolve_symbol(symbol)

    def _get_position_idx(self, ccxt_symbol: str, direction: str) -> int:
        """Returns 0 for one-way mode, 1 for hedge-buy, 2 for hedge-sell."""
        clean_dir = str(direction or "").upper()
        is_buy = clean_dir in ("BUY", "LONG")

        if self.position_mode == "MergedSingle":
            return 0
        if self.position_mode == "BothSide":
            return 1 if is_buy else 2

        try:
            exchange_symbol = ccxt_symbol
            if hasattr(self, 'config') and isinstance(self.config, dict):
                symbols_map = self.config.get("symbols", {})
                exchange_symbol = symbols_map.get(ccxt_symbol, symbols_map.get(ccxt_symbol.split('/')[0], ccxt_symbol))

            formatted = (
                exchange_symbol.replace("/", "")
                .replace(":USDT", "")
                .replace("-", "")
                .replace("_", "")
                .upper()
            )
            info = self.exchange.private_get_v5_position_list({
                "category": "linear", "symbol": formatted
            })
            positions = info.get("result", {}).get("list", [])
            position_idxs = {int(p.get("positionIdx", 0) or 0) for p in positions}
            if 1 in position_idxs or 2 in position_idxs:
                self.position_mode = "BothSide"
                return 1 if is_buy else 2
        except Exception as e:
            logger.warning(f"Could not determine position mode dynamically: {e}")

        self.position_mode = "MergedSingle"
        return 0

    def fetch_real_closed_pnl(self, symbol: str) -> tuple:
        """Centralized helper to query Bybit's /v5/position/closed-pnl endpoint."""
        real_closed_pnl = None
        real_fee = 0.0
        avg_exit_price = 0.0
        try:
            formatted_symbol = (
                symbol.replace("/", "").replace(":USDT", "").replace("_", "").upper()
            )
            resp = self.exchange.private_get_v5_position_closed_pnl(
                {"category": "linear", "symbol": formatted_symbol, "limit": 1}
            )
            records = resp.get("result", {}).get("list", [])
            if records:
                latest = records[0]
                raw_pnl = latest.get("closedPnl")
                real_closed_pnl = safe_float(raw_pnl, None) if raw_pnl not in (None, "") else None
                real_fee = abs(safe_float(latest.get("openFee"), 0.0)) + abs(
                    safe_float(latest.get("closeFee"), 0.0)
                )
                avg_exit_price = safe_float(latest.get("avgExitPrice"), 0.0)
        except Exception as e:
            logger.warning(f"[{symbol}] fetch_real_closed_pnl error: {e}")

        return real_closed_pnl, real_fee, avg_exit_price

    def fetch_recent_executions(self, symbol: str, lookback_ms: int = 300_000) -> List[Dict[str, Any]]:
        """Queries /v5/execution/list for the authoritative fill history."""
        try:
            formatted = symbol.replace("/", "").replace(":USDT", "").replace("_", "").upper()
            since = int((time.time() - lookback_ms / 1000) * 1000)
            resp = self.exchange.private_get_v5_execution_list({
                "category": "linear",
                "symbol": formatted,
                "startTime": since,
                "limit": 50,
            })
            return resp.get("result", {}).get("list", []) or []
        except Exception as e:
            logger.warning(f"[{symbol}] execution/list lookup failed: {e}")
            return []

    def _verify_sl(
        self,
        ccxt_symbol: str,
        formatted_sl: float,
        target_sl: Optional[float],
    ) -> bool:
        if not target_sl or target_sl <= 0:
            return True

        max_attempts = (
            SL_VERIFY_MAX_ATTEMPTS_TESTNET if BYBIT_TESTNET else SL_VERIFY_MAX_ATTEMPTS_LIVE
        )
        last_live_sl = 0.0

        exchange_symbol = ccxt_symbol
        if hasattr(self, 'config') and isinstance(self.config, dict):
            symbols_map = self.config.get("symbols", {})
            exchange_symbol = symbols_map.get(ccxt_symbol, symbols_map.get(ccxt_symbol.split('/')[0], ccxt_symbol))

        raw_symbol = (
            exchange_symbol.replace("/", "").replace(":USDT", "").upper()
        )

        for attempt in range(max_attempts):
            sleep_time = 0.5 + (attempt * 0.25)
            time.sleep(sleep_time)
            try:
                resp = self.exchange.private_get_v5_position_list({
                    "category": "linear", "symbol": raw_symbol
                })
                entries = resp.get("result", {}).get("list", [])
                for p in entries:
                    size = safe_float(p.get("size", 0))
                    if size > MIN_DUST_THRESHOLD:
                        live_sl = safe_float(p.get("stopLoss", 0))
                        last_live_sl = live_sl
                        if live_sl > 0 and abs(live_sl - formatted_sl) / max(formatted_sl, 1e-9) < 0.01:
                            logger.info(
                                f"[SL VERIFIED] {ccxt_symbol} SL=${live_sl} "
                                f"confirmed on exchange (attempt {attempt + 1}/{max_attempts})."
                            )
                            return True
            except Exception as verify_err:
                logger.debug(
                    f"[{ccxt_symbol}] SL verification attempt {attempt + 1} failed: {verify_err}"
                )

        logger.error(
            f"[SL VERIFY FAILED] {ccxt_symbol} expected ${formatted_sl}, "
            f"exchange reports ${last_live_sl} after {max_attempts} attempts."
        )
        return False

    def set_position_trading_stop(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_idx: Optional[int] = None,
        direction: str = "",
    ) -> bool:
        try:
            ccxt_symbol = self.format_ccxt_futures_symbol(symbol)
            
            exchange_symbol = ccxt_symbol
            if hasattr(self, 'config') and isinstance(self.config, dict):
                symbols_map = self.config.get("symbols", {})
                exchange_symbol = symbols_map.get(symbol, symbols_map.get(ccxt_symbol, ccxt_symbol))

            formatted_symbol = (
                exchange_symbol.replace("/", "")
                .replace(":USDT", "")
                .replace("-", "")
                .replace("_", "")
                .upper()
            )

            if position_idx is None:
                position_idx = self._get_position_idx(ccxt_symbol, direction)

            tpsl_mode = "Full"

            params = {
                "category": "linear",
                "symbol": formatted_symbol,
                "tpslMode": tpsl_mode,
                "positionIdx": position_idx,
                "slTriggerBy": "LastPrice",
                "tpTriggerBy": "LastPrice",
            }

            if stop_loss is not None and stop_loss > 0:
                formatted_sl = safe_float(self.exchange.price_to_precision(ccxt_symbol, stop_loss))
                params["stopLoss"] = str(formatted_sl)
                params["slOrderType"] = "Market"
            else:
                params["stopLoss"] = "0"
                formatted_sl = 0.0

            if take_profit is not None and take_profit > 0:
                formatted_tp = safe_float(self.exchange.price_to_precision(ccxt_symbol, take_profit))
                params["takeProfit"] = str(formatted_tp)
                params["tpOrderType"] = "Market"
            else:
                params["takeProfit"] = "0"
                params["tpOrderType"] = ""

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

            for api_attempt in range(SL_API_MAX_RETRIES):
                try:
                    response = method(params)
                except Exception as api_err:
                    logger.warning(
                        f"[{symbol}] trading-stop API attempt {api_attempt + 1} raised: {api_err}"
                    )
                    response = None

                if isinstance(response, dict) and response.get("retCode") == 0:
                    if self._verify_sl(ccxt_symbol, formatted_sl, stop_loss):
                        logger.info(
                            f"[{symbol}] Successfully attached SL (${stop_loss}) / "
                            f"TP (${take_profit}) on Bybit."
                        )
                        return True
                    else:
                        logger.warning(
                            f"[{symbol}] trading-stop retCode=0 but verification failed "
                            f"(attempt {api_attempt + 1}/{SL_API_MAX_RETRIES}). Retrying API call..."
                        )
                else:
                    ret_code = response.get("retCode") if isinstance(response, dict) else "N/A"
                    ret_msg = response.get("retMsg") if isinstance(response, dict) else "N/A"
                    logger.error(
                        f"[SL SET ERROR] Symbol: {formatted_symbol} | "
                        f"retCode: {ret_code} | retMsg: {ret_msg}"
                    )

                time.sleep(1.0 + api_attempt)

            logger.error(
                f"[SL SET FAILED] {symbol} — giving up after {SL_API_MAX_RETRIES} API attempts."
            )
            return False

        except Exception as e:
            logger.exception(f"[SL SET EXCEPTION] Failed to set Stop Loss for {symbol}: {e}")
            return False

    async def fetch_ticker_direct_async(self, symbol: str) -> float:
        ccxt_symbol = self.format_ccxt_futures_symbol(symbol)
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
        formatted_symbols = [self.format_ccxt_futures_symbol(s) for s in symbols]
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
        ccxt_symbol = self.format_ccxt_futures_symbol(symbol)
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
            f"(Live: ${exec_price:.5f} vs Strategy:${strategy_entry_price:.5f})."
        )
        return True, "Price deviation within accepted tolerance."

    async def get_futures_position_async(self, symbol: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.get_futures_position, symbol)

    def get_futures_position(self, symbol: str) -> Dict[str, Any]:
        # Issue 1 Resolution: Use self.resolve_symbol to dynamically resolve CCXT symbol for position queries
        exchange_symbol = self.resolve_symbol(symbol)

        clean_target = symbol.replace("/", "").replace(":", "").replace("_", "").replace("-", "").upper()
        try:
            pos = self.exchange.fetch_position(exchange_symbol)
            if pos:
                pos_symbol = str(pos.get('symbol', '')).replace("/", "").replace(":", "").replace("_", "").replace("-", "").upper()
                contracts = safe_float(pos.get('contracts', 0.0))

                if clean_target in pos_symbol or pos_symbol in clean_target or pos_symbol == exchange_symbol.replace("/", "").replace(":", "").upper():
                    if contracts > 0:
                        info = pos.get("info", {}) or {}

                        sl_raw = pos.get("stopLoss")
                        tp_raw = pos.get("takeProfit")
                        sl = safe_float(sl_raw) if sl_raw not in (None, "") else safe_float(info.get("stopLoss", 0))
                        tp = safe_float(tp_raw) if tp_raw not in (None, "") else safe_float(info.get("takeProfit", 0))

                        raw_lev = pos.get('leverage')
                        if raw_lev in (None, "", 0, 0.0):
                            raw_lev = info.get('leverage', 1.0)
                        lev_val = safe_float(raw_lev, 1.0)

                        return {
                            "symbol": symbol,
                            "side": str(pos.get('side', '')).upper(),
                            "contracts": contracts,
                            "entry_price": safe_float(pos.get('entryPrice', 0.0)),
                            "stop_loss": sl,
                            "take_profit": tp,
                            "leverage": lev_val,
                            "unrealized_pnl": safe_float(pos.get('unrealizedPnl', 0.0)),
                            "error": False
                        }
            return {
                "symbol": symbol, "side": "NONE", "contracts": 0.0,
                "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
                "leverage": 1.0, "unrealized_pnl": 0.0, "error": False
            }
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            logger.error(f"Failed to fetch futures position for {exchange_symbol}: {e}")
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

            pos_idx = self._get_position_idx(
                self.format_ccxt_futures_symbol(symbol), side
            )

            if target_sl > 0:
                pos_confirmed = False

                for attempt in range(10):
                    ccxt_symbol = self.format_ccxt_futures_symbol(symbol)
                    pos_check = self.get_futures_position(ccxt_symbol)
                    if not pos_check.get("error") and pos_check["contracts"] > MIN_DUST_THRESHOLD:
                        pos_confirmed = True
                        break
                    time.sleep(0.5)

                executed_qty = safe_float(order.get("executed_qty", 0.0))
                if not pos_confirmed and executed_qty > MIN_DUST_THRESHOLD:
                    logger.warning(
                        f"[{symbol}] Position endpoint pending propagation, "
                        f"falling back to filled order qty ({executed_qty}). Attempting SL attachment..."
                    )
                    pos_confirmed = True

                if pos_confirmed:
                    sl_success = self.set_position_trading_stop(
                        symbol=symbol, stop_loss=target_sl,
                        position_idx=pos_idx, direction=side
                    )
                else:
                    logger.error(f"[{symbol}] Position contracts not reflected on exchange after execution.")
                    sl_success = False

                order["stop_loss_attached"] = sl_success

                if not sl_success:
                    logger.warning(
                        f"⚠️ [SL ATTACH RETRY NEEDED] Could not verify exchange Stop Loss for {symbol} at ${target_sl}. "
                        f"Trade will remain open; reconciler will auto-reattach SL on next sync cycle."
                    )
                    return order

        return order

    def order_futures_bybit(self, symbol: str, direction: str, amount_usd: float,
                            entry_price: float = 0.0, stop_loss: float = 0.0, take_profit: float = 0.0,
                            leverage: float = 5.0, account_balance: float = 100.0, risk_pct: float = 1.0) -> Dict[str, Any]:
        ccxt_symbol = self.format_ccxt_futures_symbol(symbol)
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

        # Ensure 5x isolated leverage is configured on Bybit prior to order submission
        try:
            self.exchange.set_margin_mode("isolated", ccxt_symbol, params={"category": "linear"})
            self.exchange.set_leverage(int(leverage), ccxt_symbol, params={"category": "linear"})
            logger.info(f"[{symbol}] Margin mode 'isolated' & leverage {int(leverage)}x enforced on Bybit.")
        except Exception as lev_err:
            logger.error(f"[{symbol}] FAILED to enforce margin mode/leverage: {lev_err}")
            try:
                pos_check_pre = self.get_futures_position(ccxt_symbol)
                actual_lev_pre = safe_float(pos_check_pre.get("leverage", 0), 0.0)
                if actual_lev_pre and abs(actual_lev_pre - leverage) > 0.5:
                    return {
                        "status": "FAILED",
                        "error": (
                            f"Leverage mismatch: requested {leverage}x, "
                            f"exchange reports {actual_lev_pre}x. Order aborted."
                        ),
                    }
            except Exception as verify_err:
                return {
                    "status": "FAILED",
                    "error": f"Could not verify leverage after failed set_leverage: {lev_err} | verify: {verify_err}",
                }

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

        prefer_taker_due_to_momentum = False
        if entry_price > 0 and ref_price > 0:
            deviation_pct = abs(ref_price - entry_price) / entry_price
            if deviation_pct > 0.005:
                prefer_taker_due_to_momentum = True
                logger.info(
                    f"[{symbol}] Momentum override: {deviation_pct * 100:.2f}% deviation from signal "
                    f"→ using IOC taker for immediate fill."
                )

        if USE_MAKER_ORDERS and not prefer_taker_due_to_momentum:
            try:
                tick_size = float(self.exchange.markets[ccxt_symbol]['precision']['price'])
            except Exception:
                tick_size = 0.0

            book = self.fetch_ticker_data(ccxt_symbol)
            best_bid = book["bid"]
            best_ask = book["ask"]

            if is_long and best_bid > 0:
                limit_price = best_bid - tick_size if tick_size > 0 else best_bid * 0.9995
            elif not is_long and best_ask > 0:
                limit_price = best_ask + tick_size if tick_size > 0 else best_ask * 1.0005
            else:
                limit_price = ref_price

            if is_long and best_ask > 0 and limit_price >= best_ask:
                limit_price = best_ask - tick_size if tick_size > 0 else best_ask * 0.999
            if (not is_long) and best_bid > 0 and limit_price <= best_bid:
                limit_price = best_bid + tick_size if tick_size > 0 else best_bid * 1.001

            params = {"timeInForce": "PostOnly", "reduceOnly": False}
        else:
            limit_price = (
                ref_price * (1.0 + MAX_SLIPPAGE_PCT)
                if is_long
                else ref_price * (1.0 - MAX_SLIPPAGE_PCT)
            )
            params = {"timeInForce": "IOC"}

        try:
            formatted_qty = safe_float(self.exchange.amount_to_precision(ccxt_symbol, raw_qty))
            formatted_limit_price = safe_float(
                self.exchange.price_to_precision(ccxt_symbol, limit_price)
            )
        except Exception:
            formatted_qty = round(raw_qty, 4)
            formatted_limit_price = round(limit_price, 6)

        target_sl = 0.0
        if stop_loss > 0:
            if is_long and stop_loss >= ref_price:
                logger.error(f"[{symbol}] Invalid Long Stop Loss (${stop_loss:.5f}) >= Ref Price (${ref_price:.5f}). Stripping SL.")
            elif not is_long and stop_loss <= ref_price:
                logger.error(f"[{symbol}] Invalid Short Stop Loss (${stop_loss:.5f}) <= Ref Price (${ref_price:.5f}). Stripping SL.")
            else:
                target_sl = stop_loss

        target_tp = 0.0
        if take_profit > 0:
            if is_long and take_profit <= ref_price:
                logger.error(f"[{symbol}] Invalid Long Take Profit (${take_profit:.5f}) <= Ref Price (${ref_price:.5f}). Stripping TP.")
            elif not is_long and take_profit >= ref_price:
                logger.error(f"[{symbol}] Invalid Short Take Profit (${take_profit:.5f}) >= Ref Price (${ref_price:.5f}). Stripping TP.")
            else:
                target_tp = take_profit

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

            final_order = order
            deadline = time.time() + 20.0
            while time.time() < deadline:
                status = str(final_order.get("status", "")).lower()
                if status in ("closed", "canceled", "rejected", "expired"):
                    break
                time.sleep(1.5)
                try:
                    final_order = self.exchange.fetch_order(order_id, ccxt_symbol)
                except Exception as poll_err:
                    logger.debug(f"[{symbol}] Order poll error: {poll_err}")
                    break

            status = str(final_order.get("status", "")).lower()
            filled = safe_float(final_order.get("filled"), 0.0)
            avg_price = safe_float(final_order.get("average") or final_order.get("price"), 0.0)

            if status == "canceled" and filled > 0 and filled < formatted_qty:
                try:
                    trades = self.exchange.fetch_my_trades(ccxt_symbol, limit=10)
                    recent = [t for t in trades if str(t.get("order")) == str(order_id)]
                    if recent:
                        total_cost = sum(float(t.get("price", 0)) * float(t.get("amount", 0)) for t in recent)
                        total_qty = sum(float(t.get("amount", 0)) for t in recent)
                        avg_price = total_cost / total_qty if total_qty > 0 else avg_price
                        filled = total_qty
                        logger.info(
                            f"[{symbol}] Partial fill on cancelled PostOnly: "
                            f"{filled} @ ${avg_price:.6f} (computed from {len(recent)} trades)"
                        )
                except Exception as trade_err:
                    logger.warning(f"[{symbol}] Trade recomputation failed: {trade_err}")

            if filled <= 0 and USE_MAKER_ORDERS:
                logger.warning(
                    f"[{symbol}] PostOnly unfilled after 20s — cancelling and retrying as IOC taker."
                )
                try:
                    self.exchange.cancel_order(order_id, ccxt_symbol)
                except Exception:
                    pass

                fresh_book = self.fetch_ticker_data(ccxt_symbol)
                if is_long:
                    ioc_limit_price = fresh_book["ask"] * (1.0 + MAX_SLIPPAGE_PCT) if fresh_book["ask"] > 0 else ref_price * (1.0 + MAX_SLIPPAGE_PCT)
                else:
                    ioc_limit_price = fresh_book["bid"] * (1.0 - MAX_SLIPPAGE_PCT) if fresh_book["bid"] > 0 else ref_price * (1.0 - MAX_SLIPPAGE_PCT)

                try:
                    formatted_ioc_price = safe_float(
                        self.exchange.price_to_precision(ccxt_symbol, ioc_limit_price)
                    )
                except Exception:
                    formatted_ioc_price = round(ioc_limit_price, 6)

                logger.info(
                    f"[{symbol}] Retrying as IOC taker @ Limit ${formatted_ioc_price:.6f} (fallback)."
                )

                try:
                    fallback_order = self.exchange.create_order(
                        symbol=ccxt_symbol,
                        type='limit',
                        side=side,
                        amount=formatted_qty,
                        price=formatted_ioc_price,
                        params={"timeInForce": "IOC", "reduceOnly": False},
                    )
                    final_order = fallback_order
                    order_id = fallback_order.get("id")
                    filled = safe_float(fallback_order.get("filled"), 0.0)
                    avg_price = safe_float(
                        fallback_order.get("average") or fallback_order.get("price"), 0.0
                    )

                    if filled <= 0 and order_id:
                        time.sleep(1.5)
                        try:
                            refreshed = self.exchange.fetch_order(order_id, ccxt_symbol)
                            final_order = refreshed
                            filled = safe_float(refreshed.get("filled"), 0.0)
                            avg_price = safe_float(
                                refreshed.get("average") or refreshed.get("price"), 0.0
                            )
                        except Exception as fetch_err:
                            logger.warning(f"[{symbol}] IOC fallback fetch error: {fetch_err}")
                except Exception as ioc_err:
                    logger.error(f"[{symbol}] IOC fallback order failed: {ioc_err}")
                    return {"status": "FAILED", "error": f"IOC fallback failed: {ioc_err}"}

            if filled <= 0:
                try:
                    time.sleep(1.0)
                    final_pos_check = self.get_futures_position(ccxt_symbol)
                    if not final_pos_check.get("error") and final_pos_check["contracts"] > MIN_DUST_THRESHOLD:
                        logger.warning(
                            f"[{symbol}] Order appeared unfilled but position EXISTS on exchange "
                            f"({final_pos_check['contracts']} contracts @ ${final_pos_check['entry_price']}). "
                            f"Treating as filled."
                        )
                        filled = final_pos_check["contracts"]
                        avg_price = final_pos_check["entry_price"]
                        status = "closed"
                except Exception as pos_err:
                    logger.warning(f"[{symbol}] Final position check failed: {pos_err}")

            if filled <= 0:
                logger.warning(f"[{symbol}] Both PostOnly and IOC fallback unfilled — aborting.")
                try:
                    self.exchange.cancel_order(order_id, ccxt_symbol)
                except Exception:
                    pass
                return {"status": "FAILED", "error": "Order unfilled after PostOnly + IOC retry."}

            if avg_price <= 0:
                avg_price = ref_price

            executed_qty = filled
            fill_price = avg_price

            logger.info(
                f"[{symbol}] Futures Order Filled: Side {side.upper()}, "
                f"Price ${fill_price:.6f}, Qty {executed_qty}, Status={status}"
            )

            try:
                time.sleep(1.0)
                verify_pos = self.get_futures_position(ccxt_symbol)
                if verify_pos.get("contracts", 0) > MIN_DUST_THRESHOLD:
                    actual_lev = safe_float(verify_pos.get("leverage", 0), 0.0)
                    if actual_lev and abs(actual_lev - leverage) > 0.5:
                        logger.critical(
                            f"🚨 [{symbol}] LEVERAGE MISMATCH: requested {leverage}x, "
                            f"position opened at {actual_lev}x. Closing to prevent oversized risk."
                        )
                        self.close_live_position_bybit(
                            symbol=symbol, position_size=executed_qty,
                            current_price=fill_price, outcome="LEVERAGE_MISMATCH"
                        )
                        return {
                            "status": "FAILED",
                            "error": f"Leverage mismatch: {actual_lev}x vs requested {leverage}x",
                        }
                    elif actual_lev:
                        logger.info(f"[{symbol}] Post-fill leverage verified: {actual_lev}x.")
            except Exception as lev_verify_err:
                logger.warning(f"[{symbol}] Post-fill leverage verification failed: {lev_verify_err}")

            sl_attached = False
            if target_sl > 0 or target_tp > 0:
                pos_confirmed = False

                for attempt in range(10):
                    pos_check = self.get_futures_position(ccxt_symbol)
                    if not pos_check.get("error") and pos_check["contracts"] > MIN_DUST_THRESHOLD:
                        pos_confirmed = True
                        break
                    time.sleep(0.5)

                if not pos_confirmed and executed_qty > MIN_DUST_THRESHOLD:
                    logger.warning(
                        f"[{symbol}] Position endpoint pending propagation, "
                        f"falling back to filled order qty ({executed_qty}). Attempting SL attachment..."
                    )
                    pos_confirmed = True

                if pos_confirmed:
                    sl_attached = self.set_position_trading_stop(
                        symbol=symbol,
                        stop_loss=target_sl if target_sl > 0 else None,
                        take_profit=target_tp if target_tp > 0 else None,
                        position_idx=None,
                        direction=dir_clean
                    )
                else:
                    logger.error(f"[{symbol}] Position contracts not reflected on exchange after execution.")
                    sl_attached = False

                if target_sl > 0 and not sl_attached:
                    event_bus.arm_local_sl_guard(
                        symbol=symbol,
                        direction=dir_clean,
                        quantity=executed_qty,
                        stop_loss=target_sl,
                    )
                    logger.warning(
                        f"[{symbol}] Exchange SL attach failed — local SL guard ARMED at "
                        f"${target_sl:.5f} for {executed_qty} units."
                    )

                    if BYBIT_TESTNET:
                        logger.warning(
                            f"[{symbol}] Testnet propagation delay suspected — "
                            f"NOT emergency-closing. Local guard will protect the position."
                        )
                        return {
                            "status": "SUCCESS",
                            "order_id": order_id,
                            "fill_price": fill_price,
                            "executed_qty": executed_qty,
                            "cost_usd": trade_amount_usd,
                            "stop_loss_attached": False,
                            "raw_order": final_order,
                        }

                    logger.critical(
                        f"🚨 [EMERGENCY GUARD] Exchange SL attachment FAILED for {symbol} @ ${target_sl}. "
                        f"Cancelling pending orders and closing position."
                    )

                    try:
                        open_orders = self.exchange.fetch_open_orders(ccxt_symbol)
                        for oo in open_orders:
                            try:
                                self.exchange.cancel_order(oo["id"], ccxt_symbol)
                                logger.info(f"[{symbol}] Cancelled pending order {oo['id']}.")
                            except Exception as cx_err:
                                logger.warning(f"[{symbol}] Cancel {oo['id']} failed: {cx_err}")
                    except Exception as open_err:
                        logger.warning(f"[{symbol}] fetch_open_orders failed: {open_err}")

                    close_res = self.close_live_position_bybit(
                        symbol=symbol,
                        position_size=executed_qty,
                        current_price=fill_price,
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
                        "error": "Emergency close triggered: exchange SL failed to attach.",
                    }

            return {
                "status": "SUCCESS",
                "order_id": order_id,
                "fill_price": fill_price,
                "executed_qty": executed_qty,
                "cost_usd": trade_amount_usd,
                "stop_loss_attached": sl_attached,
                "raw_order": final_order,
            }

        except Exception as e:
            logger.error(f"[{symbol}] Bybit Futures Order Exception: {e}")
            return {"status": "FAILED", "error": str(e)}

    def close_live_position_bybit(
        self,
        symbol: str,
        position_size: float,
        current_price: float,
        outcome: str = "CLOSE",
    ) -> Dict[str, Any]:
        ccxt_symbol = self.format_ccxt_futures_symbol(symbol)

        try:
            open_orders = self.exchange.fetch_open_orders(ccxt_symbol)
            for oo in open_orders:
                try:
                    self.exchange.cancel_order(oo["id"], ccxt_symbol)
                    logger.info(f"[{symbol}] Cancelled pending order {oo['id']} before close.")
                except Exception as cx_err:
                    logger.warning(f"[{symbol}] Cancel {oo['id']} failed: {cx_err}")
        except Exception as open_err:
            logger.debug(f"[{symbol}] fetch_open_orders during close: {open_err}")

        pos_info = self.get_futures_position(ccxt_symbol)
        if pos_info.get("error"):
            logger.error(f"[{symbol}] Cannot attempt position close — API error during verification.")
            return {"status": "FAILED", "error": "Network/API glitch preventing close verification"}

        contracts = pos_info["contracts"]
        current_side = str(pos_info["side"]).upper()

        conn = get_db_connection()
        trade_id = None
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id FROM trade_setups 
                        WHERE UPPER(REPLACE(REPLACE(pair, '/', ''), '_', '')) = %s 
                          AND trade_state IN ('OPEN', 'EXECUTED', 'BE_LOCKED', 'TRAILING')
                        ORDER BY id DESC LIMIT 1;
                        """,
                        (symbol.replace("/", "").upper(),),
                    )
                    row = cur.fetchone()
                    if row:
                        trade_id = row[0]
            finally:
                release_db_connection(conn)

        if contracts >= MIN_DUST_THRESHOLD:
            close_side = "sell" if current_side in ["BUY", "LONG"] else "buy"
            reduce_position_idx = self._get_position_idx(ccxt_symbol, current_side)

            try:
                close_params: Dict[str, Any] = {"reduceOnly": True}
                if self.position_mode == "BothSide" or reduce_position_idx in (1, 2):
                    close_params["positionIdx"] = reduce_position_idx

                order = self.exchange.create_order(
                    symbol=ccxt_symbol,
                    type="market",
                    side=close_side,
                    amount=contracts,
                    params=close_params,
                )
                fill_price = safe_float(order.get("average") or order.get("price"), current_price)
                if fill_price <= 0:
                    fill_price = current_price

                event_bus.disarm_local_sl_guard(symbol)

                return {
                    "status": "SUCCESS",
                    "order_id": order.get("id"),
                    "exit_price": fill_price,
                    "executed_qty": contracts,
                }
            except Exception as close_err:
                logger.error(f"[{symbol}] Market close failed: {close_err}")
                return {"status": "FAILED", "error": str(close_err)}

        logger.warning(
            f"[{symbol}] Close requested but exchange reports zero contracts. "
            f"Treating as ghost close (trade_id={trade_id})."
        )
        event_bus.disarm_local_sl_guard(symbol)

        if trade_id:
            real_closed_pnl = None
            real_fee = 0.0
            verified_exit = 0.0
            try:
                formatted_symbol = symbol.replace("/", "").replace(":USDT", "").replace("_", "").upper()
                cpnl_resp = self.exchange.private_get_v5_position_closed_pnl(
                    {"category": "linear", "symbol": formatted_symbol, "limit": 1}
                )
                records = cpnl_resp.get("result", {}).get("list", [])
                if records:
                    latest = records[0]
                    raw_pnl = latest.get("closedPnl")
                    real_closed_pnl = safe_float(raw_pnl, None) if raw_pnl not in (None, "") else None
                    real_fee = abs(safe_float(latest.get("openFee"), 0.0)) + abs(
                        safe_float(latest.get("closeFee"), 0.0)
                    )
                    verified_exit = safe_float(latest.get("avgExitPrice"), 0.0)
            except Exception as pnl_err:
                logger.warning(f"[{symbol}] Ghost-close PnL lookup failed: {pnl_err}")

            conn = get_db_connection()
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT direction, entry_price, position_size, account_balance "
                            "FROM trade_setups WHERE id = %s;",
                            (trade_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            direction, entry_p, pos_qty, acc_bal = row
                            entry_p = safe_float(entry_p)

                            if verified_exit > 0:
                                exit_price = verified_exit
                            else:
                                exit_price = entry_p
                                logger.warning(
                                    f"[{symbol}] No verified exit price for ghost trade #{trade_id}. "
                                    f"Falling back to entry_price (${entry_p:.5f}); PnL will reflect only fees."
                                )

                            pnl_usd, pnl_pct, outcome_verified = calculate_pnl(
                                direction=direction,
                                entry_price=entry_p,
                                current_price=exit_price,
                                quantity=safe_float(pos_qty),
                                account_balance=safe_float(acc_bal, 100.0),
                                total_fees=real_fee,
                                exchange_closed_pnl=real_closed_pnl,
                            )

                            if real_closed_pnl is None and exit_price == entry_p:
                                outcome_verified = "UNKNOWN"

                            finalize_trade_in_db(
                                trade_id,
                                exit_price,
                                pnl_usd,
                                pnl_pct,
                                outcome_verified,
                                fee_usd=real_fee,
                            )
                finally:
                    release_db_connection(conn)

        return {
            "status": "FORCE_CLOSED_DB_ONLY",
            "order_id": "GHOST_POSITION_DB_CLOSED",
            "exit_price": current_price,
            "executed_qty": position_size,
        }


LiveExecutionEngine = BybitFuturesLiveExecutor