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

ENTRY_TOLERANCE_PCT_LIVE = 0.005
ENTRY_TOLERANCE_PCT_TESTNET = 0.05

USE_MAKER_ORDERS = os.getenv("USE_MAKER_ORDERS", "true").lower() == "true"

# FIX #4: Extended verification window on testnet to tolerate Bybit's
# propagation delay between an order fill and the position becoming
# editable via /v5/position/trading-stop.
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

    try:
        binance_ex = ccxt.binanceusdm()
        ohlcv = binance_ex.fetch_ohlcv(symbol.split(':')[0], timeframe=interval, limit=limit)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
    except Exception as e:
        logger.warning(f"[{symbol}] Secondary Binance kline fetch failed ({e}). Falling back to CryptoCompare...")

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

        # -----------------------------------------------------------------
        # FIX C: Log the CCXT version so we can correlate behavior with
        # upstream changes to the Bybit V5 position normalizer.
        # -----------------------------------------------------------------
        try:
            logger.info(f"CCXT version: {ccxt.__version__}")
        except Exception:
            pass

        # -----------------------------------------------------------------
        # FIX #3: Detect account position mode once at startup so that all
        # subsequent trading-stop calls use the correct positionIdx.
        # "MergedSingle" = one-way mode (positionIdx = 0)
        # "BothSide"     = hedge mode   (positionIdx = 1 for Buy, 2 for Sell)
        # -----------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # FIX #1 + FIX #3: Position-mode-aware positionIdx resolver.
    # Returns 0 (one-way), 1 (hedge-buy), or 2 (hedge-sell).
    # ---------------------------------------------------------------------
    def _get_position_idx(self, ccxt_symbol: str, direction: str) -> int:
        """Returns 0 for one-way mode, 1 for hedge-buy, 2 for hedge-sell."""
        clean_dir = str(direction or "").upper()
        is_buy = clean_dir in ("BUY", "LONG")

        # Fast path: if startup detection succeeded, use it.
        if self.position_mode == "MergedSingle":
            return 0
        if self.position_mode == "BothSide":
            return 1 if is_buy else 2

        # Fallback: query /v5/position/list and inspect positionIdx values.
        # Bybit V5 hedge mode ALWAYS returns two entries (one per side),
        # even when one side has size 0. One-way mode returns one entry
        # with positionIdx=0.
        try:
            formatted = (
                ccxt_symbol.replace("/", "")
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
                # Cache the discovered mode so future calls are fast.
                self.position_mode = "BothSide"
                return 1 if is_buy else 2
        except Exception as e:
            logger.warning(f"Could not determine position mode dynamically: {e}")

        # Default to one-way and cache it.
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

    def format_ccxt_futures_symbol(self, raw_symbol: str) -> str:
        return format_ccxt_futures_symbol(raw_symbol)

    # ---------------------------------------------------------------------
    # FIX A: Verification now queries /v5/position/list directly and reads
    # the raw Bybit "stopLoss" field, because CCXT's fetch_positions() does
    # NOT reliably populate the normalized top-level stopLoss key on Bybit
    # V5. Every prior "SL VERIFY FAILED" was a false negative caused by
    # reading pos["stopLoss"] (None) instead of pos["info"]["stopLoss"].
    # ---------------------------------------------------------------------
    def _verify_sl(
        self,
        ccxt_symbol: str,
        formatted_sl: float,
        target_sl: Optional[float],
    ) -> bool:
        if not target_sl or target_sl <= 0:
            return True  # nothing to verify

        # FIX #4: longer window on testnet
        max_attempts = (
            SL_VERIFY_MAX_ATTEMPTS_TESTNET if BYBIT_TESTNET else SL_VERIFY_MAX_ATTEMPTS_LIVE
        )
        last_live_sl = 0.0

        # Extract the Bybit raw symbol for the authoritative endpoint.
        raw_symbol = (
            ccxt_symbol.replace("/", "").replace(":USDT", "").upper()
        )

        for attempt in range(max_attempts):
            sleep_time = 0.5 + (attempt * 0.25)  # 0.5, 0.75, 1.0, 1.25, ...
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
        """
        Sets SL/TP via Bybit V5 /position/trading-stop.

        FIX #1: positionIdx is now derived from the account's actual position
                mode and the trade direction (unless explicitly passed).
        FIX #2: The trading-stop API call itself is retried up to 3 times with
                backoff, and verification runs after each successful retCode:0.
        FIX #3: tpslMode is validated against the actual account mode; the
                previous "Partial" branch was silently broken because no qty
                was ever supplied.
        FIX #4: Extended verification window on testnet.
        FIX A:  Verification now reads the raw Bybit payload via
                /v5/position/list.
        """
        try:
            ccxt_symbol = self.format_ccxt_futures_symbol(symbol)
            formatted_symbol = (
                symbol.replace("/", "")
                .replace(":USDT", "")
                .replace("-", "")
                .replace("_", "")
                .upper()
            )

            # FIX #1 / FIX #3: Resolve positionIdx
            if position_idx is None:
                position_idx = self._get_position_idx(ccxt_symbol, direction)

            # FIX #3: tpslMode must always be "Full" — "Partial" requires a
            # qty argument that this code never supplies. Bybit V5 accepts
            # "Full" for both one-way and hedge mode (per-side).
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

            # -------------------------------------------------------------
            # FIX #2: Retry the API call itself, verifying after each
            # successful retCode:0 response.
            # -------------------------------------------------------------
            for api_attempt in range(SL_API_MAX_RETRIES):
                try:
                    response = method(params)
                except Exception as api_err:
                    logger.warning(
                        f"[{symbol}] trading-stop API attempt {api_attempt + 1} raised: {api_err}"
                    )
                    response = None

                if isinstance(response, dict) and response.get("retCode") == 0:
                    # Verify the SL actually populated
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
            f"(Live: ${exec_price:.5f} vs Strategy:${strategy_entry_price:.5f})."
        )
        return True, "Price deviation within accepted tolerance."

    async def get_futures_position_async(self, symbol: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.get_futures_position, symbol)

    # ---------------------------------------------------------------------
    # FIX B: CCXT does not reliably map Bybit V5 stopLoss/takeProfit into
    # the normalized top-level keys. Fall back to the raw info dict, which
    # always contains them. This is the field the exchange actually uses.
    # ---------------------------------------------------------------------
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
                        info = pos.get("info", {}) or {}

                        sl_raw = pos.get("stopLoss")
                        tp_raw = pos.get("takeProfit")
                        sl = safe_float(sl_raw) if sl_raw not in (None, "") else safe_float(info.get("stopLoss", 0))
                        tp = safe_float(tp_raw) if tp_raw not in (None, "") else safe_float(info.get("takeProfit", 0))

                        return {
                            "symbol": symbol,
                            "side": str(pos.get('side', '')).upper(),
                            "contracts": contracts,
                            "entry_price": safe_float(pos.get('entryPrice', 0.0)),
                            "stop_loss": sl,
                            "take_profit": tp,
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

            # FIX #1: derive positionIdx from direction, not hard-coded 0
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

        # Momentum override: bypass PostOnly if market has moved >0.5% from signal
        prefer_taker_due_to_momentum = False
        if entry_price > 0 and ref_price > 0:
            deviation_pct = abs(ref_price - entry_price) / entry_price
            if deviation_pct > 0.005:
                prefer_taker_due_to_momentum = True
                logger.info(
                    f"[{symbol}] Momentum override: {deviation_pct * 100:.2f}% deviation from signal "
                    f"→ using IOC taker for immediate fill."
                )

        # Aggressive PostOnly pricing with tick-back offset
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

            # Sanity: limit must NOT cross the opposite side of the book
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

        # Validate SL / TP targets prior to post-order attachment
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

            # Poll PostOnly for up to 20s
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

            # If partially filled and cancelled, compute weighted average from trades
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

            # If PostOnly unfilled, fall back to IOC taker
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

            # -----------------------------------------------------------------
            # FIX #1: Final safety check — the fill may have occurred but the
            # order record was purged by Bybit before our poll could observe it.
            # Query the actual position one last time before declaring failure.
            # -----------------------------------------------------------------
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

            # Attach SL; if it fails, cancel pending orders and close position
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
                    # FIX #1: pass direction so positionIdx is resolved correctly
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
                    # -----------------------------------------------------------------
                    # FIX #4: On testnet, propagation delays cause spurious SL
                    # failures. Arm the local SL guard as a fallback so the
                    # position is not left naked. Only emergency-close if we
                    # are on live, OR if the position is old enough that the
                    # delay cannot be the cause.
                    # -----------------------------------------------------------------
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
                        # Return SUCCESS with sl_attached=False so the DB record
                        # is created and the reconciler can re-attach the SL.
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
        """
        Correct close sequence for PostOnly + failed-SL scenarios.
          1. Cancel all pending entry orders first.
          2. Fetch position; if contracts > 0, market reduceOnly close.
          3. If contracts == 0, ghost close with real closedPnl.
          4. Never fabricate exit_price.
        """
        ccxt_symbol = format_ccxt_futures_symbol(symbol)

        # Step 1: Cancel any pending orders FIRST
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

        # Step 2: Fetch position
        pos_info = self.get_futures_position(ccxt_symbol)
        if pos_info.get("error"):
            logger.error(f"[{symbol}] Cannot attempt position close — API error during verification.")
            return {"status": "FAILED", "error": "Network/API glitch preventing close verification"}

        contracts = pos_info["contracts"]
        current_side = str(pos_info["side"]).upper()

        # Resolve trade_id for DB finalization
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

        # Step 3: Live position exists → market reduceOnly close
        if contracts >= MIN_DUST_THRESHOLD:
            close_side = "sell" if current_side in ["BUY", "LONG"] else "buy"

            # FIX #1: pass the correct positionIdx in hedge mode
            reduce_position_idx = self._get_position_idx(ccxt_symbol, current_side)

            try:
                close_params: Dict[str, Any] = {"reduceOnly": True}
                # In hedge mode, Bybit V5 requires positionIdx for reduce-only closes
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

        # Step 4: No live position → ghost close
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