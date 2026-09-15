import asyncio
import logging
import os
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

# =====================================================================
# FIXIE PROXY INITIALIZATION & ENVIRONMENT SETUP
# =====================================================================
FIXIE_URL = (
    os.getenv("FIXIE_URL")
    or os.getenv("HTTP_PROXY")
    or os.getenv("HTTPS_PROXY")
    or os.getenv("PROXY_URL")
)

if FIXIE_URL:
    # Ensure URL formatting clean-up
    FIXIE_URL = FIXIE_URL.strip('"').strip("'")
    
    # Export standard proxy environment variables for requests / urllib
    os.environ["HTTP_PROXY"] = FIXIE_URL
    os.environ["HTTPS_PROXY"] = FIXIE_URL
    os.environ["http_proxy"] = FIXIE_URL
    os.environ["https_proxy"] = FIXIE_URL
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,.supabase.co"
    os.environ["no_proxy"] = "localhost,127.0.0.1,.supabase.co"
    logger.info("Fixie proxy successfully integrated into environment execution pipeline.")

MIN_DUST_THRESHOLD = 0.001
MAX_SLIPPAGE_PCT = 0.002       # 0.2% max execution limit offset
MAX_ALLOWED_SPREAD_PCT = 0.003 # 0.3% max spread threshold
ENTRY_TOLERANCE_PCT = 0.005    # 0.5% max strategy-to-exchange price deviation ceiling


def safe_float(val, default=0.0):
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
        exchange_config = {
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'recvWindow': 20000,
                'adjustForTimeDifference': True
            }
        }

        # Inject Fixie Proxy into CCXT configuration if present
        if FIXIE_URL:
            exchange_config['proxies'] = {
                'http': FIXIE_URL,
                'https': FIXIE_URL
            }
            exchange_config['aiohttp_proxy'] = FIXIE_URL

        self.exchange = ccxt.bybit(exchange_config)

        if BYBIT_TESTNET:
            self.exchange.set_sandbox_mode(True)

        try:
            self.exchange.load_markets()
        except Exception as e:
            logger.warning(f"Could not refresh market structures on init: {e}")

    def format_ccxt_futures_symbol(self, raw_symbol: str) -> str:
        return format_ccxt_futures_symbol(raw_symbol)

    async def get_futures_position_async(self, symbol: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.get_futures_position, symbol)

    def get_futures_position(self, symbol: str) -> dict:
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        clean_target = symbol.replace("/", "").replace(":", "").replace("_", "").replace("-", "").upper()
        try:
            positions = self.exchange.fetch_positions([ccxt_symbol])
            for pos in positions:
                pos_symbol = str(pos.get('symbol', '')).replace("/", "").replace(":", "").replace("_", "").replace("-", "").upper()
                contracts = safe_float(pos.get('contracts', 0.0))
                
                # Match normalized symbols rather than exact string equality
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

    def fetch_ticker_data(self, symbol: str) -> dict:
        ccxt_symbol = format_ccxt_futures_symbol(symbol)
        try:
            ticker = self.exchange.fetch_ticker(ccxt_symbol)
            bid = safe_float(ticker.get('bid'))
            ask = safe_float(ticker.get('ask'))
            last = safe_float(ticker.get('last'))
            info = ticker.get('info', {})
            
            mark_price = safe_float(ticker.get('markPrice') or info.get('markPrice') or info.get('indexPrice'))

            if BYBIT_TESTNET and mark_price > 0:
                bid = mark_price
                ask = mark_price
                last = mark_price

            spread_pct = (ask - bid) / bid if (bid > 0 and ask > 0) else 0.0

            return {
                "bid": bid,
                "ask": ask,
                "last": last,
                "mark_price": mark_price,
                "spread_pct": spread_pct
            }
        except Exception as e:
            logger.error(f"[{ccxt_symbol}] Failed to fetch ticker data: {e}")
            return {"bid": 0.0, "ask": 0.0, "last": 0.0, "mark_price": 0.0, "spread_pct": 0.0}

    def fetch_ticker_price(self, symbol: str) -> float:
        data = self.fetch_ticker_data(symbol)
        if data["mark_price"] > 0 and BYBIT_TESTNET:
            return data["mark_price"]
        if data["bid"] > 0 and data["ask"] > 0:
            return (data["bid"] + data["ask"]) / 2.0
        return data["last"]

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
        leverage: int = 5.0,
        account_balance: float = 100.0,
        risk_pct: float = 1.0
    ) -> dict:
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

        # Check available USDT Balance
        available_usdt = self.fetch_available_usdt_balance()
        if available_usdt <= 1.0:
            return {"status": "FAILED", "error": f"Insufficient live USDT balance: ${available_usdt:.2f}"}

        ticker_data = self.fetch_ticker_data(ccxt_symbol)
        ref_price = ticker_data["ask"] if is_long else ticker_data["bid"]
        spread_pct = ticker_data["spread_pct"]

        if ref_price <= 0:
            return {"status": "FAILED", "error": "Invalid reference ticker price prior to execution"}

        # Strategy Entry Price Deviation Ceiling Check
        if entry_price > 0:
            price_dev = abs(ref_price - entry_price) / entry_price
            if price_dev > ENTRY_TOLERANCE_PCT:
                logger.error(
                    f"[{symbol}] Execution Rejected: Live reference price (${ref_price:.5f}) deviates "
                    f"from Strategy Entry Price (${entry_price:.5f}) by {price_dev * 100:.2f}% (Limit: {ENTRY_TOLERANCE_PCT * 100}%)."
                )
                return {"status": "FAILED", "error": f"Strategy entry price deviation too high ({price_dev * 100:.2f}%)"}

        if spread_pct > MAX_ALLOWED_SPREAD_PCT and not BYBIT_TESTNET:
            logger.warning(f"[{symbol}] Order Rejected: High Spread detected ({spread_pct * 100:.3f}% > Max allowed {MAX_ALLOWED_SPREAD_PCT * 100:.2f}%).")
            return {"status": "FAILED", "error": f"Spread too high ({spread_pct * 100:.3f}%)"}

        try:
            self.exchange.set_leverage(leverage, ccxt_symbol)
        except Exception as lev_err:
            logger.debug(f"[{symbol}] Leverage setting notice: {lev_err}")

        # --- FIXED-RISK & EQUAL-WEIGHT SIZING LOGIC ---
        if stop_loss > 0 and abs(ref_price - stop_loss) > 0:
            # Fixed-Risk Sizing: (Balance * Risk%) / SL Distance
            risk_amount_usd = account_balance * (risk_pct / 100.0)
            sl_distance = abs(ref_price - stop_loss)
            target_contracts = risk_amount_usd / sl_distance
            
            # Verify required margin does not exceed Equal-Weight Allocated cap
            required_margin = (target_contracts * ref_price) / leverage
            if required_margin > amount_usd:
                logger.info(f"[{symbol}] Fixed-Risk margin (${required_margin:.2f}) capped by Equal-Weight cap (${amount_usd:.2f}).")
                required_margin = amount_usd
                target_contracts = (required_margin * leverage) / ref_price
            
            raw_qty = target_contracts
            trade_amount_usd = required_margin
        else:
            # Fallback to pure Equal-Weight Allocation sizing if SL isn't defined
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

        # Stop Loss Sanity Floor Checks
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

    def close_live_position_bybit(self, symbol: str, position_size: float, current_price: float, outcome: str = "CLOSE") -> dict:
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


# Class alias to maintain backward compatibility for Dashboard / Streamlit imports
LiveExecutionEngine = BybitFuturesLiveExecutor