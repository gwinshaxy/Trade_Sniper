import logging
import time
import ccxt

logger = logging.getLogger(__name__)

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

class MEXCLiveExecutor:
    _shared_exchange = None  # Class-level shared singleton session

    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = False):
        if MEXCLiveExecutor._shared_exchange is None:
            MEXCLiveExecutor._shared_exchange = ccxt.mexc({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'createMarketBuyOrderRequiresPrice': False,
                }
            })
            if testnet:
                MEXCLiveExecutor._shared_exchange.set_sandbox_mode(True)

        self.exchange = MEXCLiveExecutor._shared_exchange
        self.exchange.load_markets()

    def fetch_available_usdt_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance()
            usdt_free = safe_float(balance.get('USDT', {}).get('free', 0.0))
            logger.info(f"Live available USDT Balance retrieved: ${usdt_free:.2f}")
            return usdt_free
        except Exception as e:
            logger.error(f"Failed to fetch live USDT balance: {e}")
            return 0.0

    def get_available_usdt_balance(self) -> float:
        return self.fetch_available_usdt_balance()

    def get_spot_balance(self, symbol: str) -> float:
        base_asset = symbol.split('/')[0].upper()
        try:
            balance = self.exchange.fetch_balance()
            asset_free = safe_float(balance.get(base_asset, {}).get('free', 0.0))
            logger.info(f"[{base_asset}] Current live exchange balance: {asset_free}")
            return asset_free
        except Exception as e:
            logger.error(f"Failed to fetch spot balance for {base_asset}: {e}")
            return 0.0

    def fetch_ticker_data(self, symbol: str) -> dict:
        """Fetches live bid, ask, and spread details for strict liquidity checks."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
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
            logger.error(f"[{symbol}] Failed to fetch ticker data: {e}")
            return {"bid": 0.0, "ask": 0.0, "last": 0.0, "spread_pct": 0.0}

    def fetch_ticker_price(self, symbol: str) -> float:
        data = self.fetch_ticker_data(symbol)
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
        amount_usd: float = 25.0
    ) -> bool:
        clean_direction = direction.upper().strip()
        
        if clean_direction != "BUY" and clean_direction != "LONG":
            logger.warning(f"[{pair}] Attempted {clean_direction} execution. Spot execution only supports BUY/LONG positions.")
            return False

        res = self.buy_spot_mexc(symbol=pair, amount_usd=amount_usd)
        if res.get("status") == "SUCCESS":
            logger.info(f"[{pair}] Live spot order executed successfully via Dashboard wrapper: Order ID {res.get('order_id')}")
            return True
        else:
            logger.error(f"[{pair}] Live spot order execution failed: {res.get('error') or res.get('reason')}")
            return False

    def buy_spot_mexc(self, symbol: str, amount_usd: float) -> dict:
        clean_symbol = symbol.replace("/", "").upper()
        if "/" not in symbol and clean_symbol.endswith("USDT"):
            symbol = f"{clean_symbol[:-4]}/USDT"

        base_asset = symbol.split('/')[0]

        existing_asset_balance = self.get_spot_balance(symbol)
        if existing_asset_balance > MIN_DUST_THRESHOLD:
            logger.info(f"[{symbol}] Guard triggered: Existing {base_asset} balance found on MEXC ({existing_asset_balance:.4f}). Skipping order execution.")
            return {"status": "SKIPPED", "reason": f"Existing spot balance found ({existing_asset_balance:.4f})"}

        available_usdt = self.fetch_available_usdt_balance()
        if available_usdt <= 1.0:
            return {"status": "FAILED", "error": f"Insufficient live USDT balance: ${available_usdt:.2f}"}

        # Step 1: Real-time Liquidity & Spread Protection Check
        ticker_data = self.fetch_ticker_data(symbol)
        ask_price = ticker_data["ask"]
        spread_pct = ticker_data["spread_pct"]

        if ask_price <= 0:
            return {"status": "FAILED", "error": "Invalid ask price prior to execution"}

        if spread_pct > MAX_ALLOWED_SPREAD_PCT:
            logger.warning(
                f"[{symbol}] Order Rejected: High Spread detected ({spread_pct * 100:.3f}% > Max allowed {MAX_ALLOWED_SPREAD_PCT * 100:.2f}%)."
            )
            return {"status": "FAILED", "error": f"Spread too high ({spread_pct * 100:.3f}%)"}

        # Step 2: Calculate Explicit Limit Ceiling (0.2% Max Slippage)
        trade_amount_usd = min(amount_usd, available_usdt * 0.98)
        limit_price = ask_price * (1.0 + MAX_SLIPPAGE_PCT)

        raw_qty = trade_amount_usd / ask_price
        
        # Format Qty and Price via CCXT Precision Rules
        try:
            formatted_qty = safe_float(self.exchange.amount_to_precision(symbol, raw_qty))
            formatted_limit_price = safe_float(self.exchange.price_to_precision(symbol, limit_price))
        except Exception as prec_err:
            logger.warning(f"[{symbol}] CCXT Precision Error: {prec_err}. Using standard rounding.")
            formatted_qty = round(raw_qty, 4)
            formatted_limit_price = round(limit_price, 6)

        logger.info(
            f"[{symbol}] Initiating Limit Buy (IOC): ${trade_amount_usd:.2f} "
            f"({formatted_qty} units @ Limit Ceiling: ${formatted_limit_price:.6f} | Ask: ${ask_price:.6f})"
        )

        # Step 3: Execute Limit Order with Immediate-Or-Cancel (IOC)
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side='buy',
                amount=formatted_qty,
                price=formatted_limit_price,
                params={'timeInForce': 'IOC'}
            )

            order_id = order.get("id")
            fill_price = safe_float(order.get("average") or order.get("price"))
            executed_qty = safe_float(order.get("filled") or order.get("amount"))

            if (fill_price <= 0 or executed_qty <= 0) and order_id:
                time.sleep(0.4)
                try:
                    fetched_order = self.exchange.fetch_order(order_id, symbol)
                    fill_price = safe_float(fetched_order.get("average") or fetched_order.get("price"))
                    executed_qty = safe_float(fetched_order.get("filled") or fetched_order.get("amount"))
                except Exception as fetch_err:
                    logger.warning(f"[{symbol}] Failed post-execution order lookup: {fetch_err}")

            if executed_qty <= 0:
                logger.warning(f"[{symbol}] Limit IOC Order failed to fill within slippage ceiling (${formatted_limit_price:.6f}).")
                return {"status": "FAILED", "error": "Limit IOC order unfilled/cancelled due to slippage guard."}

            if fill_price <= 0:
                fill_price = ask_price

            logger.info(f"[{symbol}] Limit Buy Executed successfully. Fill Price: {fill_price:.6f}, Qty: {executed_qty}")

            return {
                "status": "SUCCESS",
                "order_id": order_id,
                "fill_price": fill_price,
                "executed_qty": executed_qty,
                "cost_usd": fill_price * executed_qty,
                "raw_order": order
            }

        except Exception as e:
            logger.error(f"[{symbol}] MEXC Limit Buy Failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    def close_live_position_mexc(self, symbol: str, position_size: float, current_price: float, outcome: str = "CLOSE") -> dict:
        clean_symbol = symbol.replace("/", "").upper()
        if "/" not in symbol and clean_symbol.endswith("USDT"):
            symbol = f"{clean_symbol[:-4]}/USDT"

        base_asset = symbol.split('/')[0]

        try:
            available_amount = self.get_spot_balance(symbol)
        except Exception as bal_err:
            logger.warning(f"[{symbol}] Could not fetch balance prior to sell: {bal_err}. Proceeding with required amount.")
            available_amount = position_size

        logger.info(f"[{symbol}] Closing Spot Position ({outcome}): Required={position_size}, Available on MEXC={available_amount}")

        if available_amount < MIN_DUST_THRESHOLD:
            logger.warning(
                f"[{symbol}] Insufficient {base_asset} balance on exchange (Have: {available_amount}, Need: {position_size}). "
                "Bypassing MEXC API order and force-closing trade in DB."
            )
            return {
                "status": "FORCE_CLOSED_DB_ONLY",
                "order_id": "GHOST_POSITION_DB_CLOSE",
                "exit_price": current_price,
                "pnl_usd": 0.0,
                "pnl_pct": 0.0,
                "note": f"Tokens unavailable on MEXC ({available_amount} free). Direct DB close applied."
            }

        # Dynamic Slicing & Precision Clipping
        sell_amount_raw = min(position_size, available_amount)
        try:
            sell_amount = safe_float(self.exchange.amount_to_precision(symbol, sell_amount_raw))
        except Exception:
            sell_amount = sell_amount_raw

        try:
            order = self.exchange.create_market_sell_order(
                symbol=symbol,
                amount=sell_amount
            )
            order_id = order.get("id")
            exit_price = safe_float(order.get("average") or order.get("price")) or current_price

            return {
                "status": "SUCCESS",
                "order_id": order_id,
                "exit_price": exit_price,
                "pnl_usd": 0.0,
                "pnl_pct": 0.0,
                "raw_order": order
            }
        except Exception as e:
            logger.error(f"[{symbol}] Failed to execute Spot Market Sell: {e}")
            return {"status": "FAILED", "error": str(e)}

    def calculate_dynamic_stop_loss(
        self, 
        entry_price: float, 
        signal_type: str, 
        atr_val: float = 0.0, 
        risk_reward_ratio: float = 2.0,
        min_sl_pct: float = 0.008
    ) -> tuple[float, float]:
        if entry_price <= 0:
            raise ValueError("Entry price must be greater than 0")

        if atr_val <= 0:
            sl_distance = entry_price * max(0.015, min_sl_pct)
        else:
            sl_distance = max(atr_val * 1.5, entry_price * min_sl_pct)

        tp_distance = sl_distance * risk_reward_ratio

        if signal_type.upper() in ["BUY", "LONG"]:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return round(stop_loss, 6), round(take_profit, 6)

LiveExecutionEngine = MEXCLiveExecutor