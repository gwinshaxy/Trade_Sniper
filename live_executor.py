import logging
import time
import ccxt

logger = logging.getLogger(__name__)

def safe_float(val, default=0.0):
    """Safely cast value to float with fallback."""
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


class MEXCLiveExecutor:
    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = False):
        self.exchange = ccxt.mexc({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'createMarketBuyOrderRequiresPrice': False,
            }
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)

    def fetch_ticker_price(self, symbol: str) -> float:
        """Fetch real-time bid/ask average ticker price."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            bid = safe_float(ticker.get('bid'))
            ask = safe_float(ticker.get('ask'))
            last = safe_float(ticker.get('last'))
            
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            return last
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch ticker price: {e}")
            return 0.0

    def buy_spot_mexc(self, symbol: str, amount_usd: float) -> dict:
        """
        Executes a Spot Market Buy order on MEXC with post-execution verification 
        to capture the true average fill price and executed quantity.
        """
        clean_symbol = symbol.replace("/", "").upper()
        if "/" not in symbol and clean_symbol.endswith("USDT"):
            symbol = f"{clean_symbol[:-4]}/USDT"

        current_ticker_price = self.fetch_ticker_price(symbol)
        if current_ticker_price <= 0:
            return {"status": "FAILED", "error": "Invalid ticker price before execution"}

        estimated_qty = amount_usd / current_ticker_price
        logger.info(f"[{symbol}] Initiating Market Buy: ${amount_usd:.2f} (~{estimated_qty:.4f} units @ ~{current_ticker_price})")

        try:
            order = self.exchange.create_market_buy_order(
                symbol=symbol,
                amount=estimated_qty,
                params={'cost': amount_usd}
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

            if fill_price <= 0:
                logger.warning(f"[{symbol}] Order response missing fill price. Falling back to ticker.")
                fill_price = self.fetch_ticker_price(symbol) or current_ticker_price

            if executed_qty <= 0:
                executed_qty = amount_usd / fill_price

            logger.info(f"[{symbol}] Market Buy Executed successfully. Fill Price: {fill_price:.6f}, Qty: {executed_qty:.4f}")

            return {
                "status": "SUCCESS",
                "order_id": order_id,
                "fill_price": fill_price,
                "executed_qty": executed_qty,
                "cost_usd": fill_price * executed_qty,
                "raw_order": order
            }

        except Exception as e:
            logger.error(f"[{symbol}] MEXC Market Buy Failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    def close_live_position_mexc(self, symbol: str, position_size: float, current_price: float, outcome: str = "CLOSE") -> dict:
        """Executes a Spot Market Sell order to close an active long position."""
        clean_symbol = symbol.replace("/", "").upper()
        if "/" not in symbol and clean_symbol.endswith("USDT"):
            symbol = f"{clean_symbol[:-4]}/USDT"

        logger.info(f"[{symbol}] Closing Spot Position ({outcome}): Selling {position_size:.4f} units...")

        try:
            order = self.exchange.create_market_sell_order(
                symbol=symbol,
                amount=position_size
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

        if signal_type.upper() == "BUY":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return round(stop_loss, 6), round(take_profit, 6)

# Backwards-compatibility Alias
LiveExecutionEngine = MEXCLiveExecutor