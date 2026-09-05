import asyncio
import json
import logging
import ssl
import websockets
from typing import List, Dict
from event_bus import event_bus
from common import get_db_connection, release_db_connection, finalize_trade_in_db

logger = logging.getLogger("ws_engine")

class UnifiedWebSocketEngine:
    def __init__(self, symbols: List[str], is_testnet: bool = True, api_key: str = None, api_secret: str = None):
        # Clean symbol to XRPUSDT format for Bybit V5 WS
        self.symbols = [
            s.split(":")[0].replace("/", "").replace("_", "").upper() 
            for s in symbols
        ]
        
        # Local price cache for ticker deltas
        self.last_prices: Dict[str, float] = {}
        self.is_testnet = is_testnet
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Select correct endpoints based on network mode
        if is_testnet:
            self.ws_endpoints = [
                "wss://stream-testnet.bybit.com/v5/public/linear",
                "wss://stream-testnet.bybitglobal.com/v5/public/linear"
            ]
            self.private_ws_endpoint = "wss://stream-testnet.bybit.com/v5/private"
        else:
            self.ws_endpoints = [
                "wss://stream.bybit.com/v5/public/linear",
                "wss://stream-m7.bybit.com/v5/public/linear"
            ]
            self.private_ws_endpoint = "wss://stream.bybit.com/v5/private"
            
        self.current_ep_idx = 0

    def _get_active_endpoint(self) -> str:
        return self.ws_endpoints[self.current_ep_idx]

    def _switch_endpoint(self):
        self.current_ep_idx = (self.current_ep_idx + 1) % len(self.ws_endpoints)

    async def _ping_loop(self, ws):
        try:
            while True:
                await asyncio.sleep(20)
                if ws.open:
                    await ws.send(json.dumps({"op": "ping"}))
        except (asyncio.CancelledError, Exception):
            pass

    async def _authenticate_private_ws(self, ws) -> bool:
        if not self.api_key or not self.api_secret:
            logger.warning("Private WebSocket credentials not provided. Skipping private stream authentication.")
            return False
        
        import time
        import hmac
        import hashlib
        
        expires = int((time.time() + 10) * 1000)
        signature_payload = f"GET/realtime{expires}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        auth_payload = {
            "op": "auth",
            "args": [self.api_key, expires, signature]
        }
        await ws.send(json.dumps(auth_payload))
        response = await ws.recv()
        data = json.loads(response)
        if data.get("success"):
            logger.info("Bybit Private WebSocket Authenticated Successfully.")
            return True
        else:
            logger.error(f"Bybit Private WebSocket Authentication Failed: {data}")
            return False

    async def _handle_private_execution(self, execution_data: dict):
        try:
            exec_type = execution_data.get("execType")
            order_status = execution_data.get("orderStatus")
            
            if order_status not in ["Filled", "Deactivated", "Cancelled"] and exec_type not in ["Trade"]:
                return

            symbol = execution_data.get("symbol")
            exec_price = float(execution_data.get("execPrice", 0) or 0)
            closed_pnl = float(execution_data.get("closedPnl", 0) or 0)
            
            if order_status == "Filled" or exec_type == "Trade":
                conn = get_db_connection()
                if not conn:
                    return
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT id, direction, entry_price, position_size, account_balance 
                            FROM trade_setups 
                            WHERE UPPER(REPLACE(REPLACE(pair, '/', ''), '_', '')) = %s 
                              AND trade_state IN ('OPEN', 'EXECUTED', 'BE_LOCKED', 'TRAILING')
                            ORDER BY id DESC LIMIT 1;
                        """, (symbol.replace("/", "").upper(),))
                        row = cur.fetchone()
                        if row:
                            trade_id, direction, entry_price, position_size, account_balance = row
                            account_balance = float(account_balance or 100.0)
                            
                            pnl_usd = closed_pnl
                            pnl_pct = (pnl_usd / account_balance) * 100.0 if account_balance > 0 else 0.0
                            outcome = "WIN" if pnl_usd > 0 else ("LOSS" if pnl_usd < 0 else "BREAKEVEN")
                            
                            finalize_trade_in_db(
                                trade_id=trade_id,
                                exit_price=exec_price,
                                pnl_usd=pnl_usd,
                                pnl_pct=pnl_pct,
                                outcome=outcome
                            )
                            logger.info(f"[Private WS Execution] Trade #{trade_id} ({symbol}) finalized via execution stream. PnL: ${pnl_usd:.2f}")
                except Exception as db_err:
                    logger.error(f"Error handling private WS execution data for DB sync: {db_err}")
                finally:
                    release_db_connection(conn)
        except Exception as e:
            logger.error(f"Error processing execution websocket message: {e}")

    async def start_private_listener(self):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        while True:
            try:
                async with websockets.connect(
                    self.private_ws_endpoint,
                    open_timeout=20,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                    ssl=ssl_context
                ) as ws:
                    logger.info(f"Connected to Bybit Private Feed: {self.private_ws_endpoint}")
                    if not await self._authenticate_private_ws(ws):
                        await asyncio.sleep(10)
                        continue

                    asyncio.create_task(self._ping_loop(ws))
                    subscribe_payload = {"op": "subscribe", "args": ["execution"]}
                    await ws.send(json.dumps(subscribe_payload))

                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("topic") == "execution":
                            for exec_item in data.get("data", []):
                                await self._handle_private_execution(exec_item)
            except Exception as e:
                logger.warning(f"Private WS connection dropped: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def start(self):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Launch Private Listener concurrently if credentials are present
        if self.api_key and self.api_secret:
            asyncio.create_task(self.start_private_listener())

        while True:
            endpoint = self._get_active_endpoint()
            logger.info(f"Connecting to Bybit WebSocket: {endpoint}...")

            try:
                async with websockets.connect(
                    endpoint,
                    open_timeout=20,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                    ssl=ssl_context
                ) as ws:
                    logger.info(f"Connected to Bybit Feed: {endpoint}")

                    asyncio.create_task(self._ping_loop(ws))

                    sub_args = [f"tickers.{symbol}" for symbol in self.symbols]
                    subscribe_payload = {"op": "subscribe", "args": sub_args}
                    await ws.send(json.dumps(subscribe_payload))

                    async for msg in ws:
                        data = json.loads(msg)

                        if data.get("ret_msg") == "pong" or data.get("op") == "pong":
                            continue

                        if "topic" in data and data["topic"].startswith("tickers."):
                            topic_parts = data["topic"].split(".")
                            symbol = topic_parts[1]
                            ticker_data = data.get("data", {})
                            last_price = float(ticker_data.get("lastPrice", 0.0) or 0.0)

                            if last_price > 0:
                                self.last_prices[symbol] = last_price
                                await event_bus.publish("TICKER_UPDATE", {
                                    "symbol": symbol,
                                    "price": last_price
                                })
            except Exception as e:
                logger.warning(f"WebSocket connection error ({endpoint}): {e}. Switching endpoints...")
                self._switch_endpoint()
                await asyncio.sleep(3)