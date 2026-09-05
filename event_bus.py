import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("event_bus")

class PositionStateGuard:
    """In-Memory Atomic Lock Guard preventing duplicate trade execution for identical symbols."""
    def __init__(self):
        self._active_locks = set()
        self._mutex = asyncio.Lock()

    async def try_acquire_trade_lock(self, symbol: str) -> bool:
        async with self._mutex:
            clean_symbol = symbol.replace("/", "").replace("_", "").upper()
            if clean_symbol in self._active_locks:
                return False
            self._active_locks.add(clean_symbol)
            return True

    async def release_trade_lock(self, symbol: str):
        async with self._mutex:
            clean_symbol = symbol.replace("/", "").replace("_", "").upper()
            self._active_locks.discard(clean_symbol)


class CentralEventBus:
    """Single-writer Event Bus connecting Market Streams, Strategy Signals, and Execution Pipelines."""
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.guard: PositionStateGuard = PositionStateGuard()
        self.active_local_sl_guards: Dict[str, Dict[str, Any]] = {}

    async def publish(self, event_type: str, payload: Dict[str, Any]):
        await self.queue.put({"type": event_type, "payload": payload})

    async def consume(self) -> Dict[str, Any]:
        return await self.queue.get()

    def arm_local_sl_guard(self, symbol: str, direction: str, quantity: float, stop_loss: float):
        clean_symbol = symbol.replace("/", "").replace("_", "").upper()
        self.active_local_sl_guards[clean_symbol] = {
            "direction": direction.upper(),
            "quantity": quantity,
            "stop_loss": stop_loss
        }
        logger.warning(f"[{clean_symbol}] EMERGENCY LOCAL SL GUARD ARMED ({direction.upper()}) @ ${stop_loss:.5f} for {quantity} units.")

    def disarm_local_sl_guard(self, symbol: str):
        clean_symbol = symbol.replace("/", "").replace("_", "").upper()
        if clean_symbol in self.active_local_sl_guards:
            del self.active_local_sl_guards[clean_symbol]
            logger.info(f"[{clean_symbol}] Local SL Guard disarmed.")


# Global Event Bus Singleton
event_bus = CentralEventBus()