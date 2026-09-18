import asyncio
import logging
from typing import Dict, Any, Callable, List, Optional

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
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable):
        """Registers a callback function for a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.info(f"Registered subscriber for event type: {event_type}")

    async def publish(self, event_type: str, payload: Dict[str, Any]):
        """Publishes an event to the queue and dispatches it to registered subscribers."""
        event = {"type": event_type, "payload": payload}
        await self.queue.put(event)

        # Dispatch to registered callbacks if present
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(payload))
                    else:
                        callback(payload)
                except Exception as e:
                    logger.error(
                        f"Error executing callback for {event_type}: {e}",
                        exc_info=True,
                    )

    async def consume(self) -> Dict[str, Any]:
        return await self.queue.get()

    def arm_local_sl_guard(
        self, symbol: str, direction: str, quantity: float, stop_loss: float
    ):
        clean_symbol = symbol.replace("/", "").replace("_", "").upper()
        self.active_local_sl_guards[clean_symbol] = {
            "direction": direction.upper(),
            "quantity": quantity,
            "stop_loss": stop_loss,
        }
        logger.warning(
            f"[{clean_symbol}] EMERGENCY LOCAL SL GUARD ARMED ({direction.upper()}) @ ${stop_loss:.5f} for {quantity} units."
        )

    def disarm_local_sl_guard(self, symbol: str):
        clean_symbol = symbol.replace("/", "").replace("_", "").upper()
        if clean_symbol in self.active_local_sl_guards:
            del self.active_local_sl_guards[clean_symbol]
            logger.info(f"[{clean_symbol}] Local SL Guard disarmed.")


# Global Event Bus Singleton
event_bus = CentralEventBus()