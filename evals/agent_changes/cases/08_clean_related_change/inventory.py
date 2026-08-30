import asyncio


class Inventory:
    def __init__(self, available: int) -> None:
        self.available = available
        self._lock = asyncio.Lock()

    async def reserve(self, quantity: int) -> bool:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        async with self._lock:
            if self.available < quantity:
                return False
            self.available -= quantity
            return True
