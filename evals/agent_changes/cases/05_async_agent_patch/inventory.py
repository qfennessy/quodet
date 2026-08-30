import asyncio


class Inventory:
    def __init__(self, available: int) -> None:
        self.available = available

    async def _persist(self, quantity: int) -> None:
        await asyncio.sleep(0.01)

    async def reserve(self, quantity: int) -> bool:
        if self.available < quantity:
            return False
        await self._persist(quantity)
        self.available -= quantity
        return True
