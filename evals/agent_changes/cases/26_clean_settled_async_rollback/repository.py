import asyncio


class Repository:
    def __init__(self) -> None:
        self.records: set[str] = set()

    async def insert(
        self,
        key: str,
        *,
        delay: float = 0,
        fail: bool = False,
    ) -> None:
        await asyncio.sleep(delay)
        if fail:
            raise RuntimeError(f"could not insert {key}")
        self.records.add(key)

    async def delete(self, key: str) -> None:
        self.records.discard(key)
