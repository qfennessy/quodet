import asyncio


class RecordStore:
    def __init__(self) -> None:
        self.records: set[str] = set()
        self.index: set[str] = set()
        self.index_started = asyncio.Event()
        self.release_index = asyncio.Event()

    async def save_record(self, record_id: str) -> None:
        raise RuntimeError("database unavailable")

    async def publish_index(self, record_id: str) -> None:
        self.index_started.set()
        await self.release_index.wait()
        self.index.add(record_id)

    async def rollback(self, record_id: str) -> None:
        self.records.discard(record_id)
        self.index.discard(record_id)
