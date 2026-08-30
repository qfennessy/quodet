import asyncio

from index_service import create
from record_store import RecordStore


async def check() -> None:
    store = RecordStore()
    await create(store, "r1")
    store.release_index.set()
    await asyncio.sleep(0)
    assert store.index == set()


asyncio.run(check())
