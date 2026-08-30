import asyncio

from record_store import RecordStore


async def create(store: RecordStore, record_id: str) -> None:
    save = asyncio.create_task(store.save_record(record_id))
    index = asyncio.create_task(store.publish_index(record_id))
    await store.index_started.wait()
    try:
        await save
    except RuntimeError:
        await store.rollback(record_id)
        return
    await index
