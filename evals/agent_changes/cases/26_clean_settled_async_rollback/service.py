import asyncio

from repository import Repository


async def create_pair(repository: Repository) -> None:
    left = asyncio.create_task(repository.insert("left", delay=0.05))
    right = asyncio.create_task(repository.insert("right", fail=True))

    try:
        await asyncio.gather(left, right)
    except Exception:
        left.cancel()
        right.cancel()
        await asyncio.gather(left, right, return_exceptions=True)
        await repository.delete("left")
        await repository.delete("right")
        raise
