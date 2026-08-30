import asyncio

from repository import Repository


async def import_pair(repository: Repository, first: str, second: str) -> None:
    tasks = {
        asyncio.create_task(repository.insert(first)): first,
        asyncio.create_task(repository.insert(second)): second,
    }
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    failure = next((task.exception() for task in done if task.exception()), None)
    if failure is not None:
        for task in done:
            if not task.cancelled() and task.exception() is None:
                await repository.remove(tasks[task])
        raise failure
    await asyncio.gather(*pending)
