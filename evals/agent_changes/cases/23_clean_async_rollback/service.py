from repository import Repository


async def import_pair(repository: Repository, first: str, second: str) -> None:
    await repository.insert_pair(first, second)
