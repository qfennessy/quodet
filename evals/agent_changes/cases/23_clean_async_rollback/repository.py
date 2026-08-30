from typing import Protocol


class Repository(Protocol):
    async def insert_pair(self, first: str, second: str) -> None:
        """Atomically insert both values or leave the repository unchanged."""
        ...
