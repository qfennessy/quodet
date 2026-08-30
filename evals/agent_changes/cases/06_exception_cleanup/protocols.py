from typing import Protocol


class Connection(Protocol):
    async def execute(self, job: str) -> str: ...


class ConnectionPool(Protocol):
    async def acquire(self) -> Connection: ...

    async def release(self, connection: Connection) -> None: ...
