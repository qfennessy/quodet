import asyncio


class Repository:
    def __init__(self) -> None:
        self.rows: list[str] = []

    async def insert(self, value: str) -> None:
        if value == "reject":
            raise ValueError("rejected value")
        await asyncio.sleep(0.01)
        self.rows.append(value)

    async def remove(self, value: str) -> None:
        self.rows.remove(value)
