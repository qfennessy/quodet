import asyncio
from typing import Awaitable, Callable

from billing import SpendLedger


async def generate(
    request: Callable[[], Awaitable[tuple[str, int]]], ledger: SpendLedger
) -> str:
    try:
        text, billed_cents = await asyncio.wait_for(request(), timeout=5)
    except TimeoutError:
        return ""
    ledger.record(billed_cents)
    return text
