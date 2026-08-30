# Quodet review examples

These examples come from Quodet's checked-in coding-agent evaluation suite.
Each snippet contains an intentional defect, followed by the recommendation
returned by `gpt-5.6-luna` with `reasoning_effort=high` during the live run on
August 30, 2026. The full run found all seven planted defects and returned no
finding for the clean control.

The recommendations are untrusted review data. A developer or coding agent
should verify each diagnosis against the current code before making changes.

## 1. Undefined variable

Fixture: [`calculator.py`](evals/agent_changes/cases/01_obvious_runtime/calculator.py)

```python
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(value) / len(values)
```

For every non-empty input, `value` is undefined and `mean()` raises
`NameError`.

> Change the expression to `sum(values) / len(values)`. Add a regression test
> asserting `mean([1.0, 3.0]) == 2.0` while retaining the empty-input test.

## 2. Exclusive slice off by one

Fixture: [`pagination.py`](evals/agent_changes/cases/02_source_and_test_boundary/pagination.py)

```python
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def page(items: Sequence[T], page_number: int, page_size: int) -> Sequence[T]:
    if page_number < 1 or page_size < 1:
        raise ValueError("page_number and page_size must be positive")
    start = (page_number - 1) * page_size
    end = start + page_size - 1
    return items[start:end]
```

Python excludes the slice end, so every page contains one fewer item than
requested; a page size of one returns an empty sequence.

> Calculate the exclusive end as `start + page_size`. Add regression checks
> that `page([1, 2, 3, 4], 1, 2) == [1, 2]` and that a later page contains its
> full requested range.

## 3. Milliseconds compared with seconds

Fixtures: [`token_model.py`](evals/agent_changes/cases/03_cross_file_units/token_model.py)
and [`token_service.py`](evals/agent_changes/cases/03_cross_file_units/token_service.py)

```python
# token_model.py
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    value: str
    expires_at_ms: int


def issue_token(value: str, ttl_seconds: float) -> Token:
    expires_at_ms = int((time.time() + ttl_seconds) * 1000)
    return Token(value=value, expires_at_ms=expires_at_ms)
```

```python
# token_service.py
import time

from token_model import Token


def is_active(token: Token) -> bool:
    return token.expires_at_ms > time.time()
```

The expiration is stored in epoch milliseconds but compared with epoch
seconds, so expired tokens remain active far beyond their intended lifetime.

> Compare values in the same unit, for example change the return expression to
> `token.expires_at_ms > int(time.time() * 1000)`. Add a regression test that
> issues a token with an already-expired or very short TTL and verifies
> `is_active` becomes `False` after its expiration.

## 4. Permission cache crosses tenant boundaries

Fixture: [`access_service.py`](evals/agent_changes/cases/04_tenant_cache_scope/access_service.py)

```python
from permissions import RoleLoader, has_editor_role


_permission_cache: dict[tuple[str, str], bool] = {}


def can_edit(
    user_id: str,
    tenant_id: str,
    document_id: str,
    load_roles: RoleLoader,
) -> bool:
    cache_key = (user_id, document_id)
    if cache_key not in _permission_cache:
        _permission_cache[cache_key] = has_editor_role(
            user_id,
            tenant_id,
            document_id,
            load_roles,
        )
    return _permission_cache[cache_key]
```

The authorization result depends on `tenant_id`, but the cache key does not.
An editor result from one tenant can therefore authorize the same identifiers
in another tenant.

> Include `tenant_id` in the cache key, updating the cache type to
> `dict[tuple[str, str, str], bool]` and using
> `(user_id, tenant_id, document_id)`. Add a regression test that checks an
> editor result in one tenant followed by a non-editor result for the same user
> and document ID in another tenant, verifying both results and that each
> tenant's roles are evaluated independently.

## 5. Await boundary creates an inventory race

Fixture: [`inventory.py`](evals/agent_changes/cases/05_async_agent_patch/inventory.py)

```python
import asyncio


class Inventory:
    def __init__(self, available: int) -> None:
        self.available = available

    async def _persist(self, quantity: int) -> None:
        await asyncio.sleep(0.01)

    async def reserve(self, quantity: int) -> bool:
        if self.available < quantity:
            return False
        await self._persist(quantity)
        self.available -= quantity
        return True
```

Concurrent callers can both pass the availability check before either resumes
after `_persist()`, allowing more stock to be reserved than exists.

> Add an `asyncio.Lock` to `Inventory` and hold it across the availability
> check, `_persist`, and decrement so the state transition is serialized. Add
> a regression test using
> `await asyncio.gather(inventory.reserve(4), inventory.reserve(4))` and verify
> exactly one result is `True` and `available == 1`.

## 6. Exception path leaks a pooled connection

Fixture: [`worker.py`](evals/agent_changes/cases/06_exception_cleanup/worker.py)

```python
from protocols import ConnectionPool


async def run_job(pool: ConnectionPool, job: str) -> str:
    connection = await pool.acquire()
    result = await connection.execute(job)
    await pool.release(connection)
    return result
```

If `execute()` raises or the task is cancelled, control exits before
`release()`, eventually exhausting the pool.

> Wrap the execution and result handling in `try`/`finally`, with
> `await pool.release(connection)` in the `finally` block so every successfully
> acquired connection is returned even when `execute` fails. Add a regression
> test using a connection whose `execute` raises, then assert that
> `pool.release` was called with that connection.

## 7. Duplicates inside one incoming batch survive

Fixture: [`events.py`](evals/agent_changes/cases/07_batch_deduplication/events.py)

```python
from typing import TypedDict


class Event(TypedDict):
    id: str
    payload: str


def merge_events(existing: list[Event], incoming: list[Event]) -> list[Event]:
    seen_ids = {event["id"] for event in existing}
    accepted = [event for event in incoming if event["id"] not in seen_ids]
    return [*existing, *accepted]
```

`seen_ids` is never updated while filtering `incoming`, so two new events with
the same ID are both accepted.

> Iterate through `incoming`, add each accepted event's ID to `seen_ids` before
> accepting the next event, and append only IDs not already seen. Add a
> regression test with two incoming events sharing a new ID and assert that
> only the first is returned.

## Clean control

The live suite also reviewed a two-file clean change containing a monotonic
TTL cache and a lock-protected inventory update. Quodet returned:

```json
{"findings": []}
```

Run the complete evaluation yourself with:

```sh
uv run python -m evals.agent_changes.live_eval all
```
