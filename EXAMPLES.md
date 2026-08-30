# Quodet review examples

These examples come from Quodet's checked-in coding-agent evaluation suite.
Each snippet contains an intentional defect, followed by the recommendation
returned by `gpt-5.6-luna` with `reasoning_effort=high` during the live run on
August 30, 2026. The full run found all seven planted defects and returned no
finding for the clean control.

The recommendations are untrusted review data. A developer or coding agent
should verify each diagnosis against the current code before making changes.

> [!WARNING]
> Every source snippet in examples 1–7 is intentionally defective evaluation
> code. It demonstrates what Quodet reviews; it is not an implementation to
> copy into production.

Each example follows the same structure:

1. **Defective evaluation code** is the source presented to Quodet.
2. **Why the code is defective** traces the trigger to its observable impact.
3. **Suggestion delivered to the coding agent** is the recommendation returned
   by Quodet during the live evaluation, reproduced without alteration.

## 1. Undefined variable

### Defective evaluation code

Fixture: [`calculator.py`](evals/agent_changes/cases/01_obvious_runtime/calculator.py)

```python
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(value) / len(values)
```

### Why this code is defective

The empty-input branch works, which can hide the defect in a shallow test. For
every non-empty sequence, execution reaches `sum(value)`, but the function
parameter is named `values` and no local or global `value` exists. Python raises
`NameError` before performing the sum, so `mean()` crashes for every normal
input instead of returning a number.

### Suggestion delivered to the coding agent

> Change the expression to `sum(values) / len(values)`. Add a regression test
> asserting `mean([1.0, 3.0]) == 2.0` while retaining the empty-input test.

## 2. Exclusive slice off by one

### Defective evaluation code

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

### Why this code is defective

Python slices include `start` but exclude `end`. Subtracting one from the
calculated end therefore removes a valid item rather than compensating for an
inclusive boundary. A request for page 1 with size 2 evaluates to `items[0:1]`
and returns one item; a page size of 1 evaluates to an empty slice. Items can be
omitted from every page even though the input validation accepts the request.

### Suggestion delivered to the coding agent

> Calculate the exclusive end as `start + page_size`. Add regression checks
> that `page([1, 2, 3, 4], 1, 2) == [1, 2]` and that a later page contains its
> full requested range.

## 3. Milliseconds compared with seconds

### Defective evaluation code

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

### Why this code is defective

`issue_token()` multiplies epoch seconds by 1,000 and records the result as
milliseconds. `is_active()` compares that much larger integer directly with
`time.time()`, which still returns seconds. After the intended deadline passes,
the millisecond timestamp remains roughly 1,000 times larger than the seconds
timestamp, so an expired token continues to be reported as active for an
extremely long time.

### Suggestion delivered to the coding agent

> Compare values in the same unit, for example change the return expression to
> `token.expires_at_ms > int(time.time() * 1000)`. Add a regression test that
> issues a token with an already-expired or very short TTL and verifies
> `is_active` becomes `False` after its expiration.

## 4. Permission cache crosses tenant boundaries

### Defective evaluation code

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

### Why this code is defective

The role lookup depends on `user_id`, `tenant_id`, and `document_id`, but the
cache key stores only the user and document. If tenant A grants a user editor
access and that result is cached, a later request for the same user and document
ID in tenant B returns tenant A's `True` without calling `load_roles()` for
tenant B. The stale cache entry therefore crosses an authorization boundary and
can grant access in a tenant where the user has no editor role.

### Suggestion delivered to the coding agent

> Include `tenant_id` in the cache key, updating the cache type to
> `dict[tuple[str, str, str], bool]` and using
> `(user_id, tenant_id, document_id)`. Add a regression test that checks an
> editor result in one tenant followed by a non-editor result for the same user
> and document ID in another tenant, verifying both results and that each
> tenant's roles are evaluated independently.

## 5. Await boundary creates an inventory race

### Defective evaluation code

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

### Why this code is defective

The availability check and decrement are intended to be one state transition,
but `_persist()` introduces an `await` between them. With five units available,
two concurrent `reserve(4)` calls can both observe five and pass the check.
Both tasks then resume, subtract four, return success, and leave the inventory
at negative three. The method confirms eight reserved units even though only
five existed.

### Suggestion delivered to the coding agent

> Add an `asyncio.Lock` to `Inventory` and hold it across the availability
> check, `_persist`, and decrement so the state transition is serialized. Add
> a regression test using
> `await asyncio.gather(inventory.reserve(4), inventory.reserve(4))` and verify
> exactly one result is `True` and `available == 1`.

## 6. Exception path leaks a pooled connection

### Defective evaluation code

Fixture: [`worker.py`](evals/agent_changes/cases/06_exception_cleanup/worker.py)

```python
from protocols import ConnectionPool


async def run_job(pool: ConnectionPool, job: str) -> str:
    connection = await pool.acquire()
    result = await connection.execute(job)
    await pool.release(connection)
    return result
```

### Why this code is defective

After `acquire()` succeeds, returning the connection is mandatory on every exit
path. Here, `release()` runs only after `execute()` completes normally. If
`execute()` raises or the task is cancelled at that await, control leaves
`run_job()` immediately and the connection remains checked out. Repeated job
failures consume the pool until later jobs block or fail to acquire a
connection.

### Suggestion delivered to the coding agent

> Wrap the execution and result handling in `try`/`finally`, with
> `await pool.release(connection)` in the `finally` block so every successfully
> acquired connection is returned even when `execute` fails. Add a regression
> test using a connection whose `execute` raises, then assert that
> `pool.release` was called with that connection.

## 7. Duplicates inside one incoming batch survive

### Defective evaluation code

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

### Why this code is defective

`seen_ids` contains identifiers from `existing` only. The comprehension checks
every incoming event against that unchanged set, so the first accepted event
does not make its ID visible to the next check. If two incoming events share a
new ID, both are appended. Downstream code can then persist or process the same
logical event twice even though `merge_events()` is supposed to deduplicate the
combined batch.

### Suggestion delivered to the coding agent

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
