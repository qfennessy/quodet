from dataclasses import dataclass


@dataclass(frozen=True)
class QueueCounts:
    ready: int
    blocked: int

    @property
    def total(self) -> int:
        return self.ready + self.blocked


class QueueSummaryCache:
    def __init__(self) -> None:
        self._total: int | None = None
        self._summary: str | None = None

    def render(self, counts: QueueCounts) -> str:
        if self._total != counts.total:
            self._total = counts.total
            self._summary = f"ready={counts.ready}; blocked={counts.blocked}"
        assert self._summary is not None
        return self._summary
