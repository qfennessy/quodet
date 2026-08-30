from dataclasses import dataclass


@dataclass(frozen=True)
class QueueCounts:
    ready: int
    blocked: int


class QueueSummaryCache:
    def __init__(self) -> None:
        self._fingerprint: tuple[int, int] | None = None
        self._summary = ""

    def render(self, counts: QueueCounts) -> str:
        fingerprint = (counts.ready, counts.blocked)
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self._summary = f"ready={counts.ready}; blocked={counts.blocked}"
        return self._summary
