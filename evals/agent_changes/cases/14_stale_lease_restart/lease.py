from dataclasses import dataclass


@dataclass
class Lease:
    owner_pid: int
    started_at: float


class LeaseStore:
    def __init__(self) -> None:
        self.current: Lease | None = None

    def acquire(self, owner_pid: int, now: float) -> bool:
        if self.current is not None:
            return False
        self.current = Lease(owner_pid=owner_pid, started_at=now)
        return True
