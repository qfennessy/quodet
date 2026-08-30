from dataclasses import dataclass


@dataclass
class Spend:
    known_cents: int = 0
    requests: int = 0


class SpendLedger:
    def __init__(self) -> None:
        self.value = Spend()

    def record(self, cents: int) -> None:
        self.value.known_cents += cents
        self.value.requests += 1
