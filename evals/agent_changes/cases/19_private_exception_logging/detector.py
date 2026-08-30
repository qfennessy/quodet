import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Record:
    identity: str
    private_label: str

    def checked_identity(self) -> str:
        if not self.identity:
            raise ValueError(f"missing identity for {self.private_label}")
        return self.identity


def detect_duplicates(records: list[Record]) -> bool:
    try:
        identities = [record.checked_identity() for record in records]
        return len(identities) != len(set(identities))
    except Exception as error:
        logger.error("duplicate detection failed: %s", error)
        return False
