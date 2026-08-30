import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    value: str
    expires_at_ms: int


def issue_token(value: str, ttl_seconds: float) -> Token:
    expires_at_ms = int((time.time() + ttl_seconds) * 1000)
    return Token(value=value, expires_at_ms=expires_at_ms)
