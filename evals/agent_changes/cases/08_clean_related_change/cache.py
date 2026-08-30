import time


class CachedValue:
    def __init__(self, value: str, ttl_seconds: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl_seconds

    def is_valid(self) -> bool:
        return time.monotonic() < self.expires_at
