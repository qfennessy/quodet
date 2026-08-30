from .expired_cache import Cache, CacheEntry


def test_unexpired_entry_is_returned() -> None:
    now = 10.0
    cache = Cache(clock=lambda: now)
    cache._entries["key"] = CacheEntry(value="value", expires_at=now + 1.0)

    assert cache.get("key") == "value"
