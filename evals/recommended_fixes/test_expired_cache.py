from expired_cache import Cache


def test_unexpired_entry_is_returned() -> None:
    now = 10.0
    cache = Cache(clock=lambda: now)
    cache.set("key", "value", ttl_seconds=1.0)

    assert cache.get("key") == "value"
