from events import merge_events


def test_existing_events_are_not_duplicated() -> None:
    existing = [{"id": "one", "payload": "old"}]
    incoming = [{"id": "one", "payload": "new"}]
    assert merge_events(existing, incoming) == existing
