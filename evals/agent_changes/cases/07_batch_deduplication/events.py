from typing import TypedDict


class Event(TypedDict):
    id: str
    payload: str


def merge_events(existing: list[Event], incoming: list[Event]) -> list[Event]:
    seen_ids = {event["id"] for event in existing}
    accepted = [event for event in incoming if event["id"] not in seen_ids]
    return [*existing, *accepted]
