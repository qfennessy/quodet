from datetime import date


def parse_warranty(payload: dict[str, str]) -> tuple[date, date]:
    starts_on = date.fromisoformat(payload["starts_on"])
    ends_on = date.fromisoformat(payload["ends_on"])
    if ends_on < starts_on:
        raise ValueError("warranty end precedes start")
    return starts_on, ends_on
