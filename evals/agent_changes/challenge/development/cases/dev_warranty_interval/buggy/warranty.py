from datetime import date


def parse_warranty(payload: dict[str, str]) -> tuple[date, date]:
    return date.fromisoformat(payload["starts_on"]), date.fromisoformat(payload["ends_on"])
