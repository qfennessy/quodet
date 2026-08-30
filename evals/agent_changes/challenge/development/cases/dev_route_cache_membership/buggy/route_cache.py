class RouteCache:
    def __init__(self) -> None:
        self.values: dict[tuple[str, int], list[str]] = {}

    def key(self, driver_id: str, stops: list[str]) -> tuple[str, int]:
        return driver_id, len(stops)
