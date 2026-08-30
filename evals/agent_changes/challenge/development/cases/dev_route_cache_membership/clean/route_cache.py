class RouteCache:
    def __init__(self) -> None:
        self.values: dict[tuple[str, tuple[str, ...]], list[str]] = {}

    def key(self, driver_id: str, stops: list[str]) -> tuple[str, tuple[str, ...]]:
        return driver_id, tuple(stops)
