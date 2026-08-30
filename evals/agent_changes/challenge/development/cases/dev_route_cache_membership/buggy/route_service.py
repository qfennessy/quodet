from route_cache import RouteCache


def planned_route(cache: RouteCache, driver_id: str, stops: list[str]) -> list[str]:
    key = cache.key(driver_id, stops)
    if key not in cache.values:
        cache.values[key] = sorted(stops)
    return cache.values[key]
