from route_cache import RouteCache
from route_service import planned_route


cache = RouteCache()
assert planned_route(cache, "d1", ["A", "B"]) == ["A", "B"]
assert planned_route(cache, "d1", ["A", "C"]) == ["A", "C"]
