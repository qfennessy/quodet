from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(value) / len(values)
