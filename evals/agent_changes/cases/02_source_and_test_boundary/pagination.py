from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def page(items: Sequence[T], page_number: int, page_size: int) -> Sequence[T]:
    if page_number < 1 or page_size < 1:
        raise ValueError("page_number and page_size must be positive")
    start = (page_number - 1) * page_size
    end = start + page_size - 1
    return items[start:end]
