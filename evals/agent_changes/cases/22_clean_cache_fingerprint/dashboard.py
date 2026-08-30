from cache import QueueCounts, QueueSummaryCache


def refresh(cache: QueueSummaryCache, ready: int, blocked: int) -> str:
    return cache.render(QueueCounts(ready=ready, blocked=blocked))
