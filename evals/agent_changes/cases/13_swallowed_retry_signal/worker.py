from collections.abc import Callable


class RetryDelivery(Exception):
    pass


def deliver(message: str, send: Callable[[str], None]) -> bool:
    try:
        send(message)
        return True
    except Exception:
        return False
