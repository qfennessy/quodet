class Reschedule(Exception):
    pass


def process(sender) -> bool:
    try:
        sender()
        return True
    except Exception:
        return False
