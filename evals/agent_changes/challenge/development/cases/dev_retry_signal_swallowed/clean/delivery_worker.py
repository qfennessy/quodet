class Reschedule(Exception):
    pass


def process(sender) -> bool:
    try:
        sender()
        return True
    except Reschedule:
        raise
    except Exception:
        return False
