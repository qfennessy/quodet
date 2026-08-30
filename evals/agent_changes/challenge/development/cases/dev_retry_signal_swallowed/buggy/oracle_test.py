from delivery_worker import Reschedule, process


def retry_later() -> None:
    raise Reschedule("back pressure")


try:
    process(retry_later)
except Reschedule:
    pass
else:
    raise AssertionError("queue control signal was swallowed")
