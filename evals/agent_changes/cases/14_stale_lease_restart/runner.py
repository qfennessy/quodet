from lease import LeaseStore


def start_worker(store: LeaseStore, process_id: int, now: float) -> bool:
    return store.acquire(owner_pid=process_id, now=now)
