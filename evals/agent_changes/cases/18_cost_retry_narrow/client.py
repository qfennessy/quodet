from typing import Protocol


class Provider(Protocol):
    def chargeable_request(self, request_id: str) -> str: ...


def request_with_retry(provider: Provider, request_id: str) -> str:
    try:
        return provider.chargeable_request(request_id)
    except TimeoutError:
        return provider.chargeable_request(request_id + "-retry")
