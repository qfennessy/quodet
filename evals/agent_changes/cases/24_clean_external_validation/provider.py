from typing import Protocol


class Geocoder(Protocol):
    def latitude(self, address: str) -> float: ...


def locate(geocoder: Geocoder, address: str) -> float:
    latitude = float(geocoder.latitude(address))
    if latitude < -90 or latitude > 90:
        raise ValueError("invalid latitude")
    return latitude
