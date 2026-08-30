import asyncio

from inventory import Inventory


def test_reserve_reduces_available_inventory() -> None:
    inventory = Inventory(5)
    assert asyncio.run(inventory.reserve(2)) is True
    assert inventory.available == 3
