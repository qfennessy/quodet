from pagination import page


def test_first_page_starts_with_first_item() -> None:
    assert page([1, 2, 3, 4], page_number=1, page_size=2)[0] == 1
