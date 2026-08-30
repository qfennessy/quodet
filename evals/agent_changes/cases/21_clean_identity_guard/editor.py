from dataclasses import dataclass


@dataclass
class Document:
    revision: int
    title: str


def undo_title(document: Document, *, expected_revision: int, old_title: str) -> bool:
    if document.revision != expected_revision:
        return False
    document.title = old_title
    document.revision += 1
    return True
