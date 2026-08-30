from dataclasses import dataclass


@dataclass
class Document:
    revision: int
    title: str


class TitleEditor:
    def __init__(self, document: Document) -> None:
        self.document = document
        self._undo_title: str | None = None

    def rename(self, title: str) -> None:
        self._undo_title = self.document.title
        self.document.title = title
        self.document.revision += 1

    def undo(self) -> None:
        if self._undo_title is not None:
            self.document.title = self._undo_title
            self.document.revision += 1
