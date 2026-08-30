class Documents:
    def __init__(self) -> None:
        self.owners = {"doc-secret": "alice"}

    def owner_of(self, document_id: str) -> str:
        return self.owners[document_id]

    def contents(self, document_id: str) -> bytes:
        return b"private"
