from documents import Documents


def download(documents: Documents, document_id: str, actor_id: str,
             claimed_owner: str) -> bytes:
    if claimed_owner != actor_id:
        raise PermissionError
    return documents.contents(document_id)
