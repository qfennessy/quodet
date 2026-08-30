from documents import Documents


def download(documents: Documents, document_id: str, actor_id: str,
             claimed_owner: str) -> bytes:
    del claimed_owner
    if documents.owner_of(document_id) != actor_id:
        raise PermissionError
    return documents.contents(document_id)
