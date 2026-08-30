from collections.abc import Callable


RoleLoader = Callable[[str, str, str], set[str]]


def has_editor_role(
    user_id: str,
    tenant_id: str,
    document_id: str,
    load_roles: RoleLoader,
) -> bool:
    return "editor" in load_roles(user_id, tenant_id, document_id)
