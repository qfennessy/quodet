from permissions import RoleLoader, has_editor_role


_permission_cache: dict[tuple[str, str], bool] = {}


def can_edit(
    user_id: str,
    tenant_id: str,
    document_id: str,
    load_roles: RoleLoader,
) -> bool:
    cache_key = (user_id, document_id)
    if cache_key not in _permission_cache:
        _permission_cache[cache_key] = has_editor_role(
            user_id,
            tenant_id,
            document_id,
            load_roles,
        )
    return _permission_cache[cache_key]
