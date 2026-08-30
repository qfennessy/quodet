from profile_store import ProfileStore


def normalize_bio(store: ProfileStore, user_id: str, normalizer) -> None:
    _revision, bio = store.read(user_id)
    normalized = normalizer(bio)
    store.write(user_id, normalized)
