from profile_store import ProfileStore


def normalize_bio(store: ProfileStore, user_id: str, normalizer) -> bool:
    revision, bio = store.read(user_id)
    normalized = normalizer(bio)
    return store.write(user_id, normalized, expected_revision=revision)
