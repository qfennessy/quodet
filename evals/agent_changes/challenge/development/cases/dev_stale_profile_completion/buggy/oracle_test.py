from profile_service import normalize_bio
from profile_store import ProfileStore


store = ProfileStore()


def interleaving_normalizer(bio: str) -> str:
    store.write("u1", "newer text")
    return bio.upper()


normalize_bio(store, "u1", interleaving_normalizer)
assert store.read("u1")[1] == "newer text"
