class ProfileStore:
    def __init__(self) -> None:
        self._profiles = {"u1": (1, "draft")}

    def read(self, user_id: str) -> tuple[int, str]:
        return self._profiles[user_id]

    def write(self, user_id: str, bio: str) -> None:
        revision, _ = self._profiles[user_id]
        self._profiles[user_id] = (revision + 1, bio)
