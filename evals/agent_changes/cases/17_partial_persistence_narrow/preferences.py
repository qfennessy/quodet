class PreferenceStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def save_pair(self, theme: str, locale: str) -> None:
        self.values["theme"] = theme
        self._validate_locale(locale)
        self.values["locale"] = locale

    @staticmethod
    def _validate_locale(locale: str) -> None:
        if "-" not in locale:
            raise ValueError("locale must include a region")
