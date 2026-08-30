from documents import Documents
from download_service import download


try:
    download(Documents(), "doc-secret", actor_id="mallory", claimed_owner="mallory")
except PermissionError:
    pass
else:
    raise AssertionError("spoofed owner claim authorized the download")
