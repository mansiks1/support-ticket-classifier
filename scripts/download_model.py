"""Download the trusted release bundle, verify checksum before extracting."""

import hashlib
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

root = Path(__file__).resolve().parents[1]
expected = (root / "MODEL_BUNDLE.sha256").read_text().split()[0]
url = "https://github.com/mansiks1/support-ticket-classifier/releases/download/v1.0.0/support-ticket-models.zip"
with tempfile.TemporaryDirectory() as temporary:
    bundle = Path(temporary) / "models.zip"
    urllib.request.urlretrieve(url, bundle)
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != expected:
        raise ValueError("Bundle checksum mismatch; nothing extracted")
    with zipfile.ZipFile(bundle) as archive:
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != "models"
            ):
                raise ValueError("Unsafe archive member")
        archive.extractall(root)
print("Verified model bundle downloaded and extracted to models/")
