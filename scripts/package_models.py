"""Package frozen release assets only; never include raw data or caches."""

import hashlib
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = root.parent / "support-ticket-models.zip"
paths = [
    root / "models" / name
    for name in [
        "practical.joblib",
        "distilbert_2epochs.joblib",
        "svm_c0.5.joblib",
        "THIRD_PARTY.md",
        "LICENSE-APACHE-2.0.txt",
    ]
]
paths += sorted((root / "models/distilbert").glob("*"))
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in paths:
        if path.is_file():
            archive.write(path, path.relative_to(root).as_posix())
digest = hashlib.sha256(destination.read_bytes()).hexdigest()
(root / "MODEL_BUNDLE.sha256").write_text(
    f"{digest}  support-ticket-models.zip\n", encoding="ascii", newline="\n"
)
print(f"Packaged {destination.name}: {destination.stat().st_size} bytes; sha256={digest}")
