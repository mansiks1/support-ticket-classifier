"""Synthetic CI fixture, never a measured BANKING77 model or release artifact."""

from pathlib import Path

import joblib

from src.train import build

path = Path("models/practical.joblib")
if path.exists():
    raise RuntimeError("Refusing to overwrite an existing trained artifact")
texts = [
    "lost my card",
    "lost bank card",
    "card is lost",
    "where is card",
    "transfer pending",
    "bank transfer pending",
    "my transfer pending",
    "transfer delayed",
    "reset pin code",
    "change pin code",
    "forgot pin code",
    "new pin code",
]
model = build({"method": "lr"}).fit(texts, ["card"] * 4 + ["transfer"] * 4 + ["pin"] * 4)
path.parent.mkdir(exist_ok=True)
joblib.dump(
    {
        "model": model,
        "metadata": {
            "temperature": 1.0,
            "policy": {"threshold": 0.6},
            "practical_model": "SYNTHETIC_CI_FIXTURE",
            "language": "en",
            "priority_supported": False,
            "classes": model.classes_.tolist(),
        },
    },
    path,
)
