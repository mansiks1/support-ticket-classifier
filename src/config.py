from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BASE_URL = (
    f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data"
)
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
