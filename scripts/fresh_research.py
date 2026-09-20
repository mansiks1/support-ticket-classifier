"""Create an isolated research workspace without changing published evidence."""

import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("destination", type=Path)
args = parser.parse_args()
source = Path(__file__).resolve().parents[1]
destination = args.destination.resolve()
if destination.exists() or destination == source or source in destination.parents:
    raise SystemExit("Destination must be a new directory outside the repository")
destination.mkdir(parents=True)
for name in ["src", "api", "scripts", "tests"]:
    shutil.copytree(source / name, destination / name, ignore=shutil.ignore_patterns("__pycache__"))
for name in [
    "pyproject.toml",
    "requirements.lock",
    "requirements-dev.lock",
    "requirements-transformer.lock",
    "PROTOCOL.md",
    "LICENSE",
    ".gitignore",
]:
    shutil.copy2(source / name, destination / name)
(destination / "data").mkdir()
shutil.copy2(source / "data/checksums.json", destination / "data/checksums.json")
(destination / "reports").mkdir()
(destination / "models").mkdir()
print("Created fresh research copy. Install dependencies there and run python scripts/reproduce.py")
