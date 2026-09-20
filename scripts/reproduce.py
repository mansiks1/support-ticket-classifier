"""Full research in a fresh checkout, or frozen verification in an existing one."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
os.environ["NLTK_DATA"] = str(root / "data/raw/nltk")
os.environ["HF_HOME"] = str(root / "data/raw/hf")
os.environ["HF_HUB_DISABLE_XET"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument("--verify", action="store_true")
args = parser.parse_args()


def run(*arguments):
    subprocess.run([sys.executable, *arguments], check=True)


if args.verify:
    run("-m", "src.evaluate", "--verify")
else:
    # Published reports are evidence, not mutable experiment output: never silently overwrite.
    if (root / "reports/selection.json").exists():
        raise SystemExit(
            "Published selection exists. Use scripts/fresh_research.py to create an isolated reproduction copy."
        )
    run("-m", "src.data", "prepare")
    run("-m", "src.data", "split")
    run("-m", "src.eda")
    run("-m", "src.train", "baseline")
    run("-m", "src.train", "classical")
    run(
        "-c",
        "import nltk,os; nltk.download('wordnet',download_dir=os.environ['NLTK_DATA'],quiet=True,raise_on_error=True)",
    )
    run("-m", "src.train", "ablation")
    run("-m", "src.transformer_experiment")
    run("-m", "src.calibrate")
    run("-m", "src.analyse")
    run("-m", "src.evaluate")
