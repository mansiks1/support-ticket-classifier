"""Measure final production model on CPU using validation only."""
# ruff: noqa: E402 -- thread/device environment must precede native library imports

import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"

import numpy as np
import psutil

from src.config import MODELS, REPORTS
from src.data import load_split, save_json
from src.predict import Predictor

predictor = Predictor(MODELS / "practical.joblib")
texts = load_split("validation").text.head(32).tolist()
start = time.perf_counter()
predictor.predict(texts[0])
cold_seconds = time.perf_counter() - start
single, batch = [], []
for _ in range(5):
    start = time.perf_counter()
    predictor.predict(texts[0])
    single.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    predictor.predict(texts)
    batch.append((time.perf_counter() - start) * 1000 / len(texts))
result = {
    "device": "cpu",
    "threads": 1,
    "batch_size": 32,
    "repeats": 5,
    "cold_load_and_first_seconds": cold_seconds,
    "single_ms": float(np.median(single)),
    "batch_ms_per_item": float(np.median(batch)),
    "process_rss_mb": psutil.Process().memory_info().rss / 1024**2,
}
save_json(REPORTS / "cpu-serving.json", result)
print(result)
