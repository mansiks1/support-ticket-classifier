"""Bounded, development-only experiments. No test loading is possible here."""

import argparse
import importlib.metadata
import json
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from src.config import MODELS, REPORTS, SEED
from src.data import load_split, save_json
from src.metrics import probabilities, quality
from src.preprocessing import lemmatise, remove_punctuation


def configs(stage):
    if stage == "baseline":
        return [{"name": "dummy", "method": "dummy"}, {"name": "lr_base", "method": "lr"}]
    if stage == "classical":
        return (
            [{"name": f"svm_c{c}", "method": "svm", "C": c} for c in [0.5, 1, 2]]
            + [{"name": f"nb_a{a}", "method": "nb", "alpha": a} for a in [0.1, 1]]
            + [{"name": f"lr_c{c}", "method": "lr", "C": c} for c in [0.5, 2]]
        )
    return [
        {"name": f"lr_{name}", "method": "lr", **change}
        for name, change in [
            ("case", {"lowercase": False}),
            ("punct", {"preprocessing": "punct"}),
            ("stop", {"stop_words": "english"}),
            ("lemma", {"preprocessing": "lemma"}),
            ("unigram", {"ngram_range": [1, 1]}),
            ("char", {"analyzer": "char_wb", "ngram_range": [3, 5]}),
            ("vocab", {"max_features": 5000}),
            ("balanced", {"class_weight": "balanced"}),
        ]
    ]


def build(config):
    method = config["method"]
    vector = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_features": 30000,
        "lowercase": True,
        "sublinear_tf": True,
    }
    for key in ["ngram_range", "max_features", "lowercase", "stop_words", "analyzer"]:
        if key in config:
            vector[key] = tuple(config[key]) if key == "ngram_range" else config[key]
    if config.get("preprocessing"):
        vector["preprocessor"] = {"punct": remove_punctuation, "lemma": lemmatise}[
            config["preprocessing"]
        ]
    if method == "dummy":
        classifier = DummyClassifier(strategy="most_frequent", random_state=SEED)
    elif method == "lr":
        classifier = LogisticRegression(
            C=config.get("C", 1),
            max_iter=500,
            solver="lbfgs",
            class_weight=config.get("class_weight"),
            random_state=SEED,
        )
    elif method == "svm":
        classifier = LinearSVC(C=config.get("C", 1), random_state=SEED, max_iter=3000)
    elif method == "nb":
        classifier = ComplementNB(alpha=config.get("alpha", 1))
    else:
        raise ValueError("Unknown model family")
    return Pipeline([("tfidf", TfidfVectorizer(**vector)), ("classifier", classifier)])


def benchmark(model, texts):
    batch = list(texts)
    probabilities(model, batch[:1])
    batch_times, single_times = [], []
    for _ in range(5):
        start = time.perf_counter()
        probabilities(model, batch)
        batch_times.append((time.perf_counter() - start) * 1000 / len(batch))
        start = time.perf_counter()
        probabilities(model, batch[:1])
        single_times.append((time.perf_counter() - start) * 1000)
    return {
        "batch_ms_per_item": float(np.median(batch_times)),
        "single_ms": float(np.median(single_times)),
        "batch_size": len(batch),
    }


def run(stage):
    if (REPORTS / "selection.json").exists():
        raise RuntimeError("Selection frozen; use a fresh checkout to repeat research")
    train, val = load_split("train"), load_split("validation")
    MODELS.mkdir(exist_ok=True)
    records = REPORTS / "experiments.json"
    results = json.loads(records.read_text()) if records.exists() else []
    for config in configs(stage):
        if any(row["name"] == config["name"] for row in results):
            continue
        model = build(config)
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            started = time.perf_counter()
            model.fit(train.text, train.category)
            elapsed = time.perf_counter() - started
            p = probabilities(model, val.text)
            timing = benchmark(model, val.text)
        artifact = MODELS / f"{config['name']}.joblib"
        joblib.dump(model, artifact, compress=3)
        row = {
            "name": config["name"],
            "config": config,
            "parameters": {k: str(v) for k, v in model.get_params().items()},
            "validation": quality(val.category, p, model.classes_),
            "train_seconds": elapsed,
            **timing,
            "model_bytes": artifact.stat().st_size,
            "process_rss_mb": psutil.Process().memory_info().rss / 1024**2,
        }
        results.append(row)
        save_json(records, results)
        pd.DataFrame(
            [
                {
                    **{
                        k: v
                        for k, v in r.items()
                        if k not in {"config", "parameters", "validation"}
                    },
                    **r["validation"],
                }
                for r in results
            ]
        ).to_csv(REPORTS / "experiments.csv", index=False)
        print(
            f"{config['name']}: macro-F1={row['validation']['macro_f1']:.4f}, fit={elapsed:.2f}s",
            flush=True,
        )
    save_json(
        REPORTS / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.system(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / 1024**3,
            "threads": 1,
            "seed": SEED,
            "versions": {
                k: importlib.metadata.version(k)
                for k in ["numpy", "scipy", "pandas", "scikit-learn", "joblib"]
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["baseline", "classical", "ablation"])
    run(parser.parse_args().stage)
