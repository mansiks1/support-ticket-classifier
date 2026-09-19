"""Freeze research/practical selection, calibration and abstention before test."""

import json

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss

from src.config import MODELS, REPORTS
from src.data import load_split, save_json, sha256
from src.metrics import calibration_metrics, scores, threshold_rows


def fit_temperature(logits, y, classes):
    def objective(log_temperature):
        return log_loss(y, softmax(logits / np.exp(log_temperature), axis=1), labels=classes)

    result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimisation failed")
    return float(np.exp(result.x))


def main():
    if (REPORTS / "selection.json").exists():
        raise RuntimeError("Selection already frozen")
    results = json.loads((REPORTS / "experiments.json").read_text())
    if (REPORTS / "transformer.json").exists():
        results.append(json.loads((REPORTS / "transformer.json").read_text()))
    candidates = [r for r in results if r["name"] != "dummy"]
    research = max(candidates, key=lambda r: r["validation"]["macro_f1"])
    eligible = [
        r
        for r in candidates
        if r["validation"]["macro_f1"] >= research["validation"]["macro_f1"] - 0.01
    ]
    practical = min(eligible, key=lambda r: (r["model_bytes"], r["single_ms"]))
    model = joblib.load(MODELS / f"{practical['name']}.joblib")
    cal, val = load_split("calibration"), load_split("validation")
    temperature = fit_temperature(scores(model, cal.text), cal.category, model.classes_)
    logits = scores(model, val.text)
    before_p, after_p = softmax(logits, axis=1), softmax(logits / temperature, axis=1)
    before = calibration_metrics(val.category, before_p, model.classes_)
    after = calibration_metrics(val.category, after_p, model.classes_)
    retained = after["nll"] < before["nll"]
    if not retained:
        temperature = 1.0
    p = after_p if retained else before_p
    sweep = threshold_rows(val.category, p, model.classes_)
    feasible = [r for r in sweep if r["accepted"] >= 100 and r["accuracy"] >= 0.95]
    policy = (
        max(feasible, key=lambda r: r["coverage"])
        if feasible
        else {
            "threshold": 1.01,
            "coverage": 0.0,
            "accuracy": None,
            "accepted": 0,
            "human": len(val),
        }
    )
    metadata = {
        "research_model": research["name"],
        "practical_model": practical["name"],
        "temperature": temperature,
        "calibration_retained": retained,
        "policy": policy,
        "classes": model.classes_.tolist(),
        "language": "en",
        "priority_supported": False,
        "dataset": "BANKING77",
        "max_characters": 4000,
        "seed": 2026,
    }
    artifact = MODELS / "practical.joblib"
    joblib.dump({"model": model, "metadata": metadata}, artifact, compress=3)
    metadata = {
        **metadata,
        "artifact_sha256": sha256(artifact),
        "research_artifact_sha256": sha256(MODELS / f"{research['name']}.joblib"),
        "experiments_sha256": sha256(REPORTS / "experiments.json"),
        "splits_sha256": sha256(REPORTS / "splits.json"),
    }
    save_json(
        REPORTS / "calibration.json",
        {
            "before": before,
            "after": after,
            "candidate_temperature": float(
                fit_temperature(scores(model, cal.text), cal.category, model.classes_)
            ),
            "retained": retained,
            "thresholds": sweep,
        },
    )
    save_json(REPORTS / "selection.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
