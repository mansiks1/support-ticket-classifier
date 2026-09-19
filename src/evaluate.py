"""One frozen evaluation event. Existing results can only be verified, not tuned."""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors

from src.config import MODELS, PROCESSED, REPORTS
from src.data import download, save_json, sha256, validate, normalise
from src.metrics import calibration_metrics, probabilities, quality


def wilson(correct, total):
    if not total:
        return [None, None]
    z = 1.96
    p = correct / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(center - half), float(center + half)]


def main(verify=False):
    result_path = REPORTS / "test-results.json"
    if result_path.exists() and not verify:
        raise RuntimeError(
            "Final test already evaluated. --verify only checks frozen reproducibility"
        )
    selection_path = REPORTS / "selection.json"
    selected = json.loads(selection_path.read_text())
    if sha256(MODELS / "practical.joblib") != selected["artifact_sha256"]:
        raise ValueError("Practical artifact changed after freeze")
    if sha256(REPORTS / "experiments.json") != selected["experiments_sha256"]:
        raise ValueError("Experiment log changed after freeze")
    if sha256(REPORTS / "splits.json") != selected["splits_sha256"]:
        raise ValueError("Split manifest changed after freeze")
    test = pd.read_csv(download("test.csv"))
    if not validate(test).all():
        raise ValueError("Invalid official test rows; do not silently drop")
    development = pd.read_csv(PROCESSED / "development.csv")
    vector = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    ref = vector.fit_transform(development.text)
    distances, _ = (
        NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=1)
        .fit(ref)
        .kneighbors(vector.transform(test.text))
    )
    exact = test.text.map(normalise).isin(set(development.key)).to_numpy()
    near = distances[:, 0] <= 0.08
    clean = ~(exact | near)
    artifact = joblib.load(MODELS / "practical.joblib")
    model = artifact["model"]
    p = probabilities(model, test.text, selected["temperature"])
    pred = model.classes_[p.argmax(axis=1)]
    accepted = p.max(axis=1) >= selected["policy"]["threshold"]
    correct = pred == test.category.to_numpy()
    practical = {
        "model": selected["practical_model"],
        "official": quality(test.category, p, model.classes_),
        "decontaminated": quality(test.category[clean], p[clean], model.classes_),
        "calibration": calibration_metrics(test.category, p, model.classes_),
        "selective": {
            "threshold": selected["policy"]["threshold"],
            "accepted": int(accepted.sum()),
            "coverage": float(accepted.mean()),
            "human": int((~accepted).sum()),
            "accuracy": float(correct[accepted].mean()) if accepted.any() else None,
            "accuracy_wilson_95": wilson(int(correct[accepted].sum()), int(accepted.sum())),
        },
    }
    research_path = MODELS / f"{selected['research_model']}.joblib"
    if sha256(research_path) != selected["research_artifact_sha256"]:
        raise ValueError("Research model changed after freeze")
    research_model = joblib.load(research_path)
    research_p = probabilities(research_model, test.text)
    result = {
        "selection_sha256": sha256(selection_path),
        "test_rows": len(test),
        "test_sha256": sha256(download("test.csv")),
        "exact_overlap_rows": int(exact.sum()),
        "near_or_exact_overlap_rows": int((exact | near).sum()),
        "decontaminated_rows": int(clean.sum()),
        "practical": practical,
        "research": {
            "model": selected["research_model"],
            "official": quality(test.category, research_p, research_model.classes_),
        },
    }
    if verify:
        prior = json.loads(result_path.read_text())
        if result != prior:
            raise AssertionError("Frozen test metrics do not reproduce exactly")
        print("Frozen test metrics reproduced exactly; no selection changes")
        return
    save_json(result_path, result)
    save_json(
        REPORTS / "test-per-class.json",
        classification_report(
            test.category, pred, labels=model.classes_, output_dict=True, zero_division=0
        ),
    )
    pd.DataFrame(
        confusion_matrix(test.category, pred, labels=model.classes_),
        index=model.classes_,
        columns=model.classes_,
    ).to_csv(REPORTS / "test-confusion.csv", lineterminator="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    main(parser.parse_args().verify)
