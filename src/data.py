"""Download development data; audit and group duplicates without test access."""

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from src.config import BASE_URL, PROCESSED, RAW, REPORTS, ROOT, SEED


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(name):
    if name not in {"train.csv", "test.csv", "categories.json"}:
        raise ValueError("Unknown dataset file")
    if name == "test.csv" and not (REPORTS / "selection.json").exists():
        raise RuntimeError("Test is sealed until selection.json has been frozen")
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if not path.exists():
        with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
            content = response.read()
        path.write_bytes(content)
    manifest_path = ROOT / "data/checksums.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    checksum = sha256(path)
    if name in manifest and manifest[name] != checksum:
        raise ValueError(f"Checksum mismatch for {name}")
    manifest[name] = checksum
    save_json(manifest_path, manifest)
    return path


def normalise(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def validate(frame):
    if list(frame.columns) != ["text", "category"]:
        raise ValueError("Expected exactly text,category")
    valid = frame.apply(lambda s: s.map(lambda x: isinstance(x, str) and bool(x.strip())))
    return valid.all(axis=1) & frame.text.fillna("").map(
        lambda x: bool(normalise(x)) if isinstance(x, str) else False
    )


def clean(frame):
    mask = validate(frame)
    good = frame.loc[mask].copy()
    good["key"] = good.text.map(normalise)
    conflicts = good.groupby("key").category.nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    conflict_rows = int(good.key.isin(conflicts).sum())
    good = good[~good.key.isin(conflicts)]
    duplicates = int(good.duplicated("key").sum())
    good = good.drop_duplicates("key").reset_index(drop=True)
    return good, {
        "raw_rows": len(frame),
        "invalid_rows": int((~mask).sum()),
        "conflicting_rows_removed": conflict_rows,
        "normalised_duplicates_removed": duplicates,
        "clean_rows": len(good),
    }


def duplicate_groups(texts):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    graph = NearestNeighbors(metric="cosine", radius=0.08, n_jobs=1).fit(matrix)
    edges = graph.radius_neighbors_graph(matrix, mode="connectivity")
    count, groups = connected_components(edges, directed=False)
    return groups, {
        "near_duplicate_components": int(count),
        "nontrivial_components": int((np.bincount(groups) > 1).sum()),
        "near_duplicate_pairs": int((edges.nnz - len(texts)) // 2),
    }


def split_groups(frame):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dev_idx, val_idx = next(cv.split(frame.text, frame.category, frame.group))
    dev = frame.iloc[dev_idx]
    cv2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    fit_idx, cal_idx = next(cv2.split(dev.text, dev.category, dev.group))
    parts = {
        "train": dev.iloc[fit_idx],
        "calibration": dev.iloc[cal_idx],
        "validation": frame.iloc[val_idx],
    }
    labels = set(frame.category)
    for name, part in parts.items():
        if set(part.category) != labels:
            raise ValueError(f"Missing classes in {name}")
    for a, b in [("train", "calibration"), ("train", "validation"), ("calibration", "validation")]:
        assert not set(parts[a].group) & set(parts[b].group)
        assert not set(parts[a].key) & set(parts[b].key)
    return parts


def prepare():
    frame = pd.read_csv(download("train.csv"))
    categories = json.loads(download("categories.json").read_text())
    if not set(frame.category.dropna()) <= set(categories):
        raise ValueError("Unknown label")
    cleaned, audit = clean(frame)
    # Counts only; never publish detected potential identifiers.
    audit["potential_email_rows"] = int(
        cleaned.text.str.contains(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", regex=True).sum()
    )
    audit["long_digit_sequence_rows"] = int(cleaned.text.str.contains(r"\d{8,}", regex=True).sum())
    audit["null_cells"] = int(frame.isna().sum().sum())
    audit["exact_duplicate_rows"] = int(frame.duplicated("text").sum())
    audit["class_counts"] = cleaned.category.value_counts().sort_index().to_dict()
    audit["text_length_quantiles"] = {
        str(k): float(v)
        for k, v in cleaned.text.str.len().quantile([0, 0.25, 0.5, 0.75, 0.95, 1]).items()
    }
    save_json(REPORTS / "data-audit.json", audit)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED / "development.csv", index=False)
    print(json.dumps({k: v for k, v in audit.items() if k != "class_counts"}, indent=2))


def split():
    frame = pd.read_csv(PROCESSED / "development.csv")
    groups, duplicate_audit = duplicate_groups(frame.text)
    frame["group"] = groups
    parts = split_groups(frame)
    manifest = {"seed": SEED, **duplicate_audit, "splits": {}}
    for name, part in parts.items():
        path = PROCESSED / f"{name}.csv"
        part.to_csv(path, index=False)
        manifest["splits"][name] = {
            "rows": len(part),
            "classes": part.category.nunique(),
            "sha256": sha256(path),
            "class_counts": part.category.value_counts().sort_index().to_dict(),
        }
    save_json(REPORTS / "splits.json", manifest)
    print({name: len(part) for name, part in parts.items()})


def load_split(name):
    if name not in {"train", "calibration", "validation"}:
        raise ValueError("Only development splits allowed")
    path = PROCESSED / f"{name}.csv"
    manifest = json.loads((REPORTS / "splits.json").read_text())
    if sha256(path) != manifest["splits"][name]["sha256"]:
        raise ValueError("Split checksum mismatch")
    return pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "split"])
    args = parser.parse_args()
    {"prepare": prepare, "split": split}[args.action]()
