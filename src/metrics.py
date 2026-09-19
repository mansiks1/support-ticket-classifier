import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize


def scores(model, texts):
    if hasattr(model, "predict_proba"):
        return np.log(np.clip(model.predict_proba(texts), 1e-12, 1))
    result = model.decision_function(texts)
    if result.ndim == 1:
        result = np.column_stack([-result, result])
    return result


def probabilities(model, texts, temperature=1.0):
    return softmax(scores(model, texts) / temperature, axis=1)


def calibration_metrics(y, probabilities_, classes):
    labels = np.asarray(classes)
    true = np.asarray(y)
    pred = labels[probabilities_.argmax(axis=1)]
    confidence = probabilities_.max(axis=1)
    onehot = (true[:, None] == labels[None, :]).astype(float)
    bins = []
    ece = 0.0
    for i in range(10):
        mask = (confidence >= i / 10) & (confidence < (i + 1) / 10 if i < 9 else confidence <= 1)
        if mask.any():
            acc = float((pred[mask] == true[mask]).mean())
            conf = float(confidence[mask].mean())
            ece += float(mask.mean()) * abs(acc - conf)
            bins.append({"count": int(mask.sum()), "confidence": conf, "accuracy": acc})
    return {
        "nll": float(log_loss(true, probabilities_, labels=labels)),
        "brier": float(((probabilities_ - onehot) ** 2).sum(axis=1).mean()),
        "ece": ece,
        "bins": bins,
    }


def quality(y, probabilities_, classes):
    pred = np.asarray(classes)[probabilities_.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "top2_accuracy": float(top_k_accuracy_score(y, probabilities_, k=2, labels=classes)),
        "macro_ap": float(
            average_precision_score(
                label_binarize(y, classes=classes), probabilities_, average="macro"
            )
        ),
    }


def threshold_rows(y, probabilities_, classes):
    correct = np.asarray(classes)[probabilities_.argmax(axis=1)] == np.asarray(y)
    confidence = probabilities_.max(axis=1)
    rows = []
    for threshold in np.arange(100) / 100:
        accepted = confidence >= threshold
        n = int(accepted.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "accepted": n,
                "coverage": float(accepted.mean()),
                "accuracy": float(correct[accepted].mean()) if n else None,
                "human": int((~accepted).sum()),
            }
        )
    return rows
