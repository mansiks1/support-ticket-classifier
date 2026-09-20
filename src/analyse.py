"""Development error, feature, calibration and robustness analysis."""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import MODELS, REPORTS
from src.data import load_split, save_json
from src.metrics import probabilities
from src.predict import Predictor


def main():
    artifact = joblib.load(MODELS / "practical.joblib")
    model, metadata = artifact["model"], artifact["metadata"]
    val = load_split("validation")
    p = probabilities(model, val.text, metadata["temperature"])
    pred = model.classes_[p.argmax(axis=1)]
    frame = pd.DataFrame(
        {
            "text": val.text,
            "true": val.category,
            "predicted": pred,
            "confidence": p.max(axis=1),
            "correct": pred == val.category,
            "runner_up": model.classes_[np.argsort(p, axis=1)[:, -2]],
        }
    )
    samples = pd.concat(
        [
            frame[frame.correct]
            .groupby("true", sort=True)
            .head(1)
            .sample(10, random_state=2026)
            .assign(sample="correct"),
            frame[~frame.correct].sample(20, random_state=2026).assign(sample="error"),
            frame.sort_values("confidence").head(10).assign(sample="low_confidence"),
        ]
    )
    samples.to_csv(REPORTS / "prediction-examples.csv", index=False, lineterminator="\n")
    report = classification_report(
        val.category, pred, labels=model.classes_, output_dict=True, zero_division=0
    )
    save_json(REPORTS / "validation-per-class.json", report)
    classical = model if hasattr(model, "named_steps") else joblib.load(MODELS / "svm_c0.5.joblib")
    if hasattr(classical, "named_steps"):
        clf = classical["classifier"]
        weights = clf.coef_ if hasattr(clf, "coef_") else -clf.feature_log_prob_
        terms = classical["tfidf"].get_feature_names_out()
        features = {
            str(label): terms[np.argsort(weights[i])[-12:][::-1]].tolist()
            for i, label in enumerate(classical.classes_)
        }
        save_json(REPORTS / "top-features.json", features)
    occlusions = []
    for text in val.text.head(5):
        words = text.split()
        variants = [text] + [" ".join(words[:i] + words[i + 1 :]) for i in range(len(words))]
        probs = probabilities(model, variants, metadata["temperature"])
        chosen = int(probs[0].argmax())
        occlusions.append(
            {
                "text": text,
                "predicted": str(model.classes_[chosen]),
                "word_removal_delta": [
                    {"word": word, "delta": float(probs[0, chosen] - probs[i + 1, chosen])}
                    for i, word in enumerate(words)
                ],
            }
        )
    save_json(REPORTS / "occlusion.json", occlusions)
    cm = confusion_matrix(val.category, pred, labels=model.classes_)
    pd.DataFrame(cm, index=model.classes_, columns=model.classes_).to_csv(
        REPORTS / "validation-confusion.csv", lineterminator="\n"
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set(
        xlabel="Predicted class index",
        ylabel="True class index",
        title="Validation confusion (row normalised)",
    )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures/confusion.png", dpi=130)
    plt.close(fig)
    calibration = json.loads((REPORTS / "calibration.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    for key in ["before", "after"]:
        bins = calibration[key]["bins"]
        axes[0].plot(
            [b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", label=key
        )
    axes[0].set(
        xlabel="Mean confidence",
        ylabel="Empirical accuracy",
        title="Validation calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend()
    thresholds = calibration["thresholds"]
    axes[1].plot(
        [r["coverage"] for r in thresholds],
        [r["accuracy"] if r["accuracy"] is not None else np.nan for r in thresholds],
    )
    axes[1].set(
        xlabel="Coverage", ylabel="Accepted accuracy", title="Validation selective classification"
    )
    fig.tight_layout()
    fig.savefig(REPORTS / "figures/calibration.png", dpi=130)
    plt.close(fig)
    predictor = Predictor(MODELS / "practical.joblib")
    probes = {
        "clean": "My card has not arrived yet",
        "typo": "My crad has not arived yet",
        "case": "MY CARD HAS NOT ARRIVED YET",
        "spaces": "  My   card has not   arrived yet  ",
        "short": "card",
        "long": "My card has not arrived yet. " * 100,
        "empty": "",
        "unknown": "qzxwvv jklzz",
        "mixed": "Моя card не работает",
        "russian": "Деньги списались, но заказ не оформился",
        "ood_weather": "What will the weather be like tomorrow?",
        "ood_food": "How do I bake a chocolate cake?",
        "ambiguous": "My card payment is pending and the exchange rate looks wrong",
        "unicode": "💳🌏\u0000",
        "too_long": "x" * 4001,
    }
    outputs = []
    for name, text in probes.items():
        try:
            result = predictor.predict(text)[0]
        except ValueError as error:
            result = {"error": str(error)}
        outputs.append({"probe": name, "text": text[:250], "length": len(text), **result})
    save_json(REPORTS / "robustness-probes.json", outputs)
    variants = {
        "original": val.text.tolist(),
        "uppercase": val.text.str.upper().tolist(),
        "extra_spaces": val.text.str.replace(" ", "   ").tolist(),
        "one_typo": [s[:3] + s[4:5] + s[3:4] + s[5:] if len(s) > 5 else s for s in val.text],
    }
    from src.metrics import quality

    save_json(
        REPORTS / "robustness-validation.json",
        {
            name: quality(
                val.category, probabilities(model, texts, metadata["temperature"]), model.classes_
            )
            for name, texts in variants.items()
        },
    )
    print(samples[samples["sample"] == "error"].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
