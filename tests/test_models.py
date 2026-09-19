import joblib
import numpy as np
import pytest

from src.train import build
from src.metrics import probabilities
from src.predict import Predictor
from src.preprocessing import remove_punctuation


@pytest.fixture
def fitted():
    texts = [
        "lost card please",
        "my lost card",
        "find lost card",
        "lost my card now",
        "bank transfer pending",
        "pending bank transfer",
        "transfer not arrived",
        "waiting for transfer",
        "reset my pin",
        "change pin please",
        "pin code reset",
        "new pin code",
    ]
    labels = ["card"] * 4 + ["transfer"] * 4 + ["pin"] * 4
    return build({"method": "lr"}).fit(texts, labels), texts, labels


@pytest.fixture
def artifact(fitted, tmp_path):
    model, _, _ = fitted
    path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "metadata": {
                "temperature": 1.0,
                "policy": {"threshold": 0.6},
                "classes": model.classes_.tolist(),
                "practical_model": "test",
                "language": "en",
                "priority_supported": False,
            },
        },
        path,
    )
    return path


def test_train_smoke(fitted):
    model, texts, labels = fitted
    assert (model.predict(texts) == labels).mean() > 0.9


def test_feature_order_and_serialisation(fitted, tmp_path):
    model, texts, _ = fitted
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(
        model["tfidf"].get_feature_names_out(), loaded["tfidf"].get_feature_names_out()
    )
    np.testing.assert_allclose(probabilities(model, texts), probabilities(loaded, texts))


def test_reproducibility(fitted):
    model, texts, labels = fitted
    second = build({"method": "lr"}).fit(texts, labels)
    np.testing.assert_allclose(
        probabilities(model, texts), probabilities(second, texts), atol=1e-12
    )


def test_single_and_batch(artifact):
    predictor = Predictor(artifact)
    one = predictor.predict("lost my card")
    assert one == predictor.predict(["lost my card"])
    assert len(predictor.predict(["lost card", "reset pin"])) == 2
    assert 0 <= one[0]["confidence"] <= 1


@pytest.mark.parametrize("value", ["", "  ", 123, None, [], [1], "a" * 4001, ["ok"] * 65])
def test_invalid_input(artifact, value):
    with pytest.raises(ValueError):
        Predictor(artifact).predict(value)


def test_unknown_symbols(artifact):
    result = Predictor(artifact).predict("💳\u0000世界🙂")[0]
    assert np.isfinite(result["confidence"])


def test_abstention_contract(artifact):
    predictor = Predictor(artifact)
    predictor.metadata["policy"]["threshold"] = 1.01
    result = predictor.predict("lost card")[0]
    assert result["requires_human"] and result["category"] is None
    assert result["suggested_category"] in predictor.model.classes_
    assert result["priority"] is None


def test_punctuation():
    assert remove_punctuation("Hello, CARD!") == "hello  card "
