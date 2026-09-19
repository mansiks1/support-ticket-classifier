import numpy as np
import pytest
from scipy.special import softmax

from src.calibrate import fit_temperature
from src.evaluate import wilson
from src.metrics import calibration_metrics, threshold_rows


def test_temperature_improves_overconfident_nll():
    logits = np.array([[8, 0, 0], [8, 0, 0], [0, 8, 0], [0, 0, 8]], dtype=float)
    labels = np.array(["a", "b", "b", "c"])
    classes = np.array(["a", "b", "c"])
    temperature = fit_temperature(logits, labels, classes)
    before = calibration_metrics(labels, softmax(logits, axis=1), classes)
    after = calibration_metrics(labels, softmax(logits / temperature, axis=1), classes)
    assert after["nll"] < before["nll"]
    np.testing.assert_array_equal(logits.argmax(axis=1), (logits / temperature).argmax(axis=1))


def test_threshold_coverage_monotonic():
    rows = threshold_rows(["a", "b"], np.array([[0.9, 0.1], [0.4, 0.6]]), ["a", "b"])
    assert rows[0]["coverage"] == 1
    assert rows[-1]["accepted"] == 0
    assert all(a["coverage"] >= b["coverage"] for a, b in zip(rows, rows[1:]))


def test_wilson_interval():
    low, high = wilson(95, 100)
    assert low < 0.95 < high < 1
    assert wilson(0, 0) == [None, None]


def test_perfect_calibration():
    result = calibration_metrics(["a", "b", "c"], np.eye(3), ["a", "b", "c"])
    assert result["brier"] == 0
    assert result["ece"] == 0
    assert result["nll"] == pytest.approx(0, abs=1e-12)
