import pandas as pd
import pytest

from src.data import clean, normalise, validate, download


def test_schema():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"wrong": ["x"]}))


def test_clean_invalid_and_duplicates():
    frame = pd.DataFrame(
        {
            "text": ["Card?", "card!", None, " ", "Hello", "hello", "!!!"],
            "category": ["a", "a", "a", "a", "b", "c", "a"],
        }
    )
    result, audit = clean(frame)
    assert result.text.tolist() == ["Card?"]
    assert not result.isna().any().any()
    assert audit["conflicting_rows_removed"] == 2


def test_normalise_unicode():
    assert normalise("  CARD!?  status ") == "card status"
    assert normalise("Привет 🌏") == "привет"


def test_unknown_download_rejected():
    with pytest.raises(ValueError):
        download("../../secret")
