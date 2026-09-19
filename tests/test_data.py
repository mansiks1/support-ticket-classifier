import pandas as pd
import pytest

from src.data import clean, normalise, validate, download, duplicate_groups, split_groups


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


def test_near_duplicates_grouped():
    groups, _ = duplicate_groups(
        ["Where is my bank card", "Where is my bank card", "Transfer failed"]
    )
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_split_groups_disjoint_and_reproducible():
    frame = pd.DataFrame(
        [
            {
                "text": f"{label} item {i}",
                "category": label,
                "key": f"{label}-{i}",
                "group": j * 50 + i,
            }
            for j, label in enumerate(["a", "b", "c"])
            for i in range(50)
        ]
    )
    first, second = split_groups(frame), split_groups(frame)
    for name in first:
        assert first[name].equals(second[name])
    assert sum(map(len, first.values())) == len(frame)


def test_download_local_cache_and_checksum(tmp_path, monkeypatch):
    import src.data as data

    monkeypatch.setattr(data, "RAW", tmp_path / "raw")
    monkeypatch.setattr(data, "ROOT", tmp_path)
    data.RAW.mkdir()
    path = data.RAW / "train.csv"
    path.write_bytes(b"text,category\ncard,a\n")
    assert data.download("train.csv") == path
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum"):
        data.download("train.csv")


def test_test_sealed_until_selection(tmp_path, monkeypatch):
    import src.data as data

    monkeypatch.setattr(data, "REPORTS", tmp_path)
    with pytest.raises(RuntimeError, match="sealed"):
        data.download("test.csv")
