from pathlib import Path


def test_protocol_is_present():
    assert "Seed 2026" in Path("PROTOCOL.md").read_text(encoding="utf-8")
