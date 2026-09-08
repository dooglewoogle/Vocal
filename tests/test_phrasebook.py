"""Phrasebook save/load round trip."""

from __future__ import annotations

from pathlib import Path

from vocal.input.phrasebook import load_phrasebook, save_phrasebook


def test_save_then_load(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "phrasebook.toml"
    save_phrasebook({"Cooper Netties": "Kubernetes", 'say "hi"': "greet"}, path)
    pb = load_phrasebook(path)
    assert pb.replacements == {"Cooper Netties": "Kubernetes", 'say "hi"': "greet"}
    assert pb.apply_replacements("deploy to cooper netties") == "deploy to Kubernetes"
    assert not path.with_suffix(".toml.tmp").exists()


def test_save_empty(tmp_path: Path) -> None:
    path = save_phrasebook({}, tmp_path / "p.toml")
    assert load_phrasebook(path).replacements == {}
