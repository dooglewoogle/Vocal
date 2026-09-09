"""Borrowing the system PyGObject from a venv only when the CPython build matches."""

from __future__ import annotations

from pathlib import Path

from vocal.utils import find_system_gi


def _fake_dist(tmp_path: Path, name: str, so: str) -> str:
    d = tmp_path / name / "gi"
    d.mkdir(parents=True)
    (d / so).write_bytes(b"")
    return str(tmp_path / name)


def test_find_system_gi_matches_only_same_cache_tag(tmp_path: Path) -> None:
    wrong = _fake_dist(tmp_path, "py311", "_gi.cpython-311-x86_64-linux-gnu.so")
    right = _fake_dist(tmp_path, "py310", "_gi.cpython-310-x86_64-linux-gnu.so")
    assert find_system_gi([wrong, right], cache_tag="cpython-310") == right
    assert find_system_gi([wrong], cache_tag="cpython-310") is None
    assert find_system_gi([str(tmp_path / "missing")], cache_tag="cpython-310") is None
