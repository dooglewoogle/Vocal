"""Tests for CLI engine-mode resolution."""

from __future__ import annotations

import argparse

import pytest

from vocal.cli import _resolve_initial_mode


def _args(**kw) -> argparse.Namespace:
    base = dict(hotkey=False, live=False, mode=None, duck=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({}, "live"),
        ({"hotkey": True}, "hotkey"),
        ({"live": True}, "live"),
        ({"mode": "ptt"}, "hotkey"),
        ({"mode": "toggle"}, "hotkey"),
        ({"duck": True}, "hotkey"),
        ({"mode": "ptt", "live": True}, "live"),
        ({"duck": True, "live": True}, "live"),
        ({"hotkey": True, "mode": "ptt", "duck": True}, "hotkey"),
    ],
)
def test_resolve_initial_mode(kw, expected):
    assert _resolve_initial_mode(_args(**kw)) == expected
