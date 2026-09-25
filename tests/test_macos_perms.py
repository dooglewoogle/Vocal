"""macOS permission helpers: the parts that are pure logic and run on any OS."""

from __future__ import annotations

import sys

import pytest

from vocal import macos_perms


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/Applications/iTerm.app/Contents/MacOS/iTerm2", "iTerm"),
        ("/Applications/Visual Studio Code.app/Contents/Frameworks/"
         "Code Helper (Renderer).app/Contents/MacOS/Code Helper (Renderer)", "Visual Studio Code"),
        ("/opt/homebrew/bin/python3.13", None),
    ],
)
def test_outermost_app(path: str, expected: str | None) -> None:
    assert macos_perms.outermost_app(path) == expected


def test_responsible_app_falls_back_to_term_program(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macos_perms.sys, "platform", "linux")
    assert macos_perms.responsible_app({"TERM_PROGRAM": "iTerm.app"}) == "iTerm"
    assert macos_perms.responsible_app({"TERM_PROGRAM": "WezTerm"}) == "WezTerm"
    assert macos_perms.responsible_app({}) == macos_perms.UNKNOWN_APP


def test_status_unavailable_off_macos() -> None:
    if sys.platform == "darwin":
        pytest.skip("runs the real check on macOS")
    assert macos_perms.status() == {"input_monitoring": None, "accessibility": None}


def test_advice_names_missing_panes_and_app() -> None:
    lines = macos_perms.advice({"input_monitoring": False, "accessibility": True}, "iTerm")
    assert "Input Monitoring." in lines[0] and "Accessibility" not in lines[0]
    assert "iTerm" in lines[0] and "quit iTerm" in lines[1]

    both = macos_perms.advice({"input_monitoring": False, "accessibility": False}, "iTerm")
    assert "Input Monitoring and Accessibility" in both[0]


@pytest.mark.parametrize("st", [
    {"input_monitoring": True, "accessibility": True},
    {"input_monitoring": None, "accessibility": None},  # could not check: don't claim anything is missing
])
def test_advice_when_nothing_known_missing(st: dict) -> None:
    lines = macos_perms.advice(st, "iTerm")
    assert len(lines) == 1 and "Secure Keyboard Entry" in lines[0]
