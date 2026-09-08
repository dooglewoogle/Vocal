"""TrayIcon state bookkeeping and menu shape (no display needed)."""

from __future__ import annotations

import threading

import pytest

from vocal.state import DictationState
from vocal.tray import TrayIcon


def _tray(**kw) -> tuple[TrayIcon, dict]:
    calls: dict[str, list] = {"pause": [], "stop": [], "quit": [], "open": []}
    tray = TrayIcon(
        on_toggle_pause=lambda: calls["pause"].append(True),
        on_quit=lambda: calls["quit"].append(True),
        on_stop_speaking=lambda: calls["stop"].append(True),
        **kw,
    )
    return tray, calls


def test_speaking_overlay_is_independent_of_state() -> None:
    tray, _ = _tray()
    tray.set_state(DictationState.SLEEPING)
    assert tray._title() == "Vocal — Paused"
    tray.set_speaking(True)
    assert tray._title() == "Vocal — Speaking · Paused"
    tray.set_state(DictationState.LISTENING)
    assert tray._title() == "Vocal — Speaking · Listening"
    tray.set_speaking(False)
    assert tray._title() == "Vocal — Listening"


def test_set_speaking_from_other_thread() -> None:
    tray, _ = _tray()
    t = threading.Thread(target=tray.set_speaking, args=(True,))
    t.start()
    t.join()
    assert tray._speaking is True


def _menu_labels(menu) -> list[str]:
    out = []
    for item in menu.items:
        text = str(item)
        out.append(text)
    return out


def test_menu_is_slim_and_open_is_optional() -> None:
    pytest.importorskip("pystray")
    tray, calls = _tray(on_open=lambda: calls["open"].append(True))
    labels = _menu_labels(tray._build_menu())
    # status, sep, Open, Pause, Stop speaking, sep, Quit
    assert labels == ["Status: Listening", "- - - -", "Open Vocal", "Pause", "Stop speaking", "- - - -", "Quit"]

    headless, _ = _tray()
    assert "Open Vocal" not in _menu_labels(headless._build_menu())


def test_menu_callbacks_route_to_app() -> None:
    pystray = pytest.importorskip("pystray")
    tray, calls = _tray(on_open=lambda: calls["open"].append(True))
    tray.set_speaking(True)
    by_label = {str(i): i for i in tray._build_menu().items}
    for label, key in (("Open Vocal", "open"), ("Pause", "pause"), ("Stop speaking", "stop"), ("Quit", "quit")):
        by_label[label](None)  # pystray MenuItem.__call__(icon)
        assert calls[key] == [True], label
    assert by_label["Stop speaking"].enabled is True
    tray.set_speaking(False)
    tray.set_state(DictationState.SLEEPING)
    labels = _menu_labels(tray._build_menu())
    assert "Resume" in labels and "Status: Paused" in labels
    assert isinstance(pystray.Menu.SEPARATOR, pystray.MenuItem)


def test_run_detached_reports_false_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vocal.tray.sys.platform", "darwin")
    tray, _ = _tray()
    assert tray.run_detached() is False
    assert tray._thread is None
