"""TrayIcon state bookkeeping that doesn't need a display or pystray."""

from __future__ import annotations

import threading

from vocal.state import DictationState
from vocal.tray import TrayIcon


def _tray(**kw) -> tuple[TrayIcon, dict]:
    calls: dict[str, list] = {"voice": [], "stop": []}
    tray = TrayIcon(
        on_toggle_pause=lambda: None,
        on_quit=lambda: None,
        on_select_device=lambda i: None,
        on_select_model=lambda m: None,
        on_switch_mode=lambda m: None,
        on_open_phrasebook=lambda: None,
        on_select_voice=lambda v: calls["voice"].append(v),
        on_stop_speaking=lambda: calls["stop"].append(True),
        current_voice="piper-en-lessac-medium",
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


def test_select_voice_fires_callback_once() -> None:
    tray, calls = _tray()
    tray._select_voice("piper-en-lessac-medium")  # already current
    assert calls["voice"] == []
    tray._select_voice("kokoro-af_sarah")
    tray._select_voice("kokoro-af_sarah")
    assert calls["voice"] == ["kokoro-af_sarah"]
    assert tray._current_voice == "kokoro-af_sarah"
