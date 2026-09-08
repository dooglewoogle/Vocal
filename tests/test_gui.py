"""Tk window smoke test + settings-form coverage guard. Skipped without a display."""

from __future__ import annotations

import os
import sys

import pytest

from vocal.app import Signal
from vocal.config import VocalConfig
from vocal.gui.settings_tab import build_fields, config_leaves, get_path, set_path
from vocal.state import DictationState

pytestmark = pytest.mark.skipif(
    not (os.environ.get("DISPLAY") or sys.platform in ("win32", "darwin")),
    reason="needs a display",
)


class StubApp:
    """Just enough of VocalApp for the window to build and render."""

    def __init__(self) -> None:
        self.config = VocalConfig()
        self.cli_overridden = {"input.model.size"}
        self.on_state = Signal("s")
        self.on_speaking = Signal("sp")
        self.on_transcript = Signal("t")
        self.on_rebuild = Signal("r")
        self.state = DictationState.LISTENING
        self.calls: list = []

    def toggle_pause(self) -> None:
        self.calls.append("pause")

    def stop_speaking(self) -> None:
        self.calls.append("stop")

    def say(self, *a, **k) -> None:
        self.calls.append(("say", a, k))

    def set_voice(self, name) -> list[str]:
        self.calls.append(("voice", name))
        return []

    def set_phrasebook(self, pb) -> None:
        self.calls.append(("pb", pb))

    def apply_config(self, cfg) -> list[str]:
        self.calls.append(("apply", cfg))
        self.config = cfg
        return ["ok"]

    def request_shutdown(self) -> None:
        self.calls.append("quit")


def test_every_config_leaf_has_exactly_one_field_spec() -> None:
    specs = build_fields()
    paths = [s.path for s in specs]
    assert len(paths) == len(set(paths)), "duplicate FieldSpec paths"
    leaves = set(config_leaves())
    assert set(paths) == leaves, f"missing={leaves - set(paths)} extra={set(paths) - leaves}"
    cfg = VocalConfig()
    for s in specs:
        get_path(cfg, s.path)  # resolves
        if s.kind == "choice":
            assert s.choices, s.path


def test_get_set_path() -> None:
    cfg = VocalConfig()
    set_path(cfg, "output.speech.speed", 1.5)
    assert get_path(cfg, "output.speech.speed") == 1.5


@pytest.fixture
def window():
    pytest.importorskip("tkinter")
    from vocal.gui.window import VocalWindow

    app = StubApp()
    try:
        w = VocalWindow(app)
    except Exception as e:  # pragma: no cover - headless CI without Xvfb
        pytest.skip(f"Tk unavailable: {e}")
    yield w, app
    w.destroy()


def test_window_builds_and_pumps_events(window) -> None:
    w, app = window
    w.start_pump()
    app.on_state.emit(DictationState.RECORDING)
    app.on_speaking.emit(True)
    app.on_transcript.emit("hello world")
    # Drain the queue the way the pump would.
    w._pump()
    assert w.status._state_label.cget("text") == "Recording…"
    assert w.status._speaking_label.cget("text") == "Speaking"
    assert "hello world" in w.status._text.get("1.0", "end")


def test_settings_round_trip_and_parse_errors(window) -> None:
    w, app = window
    tab = w.settings
    tab._vars["output.speech.speed"].set("1.25")
    tab._vars["input.hotkey.key"].set("F9")
    tab._vars["input.engine"].set("hotkey")
    tab._vars["output.speech.duck"].set(False)
    tab._vars["output.speech.model_path"].set("")
    cfg = tab.collect()
    assert cfg.output.speech.speed == 1.25 and cfg.input.hotkey.key == "F9"
    assert cfg.input.engine == "hotkey" and cfg.output.speech.duck is False
    assert cfg.output.speech.model_path is None
    assert app.config.output.speech.speed == 1.0  # live config untouched until apply

    tab._vars["output.server.port"].set("not-a-number")
    with pytest.raises(ValueError, match="Port"):
        tab.collect()


def test_close_hides_with_tray_and_quits_without(window) -> None:
    w, app = window
    w.set_has_tray(True)
    w._on_close()
    assert app.calls == [] and w.root.state() == "withdrawn"
    w.set_has_tray(False)
    w._on_close()
    assert app.calls == ["quit"]


def test_run_bg_reports_on_ui_thread(window) -> None:
    import time

    w, _ = window
    got: list = []
    w.run_bg(lambda: 42, on_done=got.append)
    w.run_bg(lambda: 1 / 0, on_error=lambda e: got.append(type(e).__name__))
    deadline = time.time() + 2
    while time.time() < deadline and len(got) < 2:
        w._pump()
        time.sleep(0.01)
    assert sorted(map(str, got)) == ["42", "ZeroDivisionError"]
