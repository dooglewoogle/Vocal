"""Tk window smoke test + settings-form coverage guard. Skipped without a display."""

from __future__ import annotations

import os
import sys

import pytest
from tkinter import ttk

from vocal.app import Signal
from vocal.config import VocalConfig
from vocal.gui.settings_form import (
    OWNED_ELSEWHERE,
    all_fields,
    config_leaves,
    dictation_fields,
    get_path,
    set_path,
    speech_fields,
)
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


def test_every_config_leaf_is_owned_exactly_once() -> None:
    specs = all_fields()
    paths = [s.path for s in specs]
    assert len(paths) == len(set(paths)), "duplicate FieldSpec paths"
    assert not (set(paths) & OWNED_ELSEWHERE), "a leaf is both a form field and owned elsewhere"
    leaves = set(config_leaves())
    covered = set(paths) | OWNED_ELSEWHERE
    assert covered == leaves, f"missing={leaves - covered} extra={covered - leaves}"
    assert all(s.path.startswith("input.") for s in dictation_fields())
    assert all(not s.path.startswith("input.") for s in speech_fields())
    cfg = VocalConfig()
    for s in specs:
        get_path(cfg, s.path)  # resolves
        if s.kind == "choice":
            assert s.choices, s.path
            for k in (s.labels or {}):
                assert k in s.choices, (s.path, k)


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
    d, s = w.dictation.form, w.speech.form
    assert d._vars["input.engine"].get() == "live (always listening)"  # label shown, not raw value
    d._vars["input.hotkey.key"].set("F9")
    d._vars["input.engine"].set("hotkey (hold to record)")
    cfg = d.collect()
    assert cfg.input.hotkey.key == "F9" and cfg.input.engine == "hotkey"
    assert cfg.output.speech.speed == 1.0  # untouched side carried over from live config

    s._vars["output.speech.speed"].set("1.25")
    s._vars["output.speech.duck"].set(False)
    s._vars["output.speech.model_path"].set("")
    cfg = s.collect()
    assert cfg.output.speech.speed == 1.25 and cfg.output.speech.duck is False
    assert cfg.output.speech.model_path is None
    assert app.config.output.speech.speed == 1.0  # live config untouched until apply

    s._vars["output.server.port"].set("not-a-number")
    with pytest.raises(ValueError, match="Port"):
        s.collect()


def test_grids_mark_current_and_drive_apply(window) -> None:
    w, app = window
    assert w.dictation._models.set(app.config.input.model.size, "current") == "✓"
    assert w.speech._voices.set(app.config.output.speech.voice, "current") == "✓"
    w.speech._voices.selection_set("piper-en-amy-low")
    w.speech._use_voice()
    import time
    deadline = time.time() + 2
    while time.time() < deadline and not any(c[0] == "voice" for c in app.calls if isinstance(c, tuple)):
        w._pump()
        time.sleep(0.01)
    assert ("voice", "piper-en-amy-low") in app.calls


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


def _clipped(widget) -> bool:
    """True if the widget was given less room than it asked for (text would be cut off)."""
    return widget.winfo_width() < widget.winfo_reqwidth() or widget.winfo_height() < widget.winfo_reqheight()


@pytest.mark.parametrize("advanced", [False, True])
def test_forms_lay_out_without_clipping(window, advanced) -> None:
    """Numeric stand-in for a screenshot: every label/widget fits, sections keep their order."""
    w, _ = window
    w.root.geometry("760x560")
    w.root.deiconify()
    for tab, first_section in ((w.dictation, "Whisper model"), (w.speech, "Voice model")):
        form = tab.form
        w.notebook.select(tab)
        form._show_advanced.set(advanced)
        form._toggle_advanced()
        w.root.update_idletasks()
        w.root.update()

        # Section order: grid box first, then the model section right under it.
        packed = [c for c in form._form.pack_slaves()]
        titles = [c.cget("text") for c in packed if isinstance(c, ttk.LabelFrame)]
        assert titles[0] in ("Whisper models", "Voices"), titles
        if advanced or first_section == "Whisper model":
            assert titles[1] == first_section, titles
        # Advanced-only sections hidden in basic mode
        if not advanced:
            assert "Voice detection" not in titles and "Server" not in titles

        clipped = [
            (spec.path, form._widgets[spec.path].winfo_reqwidth(), form._widgets[spec.path].winfo_width())
            for spec in form.fields
            if (advanced or not spec.advanced) and _clipped(form._widgets[spec.path])
        ]
        assert not clipped, clipped
        # Labels/tips in each visible section must fit their cell
        for frame in packed:
            if not isinstance(frame, ttk.LabelFrame):
                continue
            for child in frame.grid_slaves():
                if isinstance(child, ttk.Label) and child.winfo_ismapped():
                    assert not _clipped(child), (frame.cget("text"), child.cget("text"))
    w.root.withdraw()
