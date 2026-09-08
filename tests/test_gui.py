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
    # basic fields precede advanced ones so the Settings block reads top-down
    for fl in (dictation_fields(), speech_fields()):
        flags = [s.advanced for s in fl]
        assert flags == sorted(flags), "advanced fields must come after basic ones"
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
    """Numeric stand-in for a screenshot: every visible label/widget fits; blocks are in order."""
    w, _ = window
    w.root.geometry("760x560")
    w.root.deiconify()
    for tab in (w.dictation, w.speech):
        form = tab.form
        w.notebook.select(tab)
        form.set_advanced(advanced)
        w.root.update_idletasks()
        w.root.update()

        packed = form._form.pack_slaves()
        assert not [c for c in packed if isinstance(c, ttk.LabelFrame)], "no boxed sections: headings only"
        titles = [c.cget("text") for c in packed if isinstance(c, ttk.Label)]
        # header grid heading, then Settings heading; the Advanced block is a Collapsible
        assert titles == (["Whisper models", "Settings"] if tab is w.dictation else ["Voices", "Settings"]), titles
        assert form.advanced.body.winfo_ismapped() == advanced
        if tab is w.speech:  # server row is the very first thing on the tab
            assert form._widgets["output.server.enabled"].winfo_ismapped()
            assert form._widgets["output.server.enabled"].winfo_rooty() < form._widgets["output.speech.speed"].winfo_rooty()

        visible = [s for s in form.fields if s.path.startswith("output.server.") or advanced or not s.advanced]
        clipped = [(s.path, form._widgets[s.path].winfo_reqwidth(), form._widgets[s.path].winfo_width())
                   for s in visible if _clipped(form._widgets[s.path])]
        assert not clipped, clipped
        for parent in (form.settings_box, form.advanced.body):
            for child in parent.grid_slaves():
                if isinstance(child, ttk.Label) and child.winfo_ismapped():
                    assert not _clipped(child), child.cget("text")
    w.root.withdraw()


def test_every_field_has_help_text() -> None:
    from vocal.gui.settings_form import SERVER_FIELDS

    missing = [s.path for s in all_fields() + SERVER_FIELDS if not s.help or len(s.help) < 20]
    assert missing == [], missing


def test_tooltip_shows_after_delay_and_hides_on_leave(window) -> None:
    from vocal.gui.tooltip import Tooltip

    w, _ = window
    btn = ttk.Button(w.root, text="hover me")
    btn.pack()
    t = Tooltip(btn, "explanation", delay_ms=10)
    w.root.update()
    btn.event_generate("<Enter>")
    for _ in range(50):
        w.root.update()
        if t.visible:
            break
        w.root.after(5)
        w.root.update_idletasks()
    assert t.visible, "tooltip did not appear"
    label = t._tip.winfo_children()[0]
    assert label.cget("text") == "explanation"
    btn.event_generate("<Leave>")
    w.root.update()
    assert not t.visible


def test_form_rows_carry_tooltips(window) -> None:
    w, _ = window
    for form in (w.dictation.form, w.speech.form):
        for spec in form.fields:
            widget = form._widgets[spec.path]
            assert "<Enter>" in widget.bind(), spec.path


def test_wheel_scrolls_settings_page_from_any_widget_and_spares_comboboxes(window) -> None:
    w, _ = window
    w.root.geometry("760x300")  # short window so the Dictation page overflows
    w.root.deiconify()
    w.notebook.select(w.dictation)
    form = w.dictation.form
    form.set_advanced(True)
    w.root.update()
    canvas = form._canvas
    assert canvas.yview()[0] == 0.0

    # Wheel-down over a plain label inside the form scrolls the page.
    label = next(c for c in form.settings_box.grid_slaves() if isinstance(c, ttk.Label))
    label.event_generate("<Button-5>", x=2, y=2)
    w.root.update()
    assert canvas.yview()[0] > 0.0, "page did not scroll from a label"

    # Over a combobox: page scrolls back up, value unchanged.
    combo = form._widgets["input.engine"]
    before = combo.get()
    combo.event_generate("<Button-4>", x=2, y=2)
    w.root.update()
    assert combo.get() == before, "wheel must not change a combobox value"
    assert canvas.yview()[0] == 0.0

    # Over the Whisper grid (scrolls itself): page untouched.
    w.dictation._models.event_generate("<Button-5>", x=5, y=5)
    w.root.update()
    assert canvas.yview()[0] == 0.0
    w.root.withdraw()
