"""Reusable, data-driven settings form over VocalConfig.

Each tab embeds one :class:`SettingsForm`: an optional custom header (grid,
server row), a **Settings** block with the everyday fields, and a collapsed
**Advanced** block with everything else. Every config leaf is either a field on
exactly one form or listed in :data:`OWNED_ELSEWHERE`; a test guards this.
The form edits a deep copy; "Save & Apply" hands that copy to
``VocalApp.apply_config`` on a worker thread.
"""

from __future__ import annotations

import copy
import tkinter as tk
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from tkinter import ttk
from typing import TYPE_CHECKING, Any

from vocal.config import ENGINES, VocalConfig

if TYPE_CHECKING:
    from vocal.gui.window import VocalWindow

DEFAULT_DEVICE = "(system default)"

#: Config leaves deliberately absent from every form.
OWNED_ELSEWHERE: frozenset[str] = frozenset({
    "input.model.size",  # Whisper grid on the Dictation tab
    "output.speech.voice",  # voice grid on the Speech tab
    "input.phrasebook.seed",  # Phrasebook tab
    "input.phrasebook.replace",  # Phrasebook tab
})


@dataclass(frozen=True)
class FieldSpec:
    path: str
    label: str
    kind: str  # bool | int | float | str | choice | device_in | device_out
    advanced: bool = False
    choices: tuple[str, ...] | None = None
    labels: Mapping[str, str] | None = None  # choice value → display text
    help: str | None = None


def dictation_fields() -> list[FieldSpec]:
    """Input side. Basic fields first, then the Advanced block, each in display order."""
    F = FieldSpec
    return [
        F("input.engine", "Dictation mode", "choice", choices=ENGINES,
          labels={"live": "live (always listening)", "hotkey": "hotkey (hold to record)"}),
        F("input.hotkey.key", "Hotkey (hold to record; hold to mute in live mode)", "str",
          help="e.g. PAUSE, F18, SCROLLLOCK, END"),
        F("input.live.min_silence_duration_ms", "End of sentence after silence (ms)", "int"),
        F("input.hotkey.duck", "Lower system volume while recording", "bool"),
        # ── Advanced ──
        F("input.model.compute_type", "Compute type", "choice", advanced=True, choices=("int8", "float32")),
        F("input.model.beam_size", "Beam size (1 = fastest, 5 = thorough)", "int", advanced=True),
        F("input.model.cpu_threads", "CPU threads (0 = auto)", "int", advanced=True),
        F("input.model.language", "Language code", "str", advanced=True),
        F("input.audio.device", "Microphone", "device_in", advanced=True),
        F("input.audio.sample_rate", "Sample rate", "int", advanced=True),
        F("input.audio.block_size", "Audio block size", "int", advanced=True),
        F("input.live.min_speech_duration_ms", "Minimum speech length (ms)", "int", advanced=True),
        F("input.live.max_speech_duration_s", "Maximum utterance length (s)", "float", advanced=True),
        F("input.hotkey.backend", "Hotkey backend", "choice", advanced=True, choices=("auto", "evdev", "pynput")),
        F("input.hotkey.duck_amount", "Recording duck amount (%)", "int", advanced=True),
        F("input.inject.method", "Insert text via", "choice", advanced=True, choices=("clipboard", "xdotool"),
          labels={"clipboard": "clipboard (paste with Ctrl+V)", "xdotool": "xdotool (simulated typing)"}),
        F("input.inject.xdotool_delay", "Typing delay per key (ms)", "int", advanced=True),
        F("input.postprocess.capitalize_first", "Capitalise first letter", "bool", advanced=True),
        F("input.postprocess.strip_leading_space", "Strip leading space", "bool", advanced=True),
        F("input.postprocess.remove_filler_words", "Remove filler words (um, uh…)", "bool", advanced=True),
        F("input.postprocess.remove_hallucinations", "Drop known Whisper hallucinations", "bool", advanced=True),
        F("input.vad.enabled", "Use Whisper's VAD filter", "bool", advanced=True),
        F("input.vad.threshold", "Speech probability threshold", "float", advanced=True),
        F("input.vad.min_silence_duration_ms", "VAD min silence (ms)", "int", advanced=True),
        F("input.vad.speech_pad_ms", "Speech padding (ms)", "int", advanced=True),
    ]


#: Speech-server fields are rendered by the Speech tab's header row, not the form body.
SERVER_FIELDS: list[FieldSpec] = [
    FieldSpec("output.server.enabled", "Enable speech", "bool"),
    FieldSpec("output.server.host", "Host", "str"),
    FieldSpec("output.server.port", "Port", "int"),
]


def speech_fields() -> list[FieldSpec]:
    """Output side. Basic fields first, then the Advanced block, each in display order."""
    F = FieldSpec
    return [
        F("output.speech.speed", "Speed", "float"),
        F("output.speech.volume", "Volume (0–100)", "int"),
        F("output.speech.device", "Speaker", "device_out"),
        F("output.speech.duck", "Lower other apps while speaking", "bool"),
        # ── Advanced ──
        F("output.speech.duck_amount", "Speaking duck amount (%)", "int", advanced=True),
        F("output.speech.pause_input", "Pause dictation while speaking", "bool", advanced=True),
        F("output.speech.pause_input_tail_ms", "Keep paused after speech ends (ms)", "int", advanced=True),
        F("output.speech.model_path", "Manual model path (bypasses downloads)", "str", advanced=True),
        F("output.speech.auto_download", "Download voices automatically", "bool", advanced=True),
        F("log_level", "Log level", "choice", advanced=True, choices=("DEBUG", "INFO", "WARNING", "ERROR")),
    ]


def all_fields() -> list[FieldSpec]:
    return dictation_fields() + SERVER_FIELDS + speech_fields()


def config_leaves(obj: object = None, prefix: str = "") -> list[str]:
    """Dotted paths of every scalar field in VocalConfig (for the drift guard)."""
    obj = VocalConfig() if obj is None else obj
    out: list[str] = []
    for f in fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, f.name)
        if is_dataclass(value):
            out += config_leaves(value, f"{prefix}{f.name}.")
        else:
            out.append(f"{prefix}{f.name}")
    return out


def get_path(cfg: object, path: str) -> Any:
    for part in path.split("."):
        cfg = getattr(cfg, part)
    return cfg


def set_path(cfg: object, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        cfg = getattr(cfg, part)
    setattr(cfg, parts[-1], value)


def make_field_widget(parent: tk.Misc, spec: FieldSpec) -> tuple[tk.Widget, tk.Variable]:
    """Widget + Tk variable for a spec. Shared by the form body and custom headers."""
    if spec.kind == "bool":
        var: tk.Variable = tk.BooleanVar()
        w: tk.Widget = ttk.Checkbutton(parent, variable=var, text=spec.label if spec in SERVER_FIELDS else "")
    elif spec.kind == "choice":
        var = tk.StringVar()
        labels = spec.labels or {}
        w = ttk.Combobox(parent, textvariable=var, state="readonly",
                         values=[labels.get(c, c) for c in (spec.choices or ())])
    elif spec.kind in ("device_in", "device_out"):
        var = tk.StringVar()
        w = ttk.Combobox(parent, textvariable=var, values=[DEFAULT_DEVICE], state="readonly")
    else:  # int / float / str
        var = tk.StringVar()
        w = ttk.Entry(parent, textvariable=var)
    return w, var


class Collapsible(ttk.Frame):
    """A header button that shows/hides a body frame."""

    def __init__(self, master: tk.Misc, title: str) -> None:
        super().__init__(master)
        self._title = title
        self.open = tk.BooleanVar(value=False)
        self._button = ttk.Button(self, command=self.toggle, style="Toolbutton")
        self._button.pack(anchor="w")
        self.body = ttk.Frame(self, padding=(18, 4, 0, 4))
        self.body.columnconfigure(1, weight=1)
        self._render()

    def set_open(self, value: bool) -> None:
        self.open.set(value)
        self._render()

    def toggle(self) -> None:
        self.set_open(not self.open.get())

    def _render(self) -> None:
        is_open = self.open.get()
        self._button.configure(text=f"{'▾' if is_open else '▸'} {self._title}")
        if is_open:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()


class SettingsForm(ttk.Frame):
    """Scrollable: [header] → Settings → ▸ Advanced, with Save & Apply / Revert.

    ``header`` optionally builds widgets at the top of the scrollable area
    (grids, the server row). It receives the parent frame and this form so it
    can register extra fields via :meth:`add_field`.
    """

    def __init__(
        self,
        master: tk.Misc,
        window: "VocalWindow",
        specs: list[FieldSpec],
        header: Callable[[ttk.Frame, "SettingsForm"], None] | None = None,
    ) -> None:
        super().__init__(master, padding=8)
        self.window = window
        self.app = window.app
        self.fields: list[FieldSpec] = []
        self._vars: dict[str, tk.Variable] = {}
        self._widgets: dict[str, tk.Widget] = {}

        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(outer, highlightthickness=0)
        scroll = ttk.Scrollbar(outer, orient="vertical", command=self._canvas.yview)
        self._form = ttk.Frame(self._canvas)
        self._form.bind("<Configure>", lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._form_id = self._canvas.create_window((0, 0), window=self._form, anchor="nw")
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(self._form_id, width=e.width))
        self._canvas.configure(yscrollcommand=scroll.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for seq, step in (("<Button-4>", -3), ("<Button-5>", 3)):
            self._canvas.bind(seq, lambda e, s=step: self._canvas.yview_scroll(s, "units"))
            self._form.bind(seq, lambda e, s=step: self._canvas.yview_scroll(s, "units"))

        if header is not None:
            header(self._form, self)

        self.settings_box = ttk.LabelFrame(self._form, text="Settings", padding=(10, 6))
        self.settings_box.pack(fill="x", padx=4, pady=4)
        self.settings_box.columnconfigure(1, weight=1)
        self.advanced = Collapsible(self._form, "Advanced")
        self.advanced.pack(fill="x", padx=4, pady=(2, 6))
        for spec in specs:
            parent = self.advanced.body if spec.advanced else self.settings_box
            self._add_row(parent, spec)

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(8, 0))
        self._save_btn = ttk.Button(bar, text="Save & Apply", command=self.save_and_apply)
        self._save_btn.pack(side="left")
        ttk.Button(bar, text="Revert", command=self.reload_from_app).pack(side="left", padx=6)
        self._status = ttk.Label(bar, text="", foreground="#666")
        self._status.pack(side="right")

        self.reload_from_app()

    # ── Field registration ───────────────────────────────────────────

    def add_field(self, spec: FieldSpec, widget: tk.Widget, var: tk.Variable) -> None:
        """Register a field whose widget lives outside the Settings/Advanced blocks."""
        self.fields.append(spec)
        self._vars[spec.path] = var
        self._widgets[spec.path] = widget

    def _add_row(self, parent: tk.Widget, spec: FieldSpec) -> None:
        row = max((int(w.grid_info()["row"]) for w in parent.grid_slaves()), default=-1) + 1
        widget, var = make_field_widget(parent, spec)
        ttk.Label(parent, text=spec.label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
        widget.grid(row=row, column=1, sticky="ew", pady=2)
        if spec.help:
            ttk.Label(parent, text=spec.help, foreground="#777", wraplength=600, justify="left").grid(
                row=row + 1, column=0, columnspan=2, sticky="w", padx=(18, 0), pady=(0, 6))
        self.add_field(spec, widget, var)

    def set_advanced(self, show: bool) -> None:
        self.advanced.set_open(show)

    # ── Load / collect ───────────────────────────────────────────────

    def reload_from_app(self) -> None:
        """Populate the form from the live config (discarding edits)."""
        self._refresh_devices()
        self.load_from(self.app.config)
        self._status.configure(text="")

    def load_from(self, cfg: VocalConfig) -> None:
        for spec in self.fields:
            value = get_path(cfg, spec.path)
            var = self._vars[spec.path]
            if spec.kind == "bool":
                var.set(bool(value))
            elif spec.kind == "choice":
                var.set((spec.labels or {}).get(value, str(value)))
            elif spec.kind in ("device_in", "device_out"):
                var.set(DEFAULT_DEVICE if value in (None, "") else str(value))
            else:
                var.set("" if value is None else str(value))

    def collect(self) -> VocalConfig:
        """Live config plus this form's values. Raises ValueError naming the field on bad input."""
        cfg = copy.deepcopy(self.app.config)
        for spec in self.fields:
            raw = self._vars[spec.path].get()
            try:
                value = self._parse(spec, raw)
            except ValueError:
                raise ValueError(f"{spec.label}: {raw!r} is not a valid {spec.kind}") from None
            set_path(cfg, spec.path, value)
        return cfg

    @staticmethod
    def _parse(spec: FieldSpec, raw: Any) -> Any:
        if spec.kind == "bool":
            return bool(raw)
        if spec.kind == "choice":
            reverse = {v: k for k, v in (spec.labels or {}).items()}
            return reverse.get(raw, raw)
        if spec.kind in ("device_in", "device_out"):
            return None if raw in ("", DEFAULT_DEVICE) else str(raw)
        text = str(raw).strip()
        if spec.kind == "int":
            return int(text)
        if spec.kind == "float":
            return float(text)
        if spec.path == "output.speech.model_path":
            return text or None
        return text

    def _refresh_devices(self) -> None:
        try:
            if "input.audio.device" in self._widgets:
                from vocal.input.audio import list_input_devices
                self._widgets["input.audio.device"].configure(  # type: ignore[call-arg]
                    values=[DEFAULT_DEVICE] + [name for _, name, _ in list_input_devices()])
            if "output.speech.device" in self._widgets:
                from vocal.output.playback import list_output_devices
                self._widgets["output.speech.device"].configure(  # type: ignore[call-arg]
                    values=[DEFAULT_DEVICE] + [name for _, name, _ in list_output_devices()])
        except Exception:  # pragma: no cover - no audio subsystem
            pass

    # ── Actions ──────────────────────────────────────────────────────

    def set_busy(self, busy: bool) -> None:
        self._save_btn.configure(state="disabled" if busy else "normal")
        if busy:
            self._status.configure(text="Applying…")

    def set_status(self, text: str, error: bool = False) -> None:
        self._status.configure(text=text, foreground="#b00020" if error else "#666")

    def save_and_apply(self) -> None:
        try:
            cfg = self.collect()
        except ValueError as e:
            self.set_status(str(e), error=True)
            return
        self.set_busy(True)
        self.set_status("Applying…")

        def done(notes: object) -> None:
            self.set_busy(False)
            self.window.after_apply()  # reloads forms (clears status) — so set the note after
            self.set_status(" · ".join(notes) if isinstance(notes, list) else "Applied")

        def failed(e: BaseException) -> None:
            self.set_busy(False)
            self.set_status(f"Failed: {e}", error=True)

        self.window.run_bg(lambda: self.app.apply_config(cfg), on_done=done, on_error=failed, name="apply-config")
