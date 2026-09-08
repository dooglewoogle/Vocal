"""Reusable, data-driven settings form over VocalConfig.

Each tab embeds one :class:`SettingsForm` with the subset of :class:`FieldSpec`
it owns. Every config leaf is either a field on exactly one form or listed in
:data:`OWNED_ELSEWHERE` (chosen through a grid or another tab); a test guards
this. The form edits a deep copy; "Save & Apply" hands that copy to
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
    section: str
    advanced: bool = False
    choices: tuple[str, ...] | None = None
    labels: Mapping[str, str] | None = None  # choice value → display text
    help: str | None = None


def dictation_fields() -> list[FieldSpec]:
    """Input side, in display order. The Whisper model grid sits above these."""
    F = FieldSpec
    return [
        # ── Whisper model (next to the grid) ──
        F("input.model.compute_type", "Compute type", "choice", "Whisper model", choices=("int8", "float32")),
        F("input.model.beam_size", "Beam size (1 = fastest, 5 = thorough)", "int", "Whisper model"),
        F("input.model.cpu_threads", "CPU threads (0 = auto)", "int", "Whisper model", advanced=True),
        F("input.model.language", "Language code", "str", "Whisper model", advanced=True),
        # ── Dictation ──
        F("input.engine", "Dictation mode", "choice", "Dictation", choices=ENGINES,
          labels={"live": "live (always listening)", "hotkey": "hotkey (hold to record)"}),
        F("input.audio.device", "Microphone", "device_in", "Dictation"),
        F("input.live.min_silence_duration_ms", "End of sentence after silence (ms)", "int", "Dictation"),
        F("input.audio.sample_rate", "Sample rate", "int", "Dictation", advanced=True),
        F("input.audio.block_size", "Audio block size", "int", "Dictation", advanced=True),
        F("input.live.min_speech_duration_ms", "Minimum speech length (ms)", "int", "Dictation", advanced=True),
        F("input.live.max_speech_duration_s", "Maximum utterance length (s)", "float", "Dictation", advanced=True),
        # ── Hotkey ──
        F("input.hotkey.key", "Hotkey (hold to record; hold to mute in live mode)", "str", "Hotkey",
          help="e.g. PAUSE, F18, SCROLLLOCK, END"),
        F("input.hotkey.backend", "Hotkey backend", "choice", "Hotkey", advanced=True,
          choices=("auto", "evdev", "pynput")),
        # ── Ducking ──
        F("input.hotkey.duck", "Lower system volume while recording", "bool", "Ducking"),
        F("input.hotkey.duck_amount", "Amount (%)", "int", "Ducking"),
        # ── Text output ──
        F("input.inject.method", "Insert text via", "choice", "Text output", choices=("clipboard", "xdotool"),
          labels={"clipboard": "clipboard (paste with Ctrl+V)", "xdotool": "xdotool (simulated typing)"}),
        F("input.inject.xdotool_delay", "Typing delay per key (ms)", "int", "Text output", advanced=True),
        F("input.postprocess.capitalize_first", "Capitalise first letter", "bool", "Text output", advanced=True),
        F("input.postprocess.strip_leading_space", "Strip leading space", "bool", "Text output", advanced=True),
        F("input.postprocess.remove_filler_words", "Remove filler words (um, uh…)", "bool", "Text output", advanced=True),
        F("input.postprocess.remove_hallucinations", "Drop known Whisper hallucinations", "bool", "Text output", advanced=True),
        # ── Voice detection (advanced) ──
        F("input.vad.enabled", "Use Whisper's VAD filter", "bool", "Voice detection", advanced=True),
        F("input.vad.threshold", "Speech probability threshold", "float", "Voice detection", advanced=True),
        F("input.vad.min_silence_duration_ms", "VAD min silence (ms)", "int", "Voice detection", advanced=True),
        F("input.vad.speech_pad_ms", "Speech padding (ms)", "int", "Voice detection", advanced=True),
    ]


def speech_fields() -> list[FieldSpec]:
    """Output side, in display order. The voice grid sits above these."""
    F = FieldSpec
    return [
        # ── Voice model (next to the grid) ──
        F("output.speech.backend", "Backend (used with a manual model path)", "choice", "Voice model",
          advanced=True, choices=("piper", "kokoro", "system")),
        F("output.speech.model_path", "Manual model path (bypasses downloads)", "str", "Voice model", advanced=True),
        F("output.speech.auto_download", "Download voices automatically", "bool", "Voice model", advanced=True),
        # ── Speech ──
        F("output.speech.speed", "Speed", "float", "Speech"),
        F("output.speech.volume", "Volume (0–100)", "int", "Speech"),
        F("output.speech.device", "Speaker", "device_out", "Speech"),
        F("output.speech.pause_input", "Pause dictation while speaking", "bool", "Speech", advanced=True),
        F("output.speech.pause_input_tail_ms", "Keep paused after speech ends (ms)", "int", "Speech", advanced=True),
        # ── Ducking ──
        F("output.speech.duck", "Lower other apps while speaking", "bool", "Ducking"),
        F("output.speech.duck_amount", "Amount (%)", "int", "Ducking"),
        # ── Server ──
        F("output.server.enabled", "Enable localhost speech server", "bool", "Server", advanced=True),
        F("output.server.host", "Host", "str", "Server", advanced=True),
        F("output.server.port", "Port", "int", "Server", advanced=True),
        # ── General ──
        F("log_level", "Log level", "choice", "General", advanced=True,
          choices=("DEBUG", "INFO", "WARNING", "ERROR")),
    ]


def all_fields() -> list[FieldSpec]:
    return dictation_fields() + speech_fields()


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


class SettingsForm(ttk.Frame):
    """Scrollable form for ``specs`` with Save & Apply / Revert / Show advanced.

    ``above`` optionally builds a widget (e.g. a model grid) placed at the top
    of the scrollable area so the whole tab scrolls as one.
    """

    def __init__(
        self,
        master: tk.Misc,
        window: "VocalWindow",
        specs: list[FieldSpec],
        above: Callable[[ttk.Frame], tk.Widget] | None = None,
    ) -> None:
        super().__init__(master, padding=8)
        self.window = window
        self.app = window.app
        self.fields = specs
        self._vars: dict[str, tk.Variable] = {}
        self._widgets: dict[str, tk.Widget] = {}
        self._advanced: list[tk.Widget] = []  # rows (label/widget/tip) shown only in advanced mode
        self._sections: list[tuple[ttk.LabelFrame, bool]] = []  # (frame, all_advanced) in display order
        self._show_advanced = tk.BooleanVar(value=False)

        overridden = sorted(p for p in self.app.cli_overridden if p in {s.path for s in specs})
        if overridden:
            ttk.Label(
                self, wraplength=700, foreground="#a15c00",
                text=f"Values for {', '.join(overridden)} came from command-line flags. They are shown "
                     "here and will be written to the config file if you save.",
            ).pack(fill="x", pady=(0, 6))

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

        if above is not None:
            above(self._form).pack(fill="x", padx=4, pady=(4, 8))
        self._build_form()

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(8, 0))
        self._save_btn = ttk.Button(bar, text="Save & Apply", command=self.save_and_apply)
        self._save_btn.pack(side="left")
        ttk.Button(bar, text="Revert", command=self.reload_from_app).pack(side="left", padx=6)
        ttk.Checkbutton(bar, text="Show advanced", variable=self._show_advanced,
                        command=self._toggle_advanced).pack(side="left", padx=12)
        self._status = ttk.Label(bar, text="", foreground="#666")
        self._status.pack(side="right")

        self.reload_from_app()
        self._toggle_advanced()

    # ── Form construction ────────────────────────────────────────────

    def _build_form(self) -> None:
        sections: dict[str, ttk.LabelFrame] = {}
        for spec in self.fields:
            frame = sections.get(spec.section)
            if frame is None:
                frame = ttk.LabelFrame(self._form, text=spec.section, padding=(10, 6))
                frame.columnconfigure(1, weight=1)
                sections[spec.section] = frame
                self._sections.append((frame, spec.advanced))
            elif not spec.advanced:
                # A basic field makes the whole section visible in basic mode.
                self._sections = [(f, adv and f is not frame) for f, adv in self._sections]
            row = max((int(w.grid_info()["row"]) for w in frame.grid_slaves()), default=-1) + 1
            label = ttk.Label(frame, text=spec.label)
            widget = self._make_widget(frame, spec)
            label.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            if spec.advanced:
                self._advanced += [label, widget]
            if spec.help:
                tip = ttk.Label(frame, text=spec.help, foreground="#777", wraplength=600, justify="left")
                tip.grid(row=row + 1, column=0, columnspan=2, sticky="w", padx=(18, 0), pady=(0, 6))
                if spec.advanced:
                    self._advanced.append(tip)
            self._widgets[spec.path] = widget

    def _make_widget(self, parent: tk.Misc, spec: FieldSpec) -> tk.Widget:
        if spec.kind == "bool":
            var: tk.Variable = tk.BooleanVar()
            w: tk.Widget = ttk.Checkbutton(parent, variable=var)
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
        self._vars[spec.path] = var
        return w

    def _toggle_advanced(self) -> None:
        show = self._show_advanced.get()
        for w in self._advanced:
            if show:
                w.grid()
            else:
                w.grid_remove()
        # Re-pack sections in declaration order so a hidden section returns to
        # its place (pack_forget + pack would append it at the bottom).
        for frame, _ in self._sections:
            frame.pack_forget()
        for frame, all_advanced in self._sections:
            if show or not all_advanced:
                frame.pack(fill="x", padx=4, pady=4)

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
        paths = {s.path for s in self.fields}
        try:
            if "input.audio.device" in paths:
                from vocal.input.audio import list_input_devices
                self._widgets["input.audio.device"].configure(  # type: ignore[call-arg]
                    values=[DEFAULT_DEVICE] + [name for _, name, _ in list_input_devices()])
            if "output.speech.device" in paths:
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
