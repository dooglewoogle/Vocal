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
from vocal.gui.tooltip import tip

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
          labels={"live": "live (always listening)", "hotkey": "hotkey (hold to record)"},
          help="Live keeps the microphone open and uses voice detection to find sentence boundaries. "
               "Hotkey records only while the key is held. Changing this restarts dictation."),
        F("input.hotkey.key", "Hotkey", "str",
          help="Key name as evdev reports it, without the KEY_ prefix: PAUSE, F18, SCROLLLOCK, END… "
               "In hotkey mode, hold it to record. In live mode, hold it to mute the microphone."),
        F("input.live.min_silence_duration_ms", "End of sentence after silence (ms)", "int",
          help="Live mode: a pause this long ends the utterance and sends it for transcription. "
               "Lower is snappier but splits sentences at hesitations."),
        F("input.hotkey.duck", "Lower system volume while recording", "bool",
          help="Hotkey mode only: turns the system output volume down while you hold the key so playback "
               "does not bleed into the microphone, then ramps it back over ~300 ms."),
        # ── Advanced ──
        F("input.model.compute_type", "Compute type", "choice", advanced=True, choices=("int8", "float32"),
          help="int8 is the fast, quantised default for CPUs. float32 is slower and uses about four times "
               "the memory, with a small accuracy gain on some machines."),
        F("input.model.beam_size", "Beam size", "int", advanced=True,
          help="How many decoding hypotheses Whisper keeps in parallel. 1 is greedy and fastest; "
               "5 is noticeably slower and slightly more accurate. 3 is a good middle."),
        F("input.model.cpu_threads", "CPU threads", "int", advanced=True,
          help="Threads used for Whisper inference. 0 lets the runtime choose, which is usually right."),
        F("input.model.language", "Language code", "str", advanced=True,
          help="ISO 639-1 code such as en, de, fr. Only matters with a multilingual model (no .en suffix); "
               "English-only models ignore it."),
        F("input.audio.device", "Microphone", "device_in", advanced=True,
          help="Which input device to capture. The list comes from PortAudio; the system default follows "
               "your desktop's sound settings, so it usually needs no change."),
        F("input.audio.sample_rate", "Sample rate (Hz)", "int", advanced=True,
          help="Capture rate. Whisper expects 16000; change only if your device cannot deliver it."),
        F("input.audio.block_size", "Audio block size (frames)", "int", advanced=True,
          help="Frames per audio callback in hotkey mode. Larger blocks add latency, smaller ones cost CPU."),
        F("input.live.min_speech_duration_ms", "Minimum speech length (ms)", "int", advanced=True,
          help="Live mode: sounds shorter than this are treated as noise and never transcribed."),
        F("input.live.max_speech_duration_s", "Maximum utterance length (s)", "float", advanced=True,
          help="Live mode: an utterance is cut and transcribed once it reaches this length, even without a pause."),
        F("input.hotkey.backend", "Hotkey backend", "choice", advanced=True, choices=("auto", "evdev", "pynput"),
          help="evdev reads /dev/input directly and works on X11 and Wayland (your user must be in the input "
               "group). pynput works on macOS, Windows and X11 only. auto picks evdev when it is available."),
        F("input.hotkey.duck_amount", "Recording duck amount (%)", "int", advanced=True,
          help="Relative cut applied to the current system volume while recording: at 50, a volume of 80% "
               "drops to 40%."),
        F("input.inject.method", "Insert text via", "choice", advanced=True, choices=("clipboard", "xdotool"),
          labels={"clipboard": "clipboard (paste with Ctrl+V)", "xdotool": "xdotool (simulated typing)"},
          help="clipboard copies the text, sends Ctrl+V, then restores what was on your clipboard. "
               "xdotool (wtype on Wayland) types the characters one by one: slower, but works in apps "
               "where paste does not."),
        F("input.inject.xdotool_delay", "Typing delay per key (ms)", "int", advanced=True,
          help="Pause between simulated keystrokes when inserting via xdotool/wtype. Raise it if characters "
               "get dropped or arrive out of order."),
        F("input.postprocess.capitalize_first", "Capitalise first letter", "bool", advanced=True,
          help="Upper-case the first letter of every transcription."),
        F("input.postprocess.strip_leading_space", "Strip leading space", "bool", advanced=True,
          help="Remove the space Whisper often puts before the first word."),
        F("input.postprocess.remove_filler_words", "Remove filler words", "bool", advanced=True,
          help="Delete um, uh, hmm and similar hesitations from the transcribed text."),
        F("input.postprocess.remove_hallucinations", "Drop known hallucinations", "bool", advanced=True,
          help="Discard phrases Whisper is known to invent from silence or noise, such as "
               "\u201cThank you for watching\u201d or \u201cSubtitles by\u2026\u201d."),
        F("input.vad.enabled", "Use Whisper's VAD filter", "bool", advanced=True,
          help="Run faster-whisper's built-in voice activity filter on each utterance so silent stretches "
               "are skipped before decoding. Cheap and usually worth keeping on."),
        F("input.vad.threshold", "Speech probability threshold", "float", advanced=True,
          help="Probability (0\u20131) above which a frame counts as speech, for both the live detector and the "
               "VAD filter. Raise it in noisy rooms; lower it if quiet speech is missed."),
        F("input.vad.min_silence_duration_ms", "VAD min silence (ms)", "int", advanced=True,
          help="Silence the VAD filter needs before it splits an utterance into separate segments."),
        F("input.vad.speech_pad_ms", "Speech padding (ms)", "int", advanced=True,
          help="Extra audio kept on both sides of detected speech so word edges are not clipped."),
    ]


#: Speech-server fields are rendered by the Speech tab's header row, not the form body.
SERVER_FIELDS: list[FieldSpec] = [
    FieldSpec("output.server.enabled", "Enable speech", "bool",
              help="Run the localhost HTTP server that lets `vocal say`, scripts and editor hooks make Vocal "
                   "talk. Off means only dictation runs and nothing can request speech."),
    FieldSpec("output.server.host", "Host", "str",
              help="Interface to listen on. Keep 127.0.0.1: the server has no authentication and relies on "
                   "being reachable from this machine only."),
    FieldSpec("output.server.port", "Port", "int",
              help="TCP port for the speech server. If it is already taken, Vocal picks a free one and writes "
                   "the actual port to the runtime file so clients still find it."),
]


def speech_fields() -> list[FieldSpec]:
    """Output side. Basic fields first, then the Advanced block, each in display order."""
    F = FieldSpec
    return [
        F("output.speech.speed", "Speed", "float",
          help="Speaking-rate multiplier. 1.0 is the voice's natural pace; 1.2 is brisk, 0.8 is slow."),
        F("output.speech.volume", "Volume (0\u2013100)", "int",
          help="Digital gain applied to the synthesized audio before playback. It does not change the "
               "system volume, so 100 is simply the voice as recorded."),
        F("output.speech.device", "Speaker", "device_out",
          help="Output device for speech. The system default follows your desktop's sound settings."),
        F("output.speech.duck", "Lower other apps while speaking", "bool",
          help="While speaking, turn down every other application's audio stream (PulseAudio or PipeWire, "
               "via pactl) so speech is not drowned out. Vocal's own output stays at full volume. Streams "
               "that start mid-sentence are not ducked."),
        # ── Advanced ──
        F("output.speech.duck_amount", "Speaking duck amount (%)", "int", advanced=True,
          help="Relative cut applied to other applications' streams while speaking: at 50, a stream at 80% "
               "drops to 40%."),
        F("output.speech.pause_input", "Pause dictation while speaking", "bool", advanced=True,
          help="Stop consuming the microphone from the first frame of speech until playback ends, so Vocal "
               "never transcribes its own voice. In hotkey mode, pressing the key cuts the speech instead."),
        F("output.speech.pause_input_tail_ms", "Keep paused after speech ends (ms)", "int", advanced=True,
          help="How long the microphone stays paused after the last audio, to cover room echo and any "
               "audio still buffered in the output device."),
        F("output.speech.model_path", "Manual model path", "str", advanced=True,
          help="Load a voice you downloaded yourself instead of the registry: the .onnx file for Piper (with "
               "its .onnx.json beside it), or the folder holding kokoro-*.onnx and voices-*.bin for Kokoro. "
               "The selected voice still decides the backend and, for Kokoro, the speaker. Leave empty to "
               "use the registry."),
        F("output.speech.auto_download", "Download voices automatically", "bool", advanced=True,
          help="Fetch a voice's files the first time it is used. Turn off to require an explicit Download "
               "from the grid above."),
        F("log_level", "Log level", "choice", advanced=True, choices=("DEBUG", "INFO", "WARNING", "ERROR"),
          help="Verbosity of the log file (~/.local/state/vocal/vocal.log) and of stderr when run from a "
               "terminal. DEBUG is noisy but records every audio, hotkey and HTTP event."),
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


def section(parent: tk.Misc, title: str) -> ttk.Frame:
    """A bold heading followed by a plain frame: sections are flat, not boxed,
    so the only borders on a tab are the pane's and the controls' own."""
    ttk.Label(parent, text=title, font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=4, pady=(8, 2))
    body = ttk.Frame(parent, padding=(10, 2, 4, 6))
    body.pack(fill="x", padx=4)
    return body


class Collapsible(ttk.Frame):
    """A header button that shows/hides a body frame."""

    def __init__(self, master: tk.Misc, title: str) -> None:
        super().__init__(master)
        self._title = title
        self.open = tk.BooleanVar(value=False)
        self._button = ttk.Button(self, command=self.toggle, style="Toolbutton")
        tip(self._button, "Settings most people never change. Hover any field for an explanation.").pack(anchor="w")
        self.body = ttk.Frame(self, padding=(10, 2, 4, 6))
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
        self._canvas.wheel_scrollable = True  # picked up by VocalWindow's window-wide wheel handler

        if header is not None:
            header(self._form, self)

        self.settings_box = section(self._form, "Settings")
        self.settings_box.columnconfigure(1, weight=1)
        self.advanced = Collapsible(self._form, "Advanced")
        self.advanced.pack(fill="x", padx=4, pady=(6, 6))
        for spec in specs:
            parent = self.advanced.body if spec.advanced else self.settings_box
            self._add_row(parent, spec)

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(8, 0))
        self._save_btn = ttk.Button(bar, text="Save & Apply", command=self.save_and_apply)
        tip(self._save_btn, "Write these values to config.toml and apply them now. Dictation changes restart "
                            "the engine, which takes a few seconds; speech changes apply immediately.").pack(side="left")
        tip(ttk.Button(bar, text="Revert", command=self.reload_from_app),
            "Discard your edits and reload the values Vocal is currently running with.").pack(side="left", padx=6)
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
        tip(ttk.Label(parent, text=spec.label), spec.help).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
        tip(widget, spec.help).grid(row=row, column=1, sticky="ew", pady=2)
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
