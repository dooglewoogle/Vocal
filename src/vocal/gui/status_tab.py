"""Status tab: current state, quick actions, transcript log."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from vocal.gui.tooltip import tip
from vocal.state import DictationState

if TYPE_CHECKING:
    from vocal.gui.window import VocalWindow

_LABEL = {
    DictationState.LOADING: "Loading model…",
    DictationState.SLEEPING: "Paused",
    DictationState.LISTENING: "Listening",
    DictationState.RECORDING: "Recording…",
    DictationState.TRANSCRIBING: "Transcribing…",
}
_COLOUR = {
    DictationState.LOADING: "#d59a2b",
    DictationState.SLEEPING: "#8a8f98",
    DictationState.LISTENING: "#3f9d5a",
    DictationState.RECORDING: "#c9452e",
    DictationState.TRANSCRIBING: "#d59a2b",
}
_MAX_LINES = 200


class StatusTab(ttk.Frame):
    def __init__(self, master: tk.Misc, window: "VocalWindow") -> None:
        super().__init__(master, padding=12)
        self.window = window
        self.app = window.app
        self._state = DictationState.LISTENING
        self._speaking = False
        self._rebuilding = False

        top = ttk.Frame(self)
        top.pack(fill="x")
        self._dot = tk.Canvas(top, width=18, height=18, highlightthickness=0)
        self._dot.pack(side="left", padx=(0, 8))
        self._circle = self._dot.create_oval(2, 2, 16, 16, fill=_COLOUR[self._state], outline="")
        self._state_label = ttk.Label(top, text="Listening", font=("TkDefaultFont", 16, "bold"))
        self._state_label.pack(side="left")
        self._speaking_label = ttk.Label(top, text="", foreground="#d59a2b")
        self._speaking_label.pack(side="left", padx=12)

        self._summary = ttk.Label(self, text="", foreground="#666")
        self._summary.pack(fill="x", pady=(4, 10))

        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=(0, 10))
        self._pause_btn = ttk.Button(actions, text="Pause", command=self._toggle_pause)
        tip(self._pause_btn, "Pause or resume live listening. Only applies in live mode; in hotkey mode "
                             "nothing is recorded until you hold the key anyway.").pack(side="left")
        self._stop_btn = ttk.Button(actions, text="Stop speaking", command=self._stop_speaking, state="disabled")
        tip(self._stop_btn, "Stop the current utterance and clear everything queued for speech.").pack(side="left", padx=6)
        tip(ttk.Button(actions, text="Clear log", command=self.clear),
            "Clear this transcript list. Nothing on disk is affected.").pack(side="right")

        ttk.Label(self, text="Transcripts").pack(anchor="w")
        box = ttk.Frame(self)
        box.pack(fill="both", expand=True)
        self._text = tk.Text(box, wrap="word", height=12, state="disabled", relief="flat")
        scroll = ttk.Scrollbar(box, command=self._text.yview)
        self._text.configure(yscrollcommand=scroll.set)
        self._text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._text.tag_configure("time", foreground="#888")

        self._hint = ttk.Label(self, text="", foreground="#666")
        self._hint.pack(fill="x", pady=(8, 0))

    # ── Updates (UI thread) ──────────────────────────────────────────

    def set_state(self, state: DictationState) -> None:
        self._state = state
        self._render()

    def set_speaking(self, speaking: bool) -> None:
        self._speaking = speaking
        self._render()

    def set_rebuilding(self, rebuilding: bool) -> None:
        self._rebuilding = rebuilding
        self._render()

    def set_close_hint(self, has_tray: bool) -> None:
        self._hint.configure(text=(
            "Closing this window keeps Vocal running in the tray."
            if has_tray else "No tray icon available: closing this window quits Vocal."
        ))

    def refresh_summary(self) -> None:
        cfg = self.app.config
        mode = "live (always listening)" if cfg.input.engine == "live" else (
            f"hotkey (hold {cfg.input.hotkey.key})"
        )
        self._summary.configure(
            text=f"Mode: {mode}   ·   Whisper: {cfg.input.model.size}   ·   Voice: {cfg.output.speech.voice}"
        )
        self._state = self.app.state
        self._render()

    def add_transcript(self, text: str) -> None:
        self._text.configure(state="normal")
        self._text.insert("end", time.strftime("%H:%M:%S  "), "time")
        self._text.insert("end", text + "\n")
        lines = int(self._text.index("end-1c").split(".")[0])
        if lines > _MAX_LINES:
            self._text.delete("1.0", f"{lines - _MAX_LINES + 1}.0")
        self._text.configure(state="disabled")
        self._text.see("end")

    def clear(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")

    # ── Internals ────────────────────────────────────────────────────

    def _render(self) -> None:
        if self._rebuilding:
            label, colour = "Restarting dictation…", _COLOUR[DictationState.TRANSCRIBING]
        else:
            label, colour = _LABEL[self._state], _COLOUR[self._state]
        self._state_label.configure(text=label)
        self._dot.itemconfigure(self._circle, fill=colour)
        self._speaking_label.configure(text="Speaking" if self._speaking else "")
        self._stop_btn.configure(state="normal" if self._speaking else "disabled")
        self._pause_btn.configure(
            text="Resume" if self._state == DictationState.SLEEPING else "Pause",
            state="disabled" if (self._rebuilding or self._state == DictationState.LOADING
                                 or self.app.config.input.engine != "live") else "normal",
        )

    def _toggle_pause(self) -> None:
        self.window.run_bg(self.app.toggle_pause, name="toggle-pause")

    def _stop_speaking(self) -> None:
        self.window.run_bg(self.app.stop_speaking, name="stop-speaking")
