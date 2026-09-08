"""Speech tab: text-to-speech voice grid + every output-side setting."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from vocal.gui.settings_form import SERVER_FIELDS, SettingsForm, make_field_widget, speech_fields

if TYPE_CHECKING:
    from vocal.gui.window import VocalWindow

SAMPLE_TEXT = "Hello from Vocal. This is how this voice sounds."


class SpeechTab(ttk.Frame):
    def __init__(self, master: tk.Misc, window: "VocalWindow") -> None:
        super().__init__(master)
        self.window = window
        self.app = window.app
        self.form = SettingsForm(self, window, speech_fields(), header=self._build_header)
        self.form.pack(fill="both", expand=True)
        self.refresh()

    # ── Grid ─────────────────────────────────────────────────────────

    def _build_header(self, parent: ttk.Frame, form: SettingsForm) -> None:
        # Server row: [x] Enable speech   Host [____]  Port [____]
        row = ttk.Frame(parent, padding=(4, 6))
        row.pack(fill="x", padx=4)
        for spec in SERVER_FIELDS:
            widget, var = make_field_widget(row, spec)
            if spec.kind != "bool":
                ttk.Label(row, text=spec.label).pack(side="left", padx=(16, 4))
                widget.configure(width=16 if spec.path.endswith("host") else 7)  # type: ignore[call-arg]
            widget.pack(side="left")
            form.add_field(spec, widget, var)

        box = ttk.LabelFrame(parent, text="Voices", padding=(10, 6))
        box.pack(fill="x", padx=4, pady=(4, 8))
        cols = ("downloaded", "current", "description")
        self._voices = ttk.Treeview(box, columns=cols, show="tree headings", height=7, selectmode="browse")
        self._voices.heading("#0", text="Voice")
        self._voices.column("#0", width=210, stretch=False)
        for c, w in zip(cols, (90, 70, 370)):
            self._voices.heading(c, text=c.capitalize())
            self._voices.column(c, width=w, anchor="center" if c in ("downloaded", "current") else "w",
                                stretch=(c == "description"))
        self._voices.pack(fill="x")

        bar = ttk.Frame(box)
        bar.pack(fill="x", pady=(6, 0))
        self._dl_btn = ttk.Button(bar, text="Download", command=self._download)
        self._dl_btn.pack(side="left")
        ttk.Button(bar, text="Remove", command=self._remove).pack(side="left", padx=4)
        ttk.Button(bar, text="Use this voice", command=self._use_voice).pack(side="left", padx=4)
        ttk.Button(bar, text="Test", command=self._test).pack(side="left", padx=4)
        self._voice_status = ttk.Label(bar, text="", foreground="#666")
        self._voice_status.pack(side="left", padx=12)

    def refresh(self) -> None:
        from vocal.output.models import VOICES, is_downloaded

        selected = self._selected()
        self._voices.delete(*self._voices.get_children())
        current = self.app.config.output.speech.voice
        for name, spec in VOICES.items():
            self._voices.insert("", "end", iid=name, text=name, values=(
                "✓" if is_downloaded(spec) else "", "✓" if name == current else "", spec.description,
            ))
        if selected and self._voices.exists(selected):
            self._voices.selection_set(selected)

    def _selected(self) -> str | None:
        sel = self._voices.selection()
        return sel[0] if sel else None

    def _set_status(self, text: str, error: bool = False) -> None:
        self._voice_status.configure(text=text, foreground="#b00020" if error else "#666")

    # ── Actions ──────────────────────────────────────────────────────

    def _download(self) -> None:
        name = self._selected()
        if not name:
            return self._set_status("Select a voice first")
        from vocal.output.models import download_voice

        self._dl_btn.configure(state="disabled")
        self._set_status(f"Downloading {name}…")

        def progress(msg: str) -> None:
            self.window.call_soon(lambda: self._set_status(msg))

        def done(_r: object) -> None:
            self._dl_btn.configure(state="normal")
            self._set_status(f"{name} ready")
            self.refresh()

        def failed(e: BaseException) -> None:
            self._dl_btn.configure(state="normal")
            self._set_status(f"Download failed: {e}", error=True)
            self.refresh()

        self.window.run_bg(lambda: download_voice(name, progress=progress), on_done=done, on_error=failed,
                           name="voice-download")

    def _remove(self) -> None:
        name = self._selected()
        if not name:
            return self._set_status("Select a voice first")
        from vocal.output.models import remove_voice

        def done(removed: object) -> None:
            self._set_status(f"Removed {name}" if removed else "Nothing to remove")
            self.refresh()

        self.window.run_bg(lambda: remove_voice(name), on_done=done,
                           on_error=lambda e: self._set_status(str(e), error=True), name="voice-remove")

    def _use_voice(self) -> None:
        name = self._selected()
        if not name:
            return self._set_status("Select a voice first")
        if name == self.app.config.output.speech.voice:
            return
        self._set_status(f"Switching to {name}…")

        def done(_notes: object) -> None:
            self._set_status(f"Using {name}")
            self.window.after_apply()

        self.window.run_bg(lambda: self.app.set_voice(name), on_done=done,
                           on_error=lambda e: self._set_status(str(e), error=True), name="set-voice")

    def _test(self) -> None:
        name = self._selected()
        if not name:
            return self._set_status("Select a voice first")
        self._set_status(f"Speaking with {name}…")
        self.window.run_bg(lambda: self.app.say(SAMPLE_TEXT, interrupt=True, voice=name), name="voice-test")
