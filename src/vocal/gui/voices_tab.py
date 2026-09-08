"""Voices & Models tab: TTS voice registry and Whisper model picker."""

from __future__ import annotations

import copy
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vocal.gui.window import VocalWindow

SAMPLE_TEXT = "Hello from Vocal. This is how this voice sounds."


def whisper_model_cached(size: str) -> bool | None:
    """True/False if we can tell whether ``size`` is in the Hugging Face cache; None if unknown."""
    try:
        from faster_whisper.utils import _MODELS
        from huggingface_hub import try_to_load_from_cache

        repo = _MODELS.get(size)
        if repo is None:
            return None
        return isinstance(try_to_load_from_cache(repo, "model.bin"), str)
    except Exception:
        return None


class VoicesTab(ttk.Frame):
    def __init__(self, master: tk.Misc, window: "VocalWindow") -> None:
        super().__init__(master, padding=8)
        self.window = window
        self.app = window.app

        # ── Voices ──
        ttk.Label(self, text="Text-to-speech voices", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        cols = ("backend", "downloaded", "default", "description")
        self._voices = ttk.Treeview(self, columns=cols, show="tree headings", height=8, selectmode="browse")
        self._voices.heading("#0", text="Voice")
        self._voices.column("#0", width=210, stretch=False)
        for c, w in zip(cols, (70, 90, 70, 300)):
            self._voices.heading(c, text=c.capitalize())
            self._voices.column(c, width=w, anchor="center" if c in ("downloaded", "default") else "w",
                                stretch=(c == "description"))
        self._voices.pack(fill="x", pady=(4, 6))

        bar = ttk.Frame(self)
        bar.pack(fill="x")
        self._dl_btn = ttk.Button(bar, text="Download", command=self._download)
        self._dl_btn.pack(side="left")
        ttk.Button(bar, text="Remove", command=self._remove).pack(side="left", padx=4)
        ttk.Button(bar, text="Use as default", command=self._use_voice).pack(side="left", padx=4)
        ttk.Button(bar, text="Test", command=self._test).pack(side="left", padx=4)
        self._voice_status = ttk.Label(bar, text="", foreground="#666")
        self._voice_status.pack(side="left", padx=12)

        ttk.Separator(self).pack(fill="x", pady=12)

        # ── Whisper models ──
        ttk.Label(self, text="Whisper (dictation) models", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self._models = ttk.Treeview(self, columns=("cached", "current"), show="tree headings",
                                    height=8, selectmode="browse")
        self._models.heading("#0", text="Model")
        self._models.column("#0", width=200, stretch=False)
        self._models.heading("cached", text="Downloaded")
        self._models.column("cached", width=100, stretch=False, anchor="center")
        self._models.heading("current", text="In use")
        self._models.column("current", width=80, stretch=True, anchor="center")
        self._models.pack(fill="x", pady=(4, 6))
        mbar = ttk.Frame(self)
        mbar.pack(fill="x")
        ttk.Button(mbar, text="Use this model", command=self._use_model).pack(side="left")
        ttk.Label(mbar, text="Models download on first use (100 MB – 3 GB). Switching restarts dictation.",
                  foreground="#666").pack(side="left", padx=12)

        self.refresh()

    # ── Population ───────────────────────────────────────────────────

    def refresh(self) -> None:
        from vocal.input.transcriber import VALID_MODELS
        from vocal.output.models import VOICES, is_downloaded

        selected = self._selected_voice()
        self._voices.delete(*self._voices.get_children())
        current_voice = self.app.config.output.speech.voice
        for name, spec in VOICES.items():
            mark = "✓" if is_downloaded(spec) else ""
            self._voices.insert("", "end", iid=name, text=name,
                                values=(spec.backend, mark, "✓" if name == current_voice else "", spec.description))
        if selected and self._voices.exists(selected):
            self._voices.selection_set(selected)

        selected_model = self._selected_model()
        self._models.delete(*self._models.get_children())
        current = self.app.config.input.model.size
        from vocal.gui.settings_tab import _whisper_models

        sizes = [m for m in _whisper_models() if m in VALID_MODELS]
        for size in sizes:
            self._models.insert("", "end", iid=size, text=size, values=("…", "✓" if size == current else ""))
        if selected_model and self._models.exists(selected_model):
            self._models.selection_set(selected_model)

        def probe() -> dict[str, bool | None]:
            return {s: whisper_model_cached(s) for s in sizes}

        def done(result: object) -> None:
            if not isinstance(result, dict):
                return
            for size, cached in result.items():
                if self._models.exists(size):
                    mark = "" if cached is None else ("✓" if cached else "")
                    self._models.set(size, "cached", mark)

        self.window.run_bg(probe, on_done=done, name="whisper-cache-probe")

    def _selected_voice(self) -> str | None:
        sel = self._voices.selection()
        return sel[0] if sel else None

    def _selected_model(self) -> str | None:
        sel = self._models.selection()
        return sel[0] if sel else None

    def _set_status(self, text: str, error: bool = False) -> None:
        self._voice_status.configure(text=text, foreground="#b00020" if error else "#666")

    # ── Voice actions ────────────────────────────────────────────────

    def _download(self) -> None:
        name = self._selected_voice()
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
        name = self._selected_voice()
        if not name:
            return self._set_status("Select a voice first")
        from vocal.output.models import remove_voice

        def done(removed: object) -> None:
            self._set_status(f"Removed {name}" if removed else "Nothing to remove")
            self.refresh()

        self.window.run_bg(lambda: remove_voice(name), on_done=done,
                           on_error=lambda e: self._set_status(str(e), error=True), name="voice-remove")

    def _use_voice(self) -> None:
        name = self._selected_voice()
        if not name:
            return self._set_status("Select a voice first")
        self._set_status(f"Switching to {name}…")

        def done(notes: object) -> None:
            self._set_status(f"Default voice is now {name}")
            self.refresh()
            self.window.settings.reload_from_app()
            self.window.status.refresh_summary()

        self.window.run_bg(lambda: self.app.set_voice(name), on_done=done,
                           on_error=lambda e: self._set_status(str(e), error=True), name="set-voice")

    def _test(self) -> None:
        name = self._selected_voice()
        if not name:
            return self._set_status("Select a voice first")
        self._set_status(f"Speaking with {name}…")
        self.window.run_bg(lambda: self.app.say(SAMPLE_TEXT, interrupt=True, voice=name), name="voice-test")

    # ── Model actions ────────────────────────────────────────────────

    def _use_model(self) -> None:
        size = self._selected_model()
        if not size or size == self.app.config.input.model.size:
            return
        cfg = copy.deepcopy(self.app.config)
        cfg.input.model.size = size

        def done(_notes: object) -> None:
            self.refresh()
            self.window.settings.reload_from_app()
            self.window.status.refresh_summary()

        self.window.run_bg(lambda: self.app.apply_config(cfg), on_done=done,
                           on_error=lambda e: self._set_status(str(e), error=True), name="set-model")
