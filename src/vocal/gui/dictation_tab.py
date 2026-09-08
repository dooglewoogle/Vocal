"""Dictation tab: Whisper model grid + every input-side setting."""

from __future__ import annotations

import copy
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from vocal.gui.settings_form import SettingsForm, dictation_fields
from vocal.gui.tooltip import tip

if TYPE_CHECKING:
    from vocal.gui.window import VocalWindow


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


def whisper_models() -> list[str]:
    from vocal.input.transcriber import VALID_MODELS

    order = ["tiny", "base", "small", "medium", "large", "distil"]

    def key(m: str) -> tuple[int, str]:
        for i, o in enumerate(order):
            if m.startswith(o):
                return (i, m)
        return (len(order), m)

    return sorted(VALID_MODELS, key=key)


class DictationTab(ttk.Frame):
    def __init__(self, master: tk.Misc, window: "VocalWindow") -> None:
        super().__init__(master)
        self.window = window
        self.app = window.app
        self.form = SettingsForm(self, window, dictation_fields(), header=self._build_grid)
        self.form.pack(fill="both", expand=True)
        self.refresh()

    # ── Grid ─────────────────────────────────────────────────────────

    def _build_grid(self, parent: ttk.Frame, _form: SettingsForm) -> None:
        box = ttk.LabelFrame(parent, text="Whisper models", padding=(10, 6))
        box.pack(fill="x", padx=4, pady=(4, 8))
        self._models = ttk.Treeview(box, columns=("cached", "current"), show="tree headings",
                                    height=7, selectmode="browse")
        self._models.heading("#0", text="Model")
        self._models.column("#0", width=200, stretch=False)
        self._models.heading("cached", text="Downloaded")
        self._models.column("cached", width=100, stretch=False, anchor="center")
        self._models.heading("current", text="Current")
        self._models.column("current", width=80, stretch=True, anchor="center")
        self._models.pack(fill="x")
        bar = ttk.Frame(box)
        bar.pack(fill="x", pady=(6, 0))
        tip(ttk.Button(bar, text="Use this model", command=self._use_model),
            "Switch dictation to the selected Whisper model. It is downloaded on first use "
            "(100 MB for tiny, ~3 GB for large) and the dictation engine restarts.").pack(side="left")
        tip(self._models, "Whisper speech-to-text models. Bigger is more accurate and slower; .en models are "
                          "English-only and a little more accurate for English. Downloaded = already in the "
                          "Hugging Face cache. Current = what dictation is using now.")
        self._grid_status = ttk.Label(
            bar, foreground="#666",
            text="Models download on first use (100 MB – 3 GB). Switching restarts dictation.",
        )
        self._grid_status.pack(side="left", padx=12)

    def refresh(self) -> None:
        selected = self._selected()
        self._models.delete(*self._models.get_children())
        current = self.app.config.input.model.size
        sizes = whisper_models()
        for size in sizes:
            self._models.insert("", "end", iid=size, text=size, values=("…", "✓" if size == current else ""))
        if selected and self._models.exists(selected):
            self._models.selection_set(selected)

        def probe() -> dict[str, bool | None]:
            return {s: whisper_model_cached(s) for s in sizes}

        def done(result: object) -> None:
            if not isinstance(result, dict):
                return
            for size, cached in result.items():
                if self._models.exists(size):
                    self._models.set(size, "cached", "✓" if cached else "")

        self.window.run_bg(probe, on_done=done, name="whisper-cache-probe")

    def _selected(self) -> str | None:
        sel = self._models.selection()
        return sel[0] if sel else None

    def _use_model(self) -> None:
        size = self._selected()
        if not size:
            self._grid_status.configure(text="Select a model first")
            return
        if size == self.app.config.input.model.size:
            return
        cfg = copy.deepcopy(self.app.config)
        cfg.input.model.size = size
        self._grid_status.configure(text=f"Switching to {size}…")

        def done(_notes: object) -> None:
            self._grid_status.configure(text=f"Using {size}")
            self.window.after_apply()

        self.window.run_bg(lambda: self.app.apply_config(cfg), on_done=done,
                           on_error=lambda e: self._grid_status.configure(text=f"Failed: {e}"), name="set-model")
