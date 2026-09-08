"""Phrasebook tab: edit mishearing → correction rules and hot-swap them into the engine."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from vocal.input.phrasebook import PHRASEBOOK_PATH, load_phrasebook, save_phrasebook

if TYPE_CHECKING:
    from vocal.gui.window import VocalWindow


def open_in_editor(path=PHRASEBOOK_PATH) -> None:
    """Open the phrasebook file with the desktop's default editor, creating it if needed."""
    if not path.exists():
        save_phrasebook({}, path)
    if sys.platform == "linux":
        subprocess.Popen(["xdg-open", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        os.startfile(str(path))  # type: ignore[attr-defined]


class PhrasebookTab(ttk.Frame):
    def __init__(self, master: tk.Misc, window: "VocalWindow") -> None:
        super().__init__(master, padding=8)
        self.window = window
        self.app = window.app
        self._rules: dict[str, str] = {}

        ttk.Label(self, text="When Vocal hears the left, it writes the right (whole words, case-insensitive). "
                             "The right-hand terms also bias recognition when seeding is on.",
                  wraplength=700, foreground="#555").pack(anchor="w", pady=(0, 6))

        self._tree = ttk.Treeview(self, columns=("right",), show="tree headings", height=12, selectmode="browse")
        self._tree.heading("#0", text="Heard as")
        self._tree.heading("right", text="Write")
        self._tree.column("#0", width=280)
        self._tree.column("right", width=280)
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        edit = ttk.Frame(self)
        edit.pack(fill="x", pady=6)
        self._wrong = tk.StringVar()
        self._right = tk.StringVar()
        ttk.Entry(edit, textvariable=self._wrong, width=30).pack(side="left")
        ttk.Label(edit, text="→").pack(side="left", padx=6)
        ttk.Entry(edit, textvariable=self._right, width=30).pack(side="left")
        ttk.Button(edit, text="Add / Update", command=self._add).pack(side="left", padx=6)
        ttk.Button(edit, text="Delete", command=self._delete).pack(side="left")

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(4, 0))
        ttk.Button(bar, text="Save & Apply", command=self.save).pack(side="left")
        ttk.Button(bar, text="Reload from file", command=self.reload).pack(side="left", padx=6)
        ttk.Button(bar, text="Open in editor", command=lambda: open_in_editor()).pack(side="left")
        self._status = ttk.Label(bar, text="", foreground="#666")
        self._status.pack(side="right")

        flags = ttk.Frame(self)
        flags.pack(fill="x", pady=(8, 0))
        self._seed = tk.BooleanVar(value=self.app.config.input.phrasebook.seed)
        self._replace = tk.BooleanVar(value=self.app.config.input.phrasebook.replace)
        ttk.Checkbutton(flags, text="Bias recognition toward these terms", variable=self._seed,
                        command=self._apply_flags).pack(side="left")
        ttk.Checkbutton(flags, text="Apply corrections after transcription", variable=self._replace,
                        command=self._apply_flags).pack(side="left", padx=12)

        self.reload()

    # ── Data ─────────────────────────────────────────────────────────

    def reload(self) -> None:
        self._rules = dict(load_phrasebook().replacements)
        self._render()
        self._status.configure(text=f"{len(self._rules)} rule(s) from {PHRASEBOOK_PATH}")

    def refresh_flags(self) -> None:
        self._seed.set(self.app.config.input.phrasebook.seed)
        self._replace.set(self.app.config.input.phrasebook.replace)

    def _render(self) -> None:
        self._tree.delete(*self._tree.get_children())
        for wrong, right in self._rules.items():
            self._tree.insert("", "end", iid=wrong, text=wrong, values=(right,))

    def _on_select(self, _e: object) -> None:
        sel = self._tree.selection()
        if sel:
            self._wrong.set(sel[0])
            self._right.set(self._rules.get(sel[0], ""))

    def _add(self) -> None:
        wrong, right = self._wrong.get().strip(), self._right.get().strip()
        if not wrong or not right:
            self._status.configure(text="Both fields are required")
            return
        self._rules[wrong] = right
        self._render()
        self._tree.selection_set(wrong)
        self._status.configure(text="Unsaved changes")

    def _delete(self) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        self._rules.pop(sel[0], None)
        self._render()
        self._wrong.set("")
        self._right.set("")
        self._status.configure(text="Unsaved changes")

    # ── Actions ──────────────────────────────────────────────────────

    def save(self) -> None:
        rules = dict(self._rules)

        def work() -> int:
            save_phrasebook(rules)
            self.app.set_phrasebook(load_phrasebook())
            return len(rules)

        self.window.run_bg(
            work,
            on_done=lambda n: self._status.configure(text=f"Saved {n} rule(s); active now"),
            on_error=lambda e: self._status.configure(text=f"Save failed: {e}"),
            name="phrasebook-save",
        )

    def _apply_flags(self) -> None:
        import copy

        cfg = copy.deepcopy(self.app.config)
        cfg.input.phrasebook.seed = self._seed.get()
        cfg.input.phrasebook.replace = self._replace.get()
        self.window.run_bg(
            lambda: self.app.apply_config(cfg),
            on_done=lambda _n: (self._status.configure(text="Phrasebook settings applied"),
                                self.window.after_apply()),
            on_error=lambda e: self._status.configure(text=f"Failed: {e}"),
            name="phrasebook-flags",
        )
