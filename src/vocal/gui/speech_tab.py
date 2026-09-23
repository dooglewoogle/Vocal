"""Speech tab: text-to-speech voice grid + every output-side setting."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from vocal.gui.settings_form import SERVER_FIELDS, SettingsForm, make_field_widget, section, speech_fields
from vocal.gui.tooltip import tip

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
                tip(ttk.Label(row, text=spec.label), spec.help).pack(side="left", padx=(16, 4))
                widget.configure(width=16 if spec.path.endswith("host") else 7)  # type: ignore[call-arg]
            tip(widget, spec.help).pack(side="left")
            form.add_field(spec, widget, var)

        box = section(parent, "Voices")
        bar = ttk.Frame(box)
        bar.pack(fill="x", pady=(0, 4))
        self._filter = tk.StringVar()
        ttk.Label(bar, text="Filter").pack(side="left")
        tip(ttk.Entry(bar, textvariable=self._filter, width=28),
            "Show only voices whose name, language or description contains this text, "
            "e.g. \"britain\", \"en_GB\", \"female\" or \"speakers\".").pack(side="left", padx=(6, 2))
        ttk.Button(bar, text="✕", width=2, command=lambda: self._filter.set("")).pack(side="left")
        self._filter.trace_add("write", lambda *_: self.refresh())

        cols = ("downloaded", "size", "default", "description")
        grid = ttk.Frame(box)
        grid.pack(fill="x")
        self._voices = ttk.Treeview(grid, columns=cols, show="tree headings", height=14, selectmode="browse")
        scroll = ttk.Scrollbar(grid, orient="vertical", command=self._voices.yview)
        self._voices.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._voices.heading("#0", text="Voice")
        self._voices.column("#0", width=300, stretch=False)
        for c, w in zip(cols, (105, 70, 70, 300)):
            self._voices.heading(c, text=c.capitalize())
            self._voices.column(c, width=w, anchor="w" if c == "description" else "center",
                                stretch=(c == "description"))
        self._voices.pack(side="left", fill="x", expand=True)
        self._open: dict[str, bool] = {}  # group expansion chosen while unfiltered
        self._shown_filter = ""

        bar = ttk.Frame(box)
        bar.pack(fill="x", pady=(6, 0))
        self._dl_btn = ttk.Button(bar, text="Download", command=self._download)
        tip(self._dl_btn, "Fetch the selected voice's model files into ~/.cache/vocal/models. Piper voices "
                          "are 20-140 MB each; all Kokoro voices share one 354 MB download.").pack(side="left")
        tip(ttk.Button(bar, text="Remove", command=self._remove),
            "Delete the selected voice's downloaded files. It can be downloaded again later.").pack(side="left", padx=4)
        tip(ttk.Button(bar, text="Set as default", command=self._use_voice),
            "Make the selected voice the default, and save that choice. The default speaks whenever a "
            "request names no voice or one Vocal does not know.").pack(side="left", padx=4)
        tip(ttk.Button(bar, text="Test", command=self._test),
            "Speak a sample sentence with the selected voice, interrupting anything currently "
            "playing. Downloads the voice first if needed.").pack(side="left", padx=4)
        tip(self._voices, "Text-to-speech voices by engine and language. piper-* are fast and light; kokoro-* "
                          "sound more natural but take about a second to start on CPU; system uses the OS "
                          "speech engine. Downloaded = files present. Default = the voice used when a request "
                          "names none, or an unknown one. Multi-speaker Piper voices take #N in requests, "
                          "e.g. piper-en_GB-vctk-medium#12.")
        self._voice_status = ttk.Label(bar, text="", foreground="#666")
        self._voice_status.pack(side="left", padx=12)

    def refresh(self) -> None:
        from vocal.output.models import VOICES, effective_default, get_voice, human_size, is_downloaded, matches

        tree = self._voices
        needle = self._filter.get().strip()
        if not self._shown_filter:  # the tree on screen shows the user's own expansion
            for group in self._groups():
                self._open[group] = bool(tree.item(group, "open"))
        self._shown_filter = needle

        selected = self._selected()
        tree.delete(*tree.get_children())
        default = get_voice(effective_default(self.app.config.output.speech.voice))
        default_group = f"grp:{default.backend}:{default.language_name}"
        engines = {"kokoro": 0, "piper": 1, "system": 2}
        for spec in sorted(VOICES.values(), key=lambda v: (engines[v.backend], v.language_name, v.name)):
            if needle and not matches(spec, needle):
                continue
            parent = f"grp:{spec.backend}"
            if not tree.exists(parent):
                tree.insert("", "end", iid=parent, text=spec.backend.capitalize(),
                            open=bool(needle) or self._open.get(parent, True))
            if spec.language_name:
                engine, parent = parent, f"{parent}:{spec.language_name}"
                if not tree.exists(parent):
                    first_look = spec.language.startswith("en_") or parent == default_group
                    tree.insert(engine, "end", iid=parent, text=spec.language_name,
                                open=bool(needle) or self._open.get(parent, first_look))
            tree.insert(parent, "end", iid=spec.name, text=spec.name, values=(
                "✓" if is_downloaded(spec) else "", human_size(spec.size_bytes),
                "✓" if spec.name == default.name else "", spec.description,
            ))
        if selected and tree.exists(selected):
            tree.selection_set(selected)
            tree.see(selected)
        if needle and not tree.get_children():
            self._set_status(f"No voice matches {needle!r}")

    def _groups(self) -> list[str]:
        tree = self._voices
        return [g for e in tree.get_children() for g in (e, *tree.get_children(e)) if g.startswith("grp:")]

    def _selected(self) -> str | None:
        sel = self._voices.selection()
        return sel[0] if sel and not sel[0].startswith("grp:") else None

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
