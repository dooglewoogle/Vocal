"""Hover tooltips for Tk/ttk widgets (Tk has none built in).

Show after a short delay on ``<Enter>``, hide on ``<Leave>`` or any click. The
popup is an undecorated ``Toplevel`` positioned next to the pointer, so it
works inside scrolled canvases and above other windows.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

DELAY_MS = 500
WRAP_PX = 380
STYLE = "Tooltip.TLabel"


def install_style() -> None:
    """Register the tooltip label style. Call once after the theme is set."""
    ttk.Style().configure(STYLE, background="#fff8dc", foreground="#222", relief="solid", borderwidth=1)


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = DELAY_MS) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _event: object = None) -> None:
        self._cancel()
        self._after = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except tk.TclError:  # pragma: no cover - widget gone
                pass
            self._after = None

    def _show(self) -> None:
        self._after = None
        if self._tip is not None or not self.text or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_pointerx() + 12
        y = self.widget.winfo_pointery() + 18
        # Keep the popup on screen horizontally.
        x = min(x, self.widget.winfo_screenwidth() - WRAP_PX - 40)
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        try:
            tip.attributes("-topmost", True)
        except tk.TclError:  # pragma: no cover - some WMs
            pass
        tip.wm_geometry(f"+{x}+{y}")
        ttk.Label(tip, text=self.text, style=STYLE, wraplength=WRAP_PX, justify="left", padding=(8, 4)).pack()
        self._tip = tip

    def _hide(self, _event: object = None) -> None:
        self._cancel()
        tip, self._tip = self._tip, None
        if tip is not None:
            try:
                tip.destroy()
            except tk.TclError:  # pragma: no cover
                pass

    @property
    def visible(self) -> bool:
        return self._tip is not None


def tip(widget: tk.Widget, text: str | None) -> tk.Widget:
    """Attach a tooltip (if ``text``) and return the widget, so it chains into pack/grid calls."""
    if text:
        Tooltip(widget, text)
    return widget
