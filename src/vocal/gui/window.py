"""VocalWindow — the Tk root, tab container and cross-thread plumbing.

Threading contract (see plan §2):

- Tk lives on the main thread. Every widget update from another thread goes
  through :meth:`call_soon`, a queue drained by a 50 ms ``after`` pump.
- The pump also wakes the interpreter so Python signal handlers run promptly
  while ``mainloop()`` is blocked; it must be started before the loop and
  must never die.
- Anything that may block (``app.apply_config``, ``app.stop_speaking``,
  downloads) runs via :meth:`run_bg` on a daemon thread and reports back
  through :meth:`call_soon`. Nothing ever waits on the Tk thread.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vocal.app import VocalApp

logger = logging.getLogger(__name__)

_PUMP_MS = 50


class VocalWindow:
    def __init__(self, app: "VocalApp") -> None:
        self.app = app
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Vocal")
        self.root.minsize(640, 480)
        self.root.geometry("760x560")
        try:
            ttk.Style().theme_use("clam")
        except tk.TclError:  # pragma: no cover - theme missing on exotic builds
            pass
        from vocal.gui.tooltip import install_style

        install_style()

        self._q: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._has_tray = False
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        from vocal.gui.dictation_tab import DictationTab
        from vocal.gui.phrasebook_tab import PhrasebookTab
        from vocal.gui.speech_tab import SpeechTab
        from vocal.gui.status_tab import StatusTab

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)
        self.status = StatusTab(self.notebook, self)
        self.dictation = DictationTab(self.notebook, self)
        self.speech = SpeechTab(self.notebook, self)
        self.phrasebook = PhrasebookTab(self.notebook, self)
        for tab, label in ((self.status, "Status"), (self.dictation, "Dictation"),
                           (self.speech, "Speech"), (self.phrasebook, "Phrasebook")):
            self.notebook.add(tab, text=label)

        self._install_wheel_scrolling()

        # App events → UI thread
        app.on_state.connect(lambda s: self.call_soon(lambda: self.status.set_state(s)))
        app.on_speaking.connect(lambda b: self.call_soon(lambda: self.status.set_speaking(b)))
        app.on_transcript.connect(lambda t: self.call_soon(lambda: self.status.add_transcript(t)))
        app.on_rebuild.connect(lambda b: self.call_soon(lambda: self._on_rebuild(b)))

    # ── Mouse wheel ──────────────────────────────────────────────────

    def _install_wheel_scrolling(self) -> None:
        """Make the wheel scroll the settings page from anywhere over it.

        Tk sends wheel events only to the widget under the pointer, so a
        canvas-level binding never fires over labels and entries. One
        window-wide handler walks up from the hovered widget to the nearest
        canvas flagged ``wheel_scrollable`` and scrolls that. Widgets that
        scroll themselves (grids, the transcript box) are left alone, and
        comboboxes lose Tk's default "wheel changes the selection" behaviour,
        which is never what you want on a settings page.
        """
        for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            self.root.bind_all(seq, self._on_wheel, add="+")
            self.root.bind_class("TCombobox", seq, self._on_wheel)  # replaces the value-changing default

    @staticmethod
    def _wheel_units(event: tk.Event) -> int:
        if getattr(event, "num", None) == 4:
            return -3
        if getattr(event, "num", None) == 5:
            return 3
        delta = getattr(event, "delta", 0)
        if not delta:
            return 0
        return -3 if delta > 0 else 3  # Windows: ±120 per notch; macOS: small values

    def _on_wheel(self, event: tk.Event) -> str | None:
        widget = event.widget
        if isinstance(widget, str):  # bind_all may hand us a path name
            widget = self.root.nametowidget(widget)
        if isinstance(widget, (ttk.Treeview, tk.Text, tk.Listbox)):
            return None  # scrolls itself
        node: tk.Misc | None = widget
        while node is not None and not getattr(node, "wheel_scrollable", False):
            node = node.master
        if node is None:
            return None
        units = self._wheel_units(event)
        if units:
            node.yview_scroll(units, "units")  # type: ignore[attr-defined]
        return "break" if isinstance(widget, ttk.Combobox) else None

    # ── Cross-thread plumbing ────────────────────────────────────────

    def call_soon(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the Tk thread at the next pump tick. Never blocks."""
        self._q.put(fn)

    def run_bg(
        self,
        fn: Callable[[], object],
        on_done: Callable[[object], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        name: str = "gui-worker",
    ) -> None:
        """Run ``fn`` on a daemon thread; deliver its result/exception on the Tk thread."""

        def body() -> None:
            try:
                result = fn()
            except BaseException as e:  # noqa: BLE001 - reported to the UI
                logger.exception("Background task %s failed", name)
                exc = e  # `e` is unbound once the except block ends
                if on_error is not None:
                    self.call_soon(lambda: on_error(exc))
                return
            if on_done is not None:
                self.call_soon(lambda: on_done(result))

        threading.Thread(target=body, name=name, daemon=True).start()

    def start_pump(self) -> None:
        self.root.after(_PUMP_MS, self._pump)

    def _pump(self) -> None:
        try:
            while True:
                fn = self._q.get_nowait()
                try:
                    fn()
                except Exception:
                    logger.exception("UI callback raised")
        except queue.Empty:
            pass
        except Exception:  # pragma: no cover - the pump must never die
            logger.exception("UI pump error")
        finally:
            if not self._closed:
                self.root.after(_PUMP_MS, self._pump)

    # ── Window lifecycle ─────────────────────────────────────────────

    def set_has_tray(self, has_tray: bool) -> None:
        self._has_tray = has_tray
        self.status.set_close_hint(has_tray)

    def show(self) -> None:
        self.root.deiconify()
        self.root.lift()
        try:
            self.root.focus_force()
        except tk.TclError:  # pragma: no cover
            pass

    def show_soon(self) -> None:
        """Thread-safe show (used by the tray's Open item)."""
        self.call_soon(self.show)

    def hide(self) -> None:
        self.root.withdraw()

    def quit_soon(self) -> None:
        """Thread-safe: end mainloop() at the next pump tick."""
        self.call_soon(self.root.quit)

    def _on_close(self) -> None:
        if self._has_tray:
            self.hide()
        else:
            self.app.request_shutdown()

    def mainloop(self) -> None:
        self.root.mainloop()

    def destroy(self) -> None:
        self._closed = True
        try:
            self.root.destroy()
        except tk.TclError:  # pragma: no cover
            pass

    # ── Shared helpers for tabs ──────────────────────────────────────

    def after_apply(self) -> None:
        """Re-sync every view with the live config after any apply_config."""
        self.status.refresh_summary()
        self.dictation.form.reload_from_app()
        self.speech.form.reload_from_app()
        self.dictation.refresh()
        self.speech.refresh()
        self.phrasebook.refresh_flags()

    def _on_rebuild(self, started: bool) -> None:
        self.status.set_rebuilding(started)
        self.dictation.form.set_busy(started)
        self.speech.form.set_busy(started)
        if not started:
            self.after_apply()


def run_gui(app: "VocalApp", on_ready: Callable[[VocalWindow], None] | None = None) -> None:
    """GUI-mode main: Tk on this (main) thread, tray detached, engine in background.

    ``on_ready`` (tests/drivers) is called with the window just before the loop starts."""
    from vocal.tray import TrayIcon
    from vocal.utils import check_tray_dependencies

    window = VocalWindow(app)
    tray: TrayIcon | None = None
    if sys.platform != "darwin" and not check_tray_dependencies():
        candidate = TrayIcon(
            on_open=window.show_soon,
            on_toggle_pause=app.toggle_pause,
            on_stop_speaking=app.stop_speaking,
            on_quit=app.request_shutdown,
        )
        if candidate.run_detached():
            tray = candidate
            app.on_state.connect(tray.set_state)
            app.on_speaking.connect(tray.set_speaking)
        else:
            logger.warning("Running without a tray icon; closing the window quits")
    window.set_has_tray(tray is not None)

    app.start(quit_loop=window.quit_soon)
    app.install_signal_handlers(glib=False)  # the pump wakes Tk for these
    window.start_pump()
    window.status.refresh_summary()
    window.show()
    if on_ready is not None:
        on_ready(window)
    try:
        window.mainloop()
    finally:
        app.shutdown()
        if tray is not None:
            tray.stop()
        window.destroy()
        _exit_backstop()


def _exit_backstop(grace: float = 2.0) -> None:
    """pystray's setup thread is non-daemon and can pin the interpreter if its
    backend never marked ready. Join briefly, then force exit rather than hang."""
    others = [t for t in threading.enumerate()
              if t is not threading.main_thread() and not t.daemon]
    for t in others:
        t.join(timeout=grace)
    stragglers = [t.name for t in others if t.is_alive()]
    if stragglers:
        logger.warning("Forcing exit; non-daemon threads still alive: %s", ", ".join(stragglers))
        logging.shutdown()
        os._exit(0)
