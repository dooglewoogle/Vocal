"""System tray icon — wraps pystray, isolates the app from the GUI toolkit.

The menu is deliberately small (status, Open, Pause/Resume, Stop speaking,
Quit); everything else lives in the GUI window. Two ways to run:

- ``run()``: blocks the calling (main) thread on pystray's own loop. Used in
  headless mode.
- ``run_detached()``: spawns a "tray" thread that imports pystray, builds the
  icon and — on Linux — runs a GLib main loop of our own, because pystray's
  GTK backends do not spin one when detached. Everything GTK then happens on
  that single thread (pystray schedules its updates via ``idle_add``). Used
  in GUI mode, where Tk owns the main thread.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from importlib.resources import files
from typing import TYPE_CHECKING

from vocal.state import DictationState

if TYPE_CHECKING:  # avoid importing pystray at module load
    import pystray  # noqa: F401
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

APP_NAME = "vocal"
APP_TITLE = "Vocal"


# State → asset filename. Recording reuses the awake icon so a momentary
# capture doesn't flash a third icon; TRANSCRIBING uses busy.
_ASSET_FOR_STATE: dict[DictationState, str] = {
    DictationState.SLEEPING: "vocal-sleep.png",
    DictationState.LISTENING: "vocal-awake.png",
    DictationState.RECORDING: "vocal-awake.png",
    DictationState.TRANSCRIBING: "vocal-busy.png",
}

_STATE_LABEL: dict[DictationState, str] = {
    DictationState.SLEEPING: "Paused",
    DictationState.LISTENING: "Listening",
    DictationState.RECORDING: "Recording",
    DictationState.TRANSCRIBING: "Transcribing...",
}


def _load_image(filename: str) -> "PILImage":
    """Load a packaged PNG asset. Raises FileNotFoundError with a clear message."""
    from PIL import Image

    asset = files("vocal").joinpath("assets", filename)
    with asset.open("rb") as f:
        return Image.open(f).copy()  # copy() detaches from the file handle


class TrayIcon:
    """Thread-safe wrapper around pystray.Icon.

    - Construct it on any thread.
    - Call ``run()`` on the main thread (blocks) **or** ``run_detached()``.
    - ``set_state()``, ``set_speaking()`` and ``stop()`` are safe from any thread.
    """

    def __init__(
        self,
        *,
        on_toggle_pause: Callable[[], None],
        on_quit: Callable[[], None],
        on_stop_speaking: Callable[[], None] | None = None,
        on_open: Callable[[], None] | None = None,
    ) -> None:
        self._on_toggle_pause = on_toggle_pause
        self._on_quit = on_quit
        self._on_stop_speaking = on_stop_speaking
        self._on_open = on_open

        self._state: DictationState = DictationState.LISTENING
        self._state_lock = threading.Lock()
        # Speech output is an overlay, independent of the dictation state:
        # we can be Listening *and* Speaking.
        self._speaking = False

        self._images: dict[DictationState, object] = {}
        self._icon: "pystray.Icon | None" = None
        self._stop_requested = False

        # run_detached() bookkeeping
        self._thread: threading.Thread | None = None
        self._loop: object | None = None  # GLib.MainLoop on Linux
        self._ready = threading.Event()
        self._detached_ok = False

    # ── Public API ──────────────────────────────────────────────────

    def set_state(self, state: DictationState) -> None:
        """Update the tray icon + menu to reflect a new engine state."""
        with self._state_lock:
            if self._state == state:
                return
            self._state = state
        self._refresh_icon()

    def set_speaking(self, speaking: bool) -> None:
        """Show/hide the speaking overlay (busy icon + title suffix). Any thread."""
        with self._state_lock:
            if self._speaking == speaking:
                return
            self._speaking = speaking
        self._refresh_icon()

    def run(self) -> None:
        """Block on the tray event loop. Must be called from the main thread."""
        import pystray  # noqa: F401

        if self._stop_requested:
            logger.info("Tray stop requested before run; skipping loop")
            return
        icon = self._create_icon()
        logger.info("Tray icon starting (initial state: %s)", self._state.value)
        # Handle the stop-before-icon-exists race: if stop() was called in the
        # tiny window between the check above and Icon() construction, call
        # icon.stop() immediately to mark it for exit.
        if self._stop_requested:
            try:
                icon.stop()
            except Exception:
                pass
        icon.run()  # blocks until stop() is called
        logger.info("Tray icon run loop returned")

    def run_detached(self, timeout: float = 10.0) -> bool:
        """Start the icon on a background thread. Returns False if no tray could
        be shown (macOS, missing backend, no session bus…) — the caller then
        runs without one."""
        if sys.platform == "darwin":
            # pystray's Cocoa backend needs the NSApplication that Tk owns, and
            # fails silently rather than raising; don't pretend we have a tray.
            logger.info("Tray disabled on macOS in GUI mode")
            return False
        if self._thread is not None:
            return self._detached_ok
        self._thread = threading.Thread(target=self._detached_main, name="tray", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            logger.warning("Tray did not become ready within %.0fs; continuing without it", timeout)
            return False
        return self._detached_ok

    def stop(self) -> None:
        """Signal the tray loop to exit. Safe from any thread."""
        self._stop_requested = True
        loop = self._loop
        if loop is not None:
            from gi.repository import GLib
            GLib.idle_add(loop.quit)  # type: ignore[attr-defined]
        icon = self._icon
        if icon is not None:
            try:
                icon.stop()  # no-op for a detached GTK icon; needed for run()/win32
            except Exception:
                logger.exception("Error stopping tray icon")
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("Tray thread did not exit")

    # ── Internal: construction / loops ──────────────────────────────

    def _create_icon(self) -> "pystray.Icon":
        import pystray

        # Pre-load all images so state changes don't hit disk during a transition.
        for st, fname in _ASSET_FOR_STATE.items():
            try:
                self._images[st] = _load_image(fname)
            except Exception:
                logger.exception("Missing tray asset %s for state %s", fname, st.value)
        self._icon = pystray.Icon(
            APP_NAME,
            icon=self._image_for(self._state),
            title=self._title(),
            menu=self._build_menu(),
        )
        return self._icon

    def _detached_main(self) -> None:
        # Import + construction must happen here: pystray's GTK backend calls
        # Gtk.init_check() at import and AppIndicator.Indicator.new() in
        # Icon.__init__ on the calling thread, and its _initialize() would
        # reset SIGINT to default if it ran on the main thread.
        assert threading.current_thread() is not threading.main_thread()
        try:
            icon = self._create_icon()
            if sys.platform == "linux":
                from gi.repository import GLib
                self._loop = GLib.MainLoop()
                icon.run_detached()
                self._detached_ok = True
                self._ready.set()
                logger.info("Tray icon running detached (GLib loop on tray thread)")
                self._loop.run()  # type: ignore[attr-defined]
                logger.info("Tray GLib loop exited")
            else:
                icon.run_detached()  # win32: pystray spawns its own message loop thread
                self._detached_ok = True
                self._ready.set()
        except Exception:
            logger.exception("Tray icon failed to start; continuing without a tray")
            self._icon = None
            self._ready.set()

    # ── Internal helpers ────────────────────────────────────────────

    def _title(self) -> str:
        with self._state_lock:
            label = _STATE_LABEL[self._state]
            speaking = self._speaking
        return f"{APP_TITLE} — {'Speaking · ' if speaking else ''}{label}"

    def _refresh_icon(self) -> None:
        icon = self._icon
        if icon is None:
            return  # not running yet; initial state picked up on start
        with self._state_lock:
            state = DictationState.TRANSCRIBING if self._speaking else self._state
        try:
            icon.icon = self._image_for(state)
            icon.title = self._title()
            icon.update_menu()
        except Exception:
            logger.exception("Failed to update tray for state %s", state.value)

    def _image_for(self, state: DictationState) -> object:
        return self._images.get(state) or self._images.get(DictationState.LISTENING)

    def _current_state(self) -> DictationState:
        with self._state_lock:
            return self._state

    def _call(self, name: str, fn: Callable[[], None] | None) -> None:
        if fn is None:
            return
        try:
            fn()
        except Exception:
            logger.exception("%s callback raised", name)

    # ── Menu construction ───────────────────────────────────────────

    def _build_menu(self) -> "pystray.Menu":
        import pystray

        def status_text(_item: object) -> str:
            label = _STATE_LABEL[self._current_state()]
            return f"Status: {'Speaking · ' if self._speaking else ''}{label}"

        def pause_text(_item: object) -> str:
            return "Resume" if self._current_state() == DictationState.SLEEPING else "Pause"

        items: list = [
            pystray.MenuItem(status_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
        ]
        if self._on_open is not None:
            items.append(pystray.MenuItem(
                "Open Vocal",
                lambda _icon, _item: self._call("open", self._on_open),
                default=True,
            ))
        items.append(pystray.MenuItem(
            pause_text, lambda _icon, _item: self._call("pause", self._on_toggle_pause),
        ))
        if self._on_stop_speaking is not None:
            items.append(pystray.MenuItem(
                "Stop speaking",
                lambda _icon, _item: self._call("stop speaking", self._on_stop_speaking),
                enabled=lambda _item: self._speaking,
            ))
        items.append(pystray.Menu.SEPARATOR)
        items.append(pystray.MenuItem("Quit", lambda _icon, _item: self._call("quit", self._on_quit)))
        return pystray.Menu(*items)
