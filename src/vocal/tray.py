"""System tray icon — wraps pystray, isolates the engine from the GUI toolkit.

The engine never imports pystray directly. All GUI-toolkit touchpoints live
here so a future swap (e.g. direct AyatanaAppIndicator3 via gi) only requires
replacing this file.
"""

from __future__ import annotations

import logging
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
    - Call `run()` on the **main thread** — it blocks until `stop()` is invoked.
    - Call `set_state()`, `stop()`, and the menu-callback setters from any thread.
    """

    def __init__(
        self,
        *,
        on_toggle_pause: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        self._on_toggle_pause = on_toggle_pause
        self._on_quit = on_quit

        self._state: DictationState = DictationState.LISTENING
        self._state_lock = threading.Lock()

        self._images: dict[DictationState, object] = {}
        self._icon: "pystray.Icon | None" = None
        self._stop_requested = False

    # ── Public API ──────────────────────────────────────────────────

    def set_state(self, state: DictationState) -> None:
        """Update the tray icon + menu to reflect a new engine state."""
        with self._state_lock:
            if self._state == state:
                return
            self._state = state

        icon = self._icon
        if icon is None:
            return  # run() hasn't been called yet; initial state picked up on start

        try:
            icon.icon = self._image_for(state)
            icon.title = f"{APP_TITLE} — {_STATE_LABEL[state]}"
            icon.update_menu()
        except Exception:
            logger.exception("Failed to update tray for state %s", state.value)

    def run(self) -> None:
        """Block on the tray event loop. Must be called from the main thread."""
        import pystray

        if self._stop_requested:
            logger.info("Tray stop requested before run; skipping loop")
            return

        # Pre-load all images so state changes don't hit disk during a transition.
        for st, fname in _ASSET_FOR_STATE.items():
            try:
                self._images[st] = _load_image(fname)
            except Exception:
                logger.exception("Missing tray asset %s for state %s", fname, st.value)

        initial_state = self._state
        self._icon = pystray.Icon(
            APP_NAME,
            icon=self._image_for(initial_state),
            title=f"{APP_TITLE} — {_STATE_LABEL[initial_state]}",
            menu=self._build_menu(),
        )
        logger.info("Tray icon starting (initial state: %s)", initial_state.value)

        # Handle the stop-before-icon-exists race: if stop() was called in the
        # tiny window between the _stop_requested check above and Icon()
        # construction, call icon.stop() immediately to mark it for exit.
        if self._stop_requested:
            try:
                self._icon.stop()
            except Exception:
                pass

        self._icon.run()  # blocks until stop() is called
        logger.info("Tray icon run loop returned")

    def stop(self) -> None:
        """Signal the tray loop to exit. Safe from any thread."""
        self._stop_requested = True
        icon = self._icon
        if icon is None:
            return
        try:
            icon.stop()
        except Exception:
            logger.exception("Error stopping tray icon")

    # ── Internal ────────────────────────────────────────────────────

    def _image_for(self, state: DictationState) -> object:
        return self._images.get(state) or self._images.get(DictationState.LISTENING)

    def _current_state(self) -> DictationState:
        with self._state_lock:
            return self._state

    def _build_menu(self) -> "pystray.Menu":
        import pystray

        def status_text(_item: object) -> str:
            return f"Status: {_STATE_LABEL[self._current_state()]}"

        def pause_text(_item: object) -> str:
            return "Resume" if self._current_state() == DictationState.SLEEPING else "Pause"

        def on_pause_clicked(_icon: object, _item: object) -> None:
            try:
                self._on_toggle_pause()
            except Exception:
                logger.exception("pause callback raised")

        def on_quit_clicked(_icon: object, _item: object) -> None:
            try:
                self._on_quit()
            except Exception:
                logger.exception("quit callback raised")

        return pystray.Menu(
            pystray.MenuItem(status_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(pause_text, on_pause_clicked),
            pystray.MenuItem("Quit", on_quit_clicked),
        )
