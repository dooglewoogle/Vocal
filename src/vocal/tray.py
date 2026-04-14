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

# Models shown in the tray submenu, ordered by size.
_TRAY_MODELS: list[str] = [
    "tiny.en", "base.en", "small.en", "medium.en",
    "tiny", "base", "small", "medium",
    "large-v3",
    "distil-small.en", "distil-medium.en", "distil-large-v3",
]


def _load_image(filename: str) -> "PILImage":
    """Load a packaged PNG asset. Raises FileNotFoundError with a clear message."""
    from PIL import Image

    asset = files("vocal").joinpath("assets", filename)
    with asset.open("rb") as f:
        return Image.open(f).copy()  # copy() detaches from the file handle


def _get_input_devices() -> list[tuple[int, str, bool]]:
    """Return [(index, name, is_system_default)] for all input-capable devices."""
    try:
        import sounddevice as sd

        default_input = sd.default.device[0]
        return [
            (i, dev["name"], i == default_input)
            for i, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0
        ]
    except Exception:
        logger.exception("Failed to query audio devices")
        return []


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
        on_select_device: Callable[[int | None], None],
        on_select_model: Callable[[str], None],
        on_switch_mode: Callable[[str], None],
        on_open_phrasebook: Callable[[], None],
        current_model: str = "small.en",
        current_mode: str = "live",
        current_device: int | None = None,
    ) -> None:
        self._on_toggle_pause = on_toggle_pause
        self._on_quit = on_quit
        self._on_select_device = on_select_device
        self._on_select_model = on_select_model
        self._on_switch_mode = on_switch_mode
        self._on_open_phrasebook = on_open_phrasebook

        self._state: DictationState = DictationState.LISTENING
        self._state_lock = threading.Lock()

        self._current_device: int | None = current_device
        self._current_model: str = current_model
        self._current_mode: str = current_mode

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

    # ── Internal helpers ────────────────────────────────────────────

    def _image_for(self, state: DictationState) -> object:
        return self._images.get(state) or self._images.get(DictationState.LISTENING)

    def _current_state(self) -> DictationState:
        with self._state_lock:
            return self._state

    def _update_menu(self) -> None:
        """Re-evaluate dynamic menu state (radio buttons, labels)."""
        icon = self._icon
        if icon is not None:
            icon.update_menu()

    def _rebuild_menu(self) -> None:
        """Rebuild the entire menu (e.g. when the device list may have changed)."""
        icon = self._icon
        if icon is not None:
            icon.menu = self._build_menu()
            icon.update_menu()

    # ── Selection handlers (tray click → update state → fire callback) ──

    def _select_device(self, device_index: int | None) -> None:
        if self._current_device == device_index:
            return
        self._current_device = device_index
        try:
            self._on_select_device(device_index)
        except Exception:
            logger.exception("device selection callback raised")
        self._rebuild_menu()  # device list may have changed

    def _select_model(self, model_name: str) -> None:
        if self._current_model == model_name:
            return
        self._current_model = model_name
        try:
            self._on_select_model(model_name)
        except Exception:
            logger.exception("model selection callback raised")
        self._update_menu()

    def _select_mode(self, mode: str) -> None:
        if self._current_mode == mode:
            return
        self._current_mode = mode
        try:
            self._on_switch_mode(mode)
        except Exception:
            logger.exception("mode switch callback raised")
        self._update_menu()

    # ── Menu construction ───────────────────────────────────────────

    def _build_menu(self) -> "pystray.Menu":
        import pystray

        # ── Status (read-only) ──────────────────────────────────────
        def status_text(_item: object) -> str:
            return f"Status: {_STATE_LABEL[self._current_state()]}"

        # ── Pause / Resume ──────────────────────────────────────────
        def pause_text(_item: object) -> str:
            return "Resume" if self._current_state() == DictationState.SLEEPING else "Pause"

        def on_pause_clicked(_icon: object, _item: object) -> None:
            try:
                self._on_toggle_pause()
            except Exception:
                logger.exception("pause callback raised")

        # ── Audio Device submenu ────────────────────────────────────
        def _dev_action(i: int | None) -> Callable:
            def _cb(_icon: object, _item: object) -> None:
                self._select_device(i)
            return _cb

        def _dev_checked(i: int | None) -> Callable:
            def _cb(_item: object) -> bool:
                return self._current_device == i
            return _cb

        devices = _get_input_devices()
        device_items: list = [
            pystray.MenuItem(
                "Default",
                _dev_action(None),
                checked=_dev_checked(None),
                radio=True,
            ),
        ]
        if devices:
            device_items.append(pystray.Menu.SEPARATOR)
            for idx, name, is_default in devices:
                label = f"{name} (system default)" if is_default else name
                device_items.append(pystray.MenuItem(
                    label,
                    _dev_action(idx),
                    checked=_dev_checked(idx),
                    radio=True,
                ))

        # ── Model submenu ───────────────────────────────────────────
        def _model_action(m: str) -> Callable:
            def _cb(_icon: object, _item: object) -> None:
                self._select_model(m)
            return _cb

        def _model_checked(m: str) -> Callable:
            def _cb(_item: object) -> bool:
                return self._current_model == m
            return _cb

        model_items = []
        for model_name in _TRAY_MODELS:
            model_items.append(pystray.MenuItem(
                model_name,
                _model_action(model_name),
                checked=_model_checked(model_name),
                radio=True,
            ))

        # ── Mode submenu ────────────────────────────────────────────
        def _mode_action(m: str) -> Callable:
            def _cb(_icon: object, _item: object) -> None:
                self._select_mode(m)
            return _cb

        def _mode_checked(m: str) -> Callable:
            def _cb(_item: object) -> bool:
                return self._current_mode == m
            return _cb

        mode_items = [
            pystray.MenuItem(
                "Live (always listening)",
                _mode_action("live"),
                checked=_mode_checked("live"),
                radio=True,
            ),
            pystray.MenuItem(
                "Hotkey (push to talk)",
                _mode_action("hotkey"),
                checked=_mode_checked("hotkey"),
                radio=True,
            ),
        ]

        # ── Quit ────────────────────────────────────────────────────
        def on_quit_clicked(_icon: object, _item: object) -> None:
            try:
                self._on_quit()
            except Exception:
                logger.exception("quit callback raised")

        return pystray.Menu(
            pystray.MenuItem(status_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(pause_text, on_pause_clicked),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Audio Device", pystray.Menu(*device_items)),
            pystray.MenuItem("Model", pystray.Menu(*model_items)),
            pystray.MenuItem("Mode", pystray.Menu(*mode_items)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Edit Phrasebook",
                lambda _icon, _item: self._on_open_phrasebook(),
            ),
            pystray.MenuItem("Quit", on_quit_clicked),
        )
