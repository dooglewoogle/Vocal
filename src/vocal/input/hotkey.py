"""Global hotkey listener via evdev (Linux) or pynput (cross-platform).

Hold-to-talk only: ``on_start`` fires on key down, ``on_stop`` on key up.
"""

from __future__ import annotations

import importlib
import logging
import select
import sys
import threading
from collections.abc import Callable

from vocal.config import HotkeyConfig

logger = logging.getLogger(__name__)


class EvdevHotkeyListener:
    """Listen for a global hotkey using python-evdev (reads /dev/input directly)."""

    def __init__(
        self,
        config: HotkeyConfig,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        self._config = config
        self._on_start = on_start
        self._on_stop = on_stop
        self._recording = False
        self._running = False

    def _find_keyboards(self) -> list:
        """Find all keyboard devices that have letter keys."""
        import evdev

        keyboards = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                caps = dev.capabilities(verbose=False)
                # EV_KEY = 1; check for letter keys (KEY_A=30 through KEY_Z=44)
                if 1 in caps:
                    key_codes = caps[1]
                    if any(30 <= k <= 44 for k in key_codes):
                        logger.info("Found keyboard: %s (%s)", dev.name, dev.path)
                        keyboards.append(dev)
                    else:
                        dev.close()
                else:
                    dev.close()
            except (PermissionError, OSError) as e:
                logger.debug("Cannot open %s: %s", path, e)

        if not keyboards:
            raise RuntimeError(
                "No keyboard devices found. Ensure you are in the 'input' group: "
                "sudo usermod -aG input $USER (then re-login)"
            )

        return keyboards

    def _resolve_key_code(self) -> int:
        """Convert a key name like 'PAUSE' to an evdev key code."""
        import evdev.ecodes as ecodes

        name = f"KEY_{self._config.key.upper()}"
        code = getattr(ecodes, name, None)
        if code is None:
            raise ValueError(
                f"Unknown key name: {self._config.key!r}. "
                f"Use evdev key names without the KEY_ prefix (e.g., PAUSE, F18, SCROLLLOCK)"
            )
        return code

    def run(self) -> None:
        """Block and listen for hotkey events. Call from main thread."""
        keyboards = self._find_keyboards()
        key_code = self._resolve_key_code()
        self._running = True

        logger.info(
            "Listening for %s (code=%d) on %d device(s)",
            self._config.key, key_code, len(keyboards),
        )

        try:
            while self._running:
                if not keyboards:
                    logger.error("All keyboard devices lost — stopping listener")
                    break
                r, _, _ = select.select(keyboards, [], [], 0.5)
                for dev in r:
                    try:
                        for event in dev.read():
                            if event.type == 1 and event.code == key_code:
                                self._handle_event(event.value)
                    except OSError:
                        logger.warning("Lost device: %s", dev.path)
                        try:
                            dev.close()
                        except Exception:
                            pass
                        keyboards.remove(dev)
        finally:
            for dev in keyboards:
                try:
                    dev.close()
                except Exception:
                    pass

    def _handle_event(self, value: int) -> None:
        """Hold-to-talk. value: 0=up, 1=down, 2=hold/repeat (ignored)."""
        if value == 1:  # key down
            if not self._recording:
                self._recording = True
                logger.info("Hotkey DOWN — starting recording")
                self._on_start()
        elif value == 0:  # key up
            if self._recording:
                self._recording = False
                logger.info("Hotkey UP — stopping recording")
                self._on_stop()

    def stop(self) -> None:
        """Signal the listener to stop."""
        self._running = False


class PynputHotkeyListener:
    """Cross-platform hotkey listener using pynput (Linux/X11, macOS/Cocoa, Windows/Win32)."""

    def __init__(
        self,
        config: HotkeyConfig,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        self._config = config
        self._on_start = on_start
        self._on_stop = on_stop
        self._recording = False
        self._listener = None
        self._stop = threading.Event()

    def _resolve_key(self):
        """Convert key name to pynput Key or KeyCode."""
        from pynput.keyboard import Key

        name = self._config.key.lower()
        # Try as a named key first (pause, f1, etc.)
        key = getattr(Key, name, None)
        if key is not None:
            return key
        # Try as a character
        if len(name) == 1:
            from pynput.keyboard import KeyCode
            return KeyCode.from_char(name)
        raise ValueError(f"Unknown pynput key: {self._config.key!r}")

    def run(self) -> None:
        """Block and listen for hotkey events."""
        from pynput.keyboard import Listener

        target_key = self._resolve_key()

        def on_press(key):
            if key == target_key and not self._recording:
                self._recording = True
                self._on_start()

        def on_release(key):
            if key == target_key and self._recording:
                self._recording = False
                self._on_stop()

        self._listener = Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        self._listener.join()
        if self._stop.is_set():
            return
        # pynput's thread ends silently when the OS refuses its event tap (macOS
        # without Input Monitoring). Returning would shut the engine down, so
        # keep blocking: live dictation still works without the hotkey.
        if sys.platform == "darwin":
            from vocal.macos_perms import responsible_app

            logger.error("macOS refused the keyboard listener, so the hotkey is disabled. Allow %s in "
                         "Input Monitoring and Accessibility, or run 'vocal permissions'.", responsible_app())
        else:
            logger.error("The pynput keyboard listener stopped unexpectedly; the hotkey is disabled.")
        self._stop.wait()

    def stop(self) -> None:
        """Signal the listener to stop."""
        self._stop.set()
        if self._listener is not None:
            self._listener.stop()


class NullHotkeyListener:
    """Used when no backend is importable: never fires, blocks until stopped."""

    def __init__(self, config: HotkeyConfig, on_start: Callable[[], None], on_stop: Callable[[], None]) -> None:
        self._stop = threading.Event()

    def run(self) -> None:
        self._stop.wait()

    def stop(self) -> None:
        self._stop.set()


INSTALL_HINT = (
    "pip install 'vocal[hotkey]' (Linux: needs python3-dev and a C compiler, and your user in the "
    "'input' group)"
)


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def available_backends() -> list[str]:
    """Backends whose Python package is importable, best first."""
    out: list[str] = []
    if sys.platform == "linux" and _importable("evdev"):
        out.append("evdev")
    if _importable("pynput"):
        out.append("pynput")
    return out


def create_listener(
    config: HotkeyConfig,
    on_start: Callable[[], None],
    on_stop: Callable[[], None],
) -> EvdevHotkeyListener | PynputHotkeyListener | NullHotkeyListener:
    """Create the best available hotkey listener, or a no-op one with a clear warning."""
    from vocal.utils import is_wayland

    available = available_backends()
    backend = config.backend
    if backend == "auto":
        backend = available[0] if available else "none"
        logger.info("Auto-detected hotkey backend: %s", backend)
    elif backend not in ("evdev", "pynput"):
        raise ValueError(f"Unknown hotkey backend: {config.backend!r}")
    elif backend not in available:
        logger.warning("Hotkey backend %r requested but not installed", backend)
        backend = "none"

    if backend == "evdev":
        return EvdevHotkeyListener(config, on_start, on_stop)
    if backend == "pynput":
        if is_wayland():
            logger.warning("pynput cannot capture keys under Wayland; the hotkey will likely not work — "
                           "use the evdev backend (%s)", INSTALL_HINT)
        return PynputHotkeyListener(config, on_start, on_stop)
    logger.warning("No hotkey backend installed: the hotkey does nothing. Live dictation still works. "
                   "To enable it: %s", INSTALL_HINT)
    return NullHotkeyListener(config, on_start, on_stop)
