"""Global hotkey listener via evdev (Linux) or pynput (cross-platform)."""

from __future__ import annotations

import logging
import select
import sys
from typing import Callable

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
        mode = self._config.mode
        self._running = True

        logger.info(
            "Listening for %s (code=%d) in %s mode on %d device(s)",
            self._config.key, key_code, mode, len(keyboards),
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
                                self._handle_event(event.value, mode)
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

    def _handle_event(self, value: int, mode: str) -> None:
        """Handle a key event. value: 0=up, 1=down, 2=hold/repeat."""
        if mode == "toggle":
            if value == 1:  # key down
                if self._recording:
                    self._recording = False
                    logger.info("Toggle OFF — stopping recording")
                    self._on_stop()
                else:
                    self._recording = True
                    logger.info("Toggle ON — starting recording")
                    self._on_start()
        elif mode == "ptt":
            if value == 1:  # key down
                self._recording = True
                logger.info("PTT DOWN — starting recording")
                self._on_start()
            elif value == 0:  # key up
                if self._recording:
                    self._recording = False
                    logger.info("PTT UP — stopping recording")
                    self._on_stop()
            # value == 2 (hold/repeat) is intentionally ignored

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

        if sys.platform == "darwin":
            logger.info(
                "macOS: pynput requires Accessibility permissions. "
                "Grant access in System Settings → Privacy & Security → Accessibility."
            )

        target_key = self._resolve_key()
        mode = self._config.mode

        def on_press(key):
            if key == target_key:
                if mode == "toggle":
                    if self._recording:
                        self._recording = False
                        self._on_stop()
                    else:
                        self._recording = True
                        self._on_start()
                elif mode == "ptt":
                    if not self._recording:
                        self._recording = True
                        self._on_start()

        def on_release(key):
            if key == target_key and mode == "ptt" and self._recording:
                self._recording = False
                self._on_stop()

        self._listener = Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        self._listener.join()

    def stop(self) -> None:
        """Signal the listener to stop."""
        if self._listener is not None:
            self._listener.stop()


def _auto_detect_backend() -> str:
    """Pick the best hotkey backend for the current platform."""
    if sys.platform == "linux":
        try:
            import evdev  # noqa: F401
            return "evdev"
        except ImportError:
            logger.info("evdev not available, falling back to pynput")
            return "pynput"
    # macOS, Windows — pynput is the cross-platform option
    return "pynput"


def create_listener(
    config: HotkeyConfig,
    on_start: Callable[[], None],
    on_stop: Callable[[], None],
) -> EvdevHotkeyListener | PynputHotkeyListener:
    """Create the appropriate hotkey listener based on config."""
    backend = config.backend
    if backend == "auto":
        backend = _auto_detect_backend()
        logger.info("Auto-detected hotkey backend: %s", backend)

    if backend == "evdev":
        return EvdevHotkeyListener(config, on_start, on_stop)
    elif backend == "pynput":
        return PynputHotkeyListener(config, on_start, on_stop)
    else:
        raise ValueError(f"Unknown hotkey backend: {config.backend!r}")
