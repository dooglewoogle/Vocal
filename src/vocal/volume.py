"""System output volume control and ducking.

Ducking lowers the master output volume while recording so playback does
not bleed into the microphone, then ramps it back once recording stops.

Backends shell out to whatever volume CLI is available:
  Linux  — pactl (PulseAudio / PipeWire-pulse), wpctl (PipeWire), amixer (ALSA)
  macOS  — osascript
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import threading
import time

logger = logging.getLogger(__name__)

_TIMEOUT_S = 1.0


def _run(cmd: list[str]) -> str | None:
    """Run a command; return stdout on success, None on any failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT_S, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("Volume command failed: %s (%s)", " ".join(cmd), e)
        return None
    if result.returncode != 0:
        logger.warning(
            "Volume command failed: %s (rc=%d) %s",
            " ".join(cmd), result.returncode, result.stderr.strip(),
        )
        return None
    return result.stdout


def _clamp(percent: int) -> int:
    return max(0, min(100, int(percent)))


# ── Backends ────────────────────────────────────────────────────────


class VolumeBackend:
    """Base for volume backends. Subclasses set ``tool`` and the command lists."""

    tool: str = ""

    def get(self) -> int | None:
        """Current master output volume as 0–100, or None if unavailable."""
        out = _run(self._get_cmd())
        if out is None:
            return None
        level = self._parse(out)
        if level is None:
            logger.warning("%s: could not parse volume from %r", self.tool, out.strip())
        return level

    def set(self, percent: int) -> bool:
        """Set master output volume to ``percent`` (0–100). Returns success."""
        return _run(self._set_cmd(_clamp(percent))) is not None

    def _get_cmd(self) -> list[str]:
        raise NotImplementedError

    def _set_cmd(self, percent: int) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def _parse(output: str) -> int | None:
        raise NotImplementedError


class PactlBackend(VolumeBackend):
    tool = "pactl"

    def _get_cmd(self) -> list[str]:
        return ["pactl", "get-sink-volume", "@DEFAULT_SINK@"]

    def _set_cmd(self, percent: int) -> list[str]:
        return ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"]

    @staticmethod
    def _parse(output: str) -> int | None:
        # "Volume: front-left: 65536 / 100% / 0.00 dB, front-right: ..."
        m = re.search(r"(\d+)%", output)
        return int(m.group(1)) if m else None


class WpctlBackend(VolumeBackend):
    tool = "wpctl"

    def _get_cmd(self) -> list[str]:
        return ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"]

    def _set_cmd(self, percent: int) -> list[str]:
        return ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{percent}%"]

    @staticmethod
    def _parse(output: str) -> int | None:
        # "Volume: 0.65" or "Volume: 0.65 [MUTED]"
        m = re.search(r"Volume:\s*([0-9]*\.?[0-9]+)", output)
        return round(float(m.group(1)) * 100) if m else None


class AmixerBackend(VolumeBackend):
    tool = "amixer"

    def _get_cmd(self) -> list[str]:
        return ["amixer", "get", "Master"]

    def _set_cmd(self, percent: int) -> list[str]:
        return ["amixer", "set", "Master", f"{percent}%"]

    @staticmethod
    def _parse(output: str) -> int | None:
        # "  Front Left: Playback 65536 [100%] [0.00dB] [on]"
        m = re.search(r"\[(\d+)%\]", output)
        return int(m.group(1)) if m else None


class OsascriptBackend(VolumeBackend):
    tool = "osascript"

    def _get_cmd(self) -> list[str]:
        return ["osascript", "-e", "output volume of (get volume settings)"]

    def _set_cmd(self, percent: int) -> list[str]:
        return ["osascript", "-e", f"set volume output volume {percent}"]

    @staticmethod
    def _parse(output: str) -> int | None:
        m = re.search(r"(\d+)", output)
        return int(m.group(1)) if m else None


_LINUX_BACKENDS: tuple[type[VolumeBackend], ...] = (PactlBackend, WpctlBackend, AmixerBackend)
_DARWIN_BACKENDS: tuple[type[VolumeBackend], ...] = (OsascriptBackend,)


def candidate_tools() -> list[str]:
    """Names of the CLI tools tried on this platform, in order."""
    if sys.platform == "linux":
        return [b.tool for b in _LINUX_BACKENDS]
    if sys.platform == "darwin":
        return [b.tool for b in _DARWIN_BACKENDS]
    return []


def detect_backend() -> VolumeBackend | None:
    """Return the first available volume backend for this platform, or None."""
    if sys.platform == "linux":
        candidates = _LINUX_BACKENDS
    elif sys.platform == "darwin":
        candidates = _DARWIN_BACKENDS
    else:
        return None

    for cls in candidates:
        if shutil.which(cls.tool) is not None:
            logger.debug("Volume backend: %s", cls.tool)
            return cls()
    return None


# ── Ducker ──────────────────────────────────────────────────────────


class Ducker:
    """Lower master volume by a relative percentage, then ramp it back.

    ``duck()`` is synchronous (one get + one set). ``restore()`` returns
    immediately and ramps on a background thread. A ``duck()`` during a
    ramp cancels it and re-ducks from the originally saved level, so the
    restore target never drifts across rapid press/release cycles.
    """

    def __init__(
        self,
        amount: int,
        backend: VolumeBackend,
        ramp_ms: int = 300,
        steps: int = 6,
    ) -> None:
        clamped = _clamp(amount)
        if clamped != amount:
            logger.warning("duck_amount %d out of range; clamped to %d", amount, clamped)
        self._amount = clamped
        self._backend = backend
        self._ramp_ms = max(0, ramp_ms)
        self._steps = max(1, steps)

        self._lock = threading.Lock()
        self._saved: int | None = None      # level to restore to; None = not ducked
        self._ducked_level: int | None = None
        self._ramp_thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._warned_unavailable = False

    @property
    def is_ducked(self) -> bool:
        with self._lock:
            return self._saved is not None

    # ── Public API ──────────────────────────────────────────────────

    def duck(self) -> None:
        """Drop volume immediately. Idempotent while already ducked."""
        self._cancel_ramp()
        with self._lock:
            if self._saved is None:
                current = self._backend.get()
                if current is None:
                    if not self._warned_unavailable:
                        logger.warning("Ducking disabled: cannot read system volume")
                        self._warned_unavailable = True
                    return
                self._saved = current
            target = round(self._saved * (100 - self._amount) / 100)
            if self._ducked_level == target:
                return
            self._ducked_level = target
            logger.debug("Duck %d%% -> %d%%", self._saved, target)
            self._backend.set(target)

    def restore(self) -> None:
        """Ramp volume back to the saved level on a background thread."""
        with self._lock:
            if self._saved is None:
                return
            if self._ramp_thread is not None and self._ramp_thread.is_alive():
                return  # ramp already in flight
            start = self._ducked_level if self._ducked_level is not None else self._saved
            target = self._saved
            self._cancel.clear()
            self._ramp_thread = threading.Thread(
                target=self._ramp, args=(start, target),
                name="volume-ramp", daemon=True,
            )
            self._ramp_thread.start()

    def close(self) -> None:
        """Cancel any ramp and snap back to the saved level immediately."""
        self._cancel_ramp()
        with self._lock:
            if self._saved is not None:
                self._backend.set(self._saved)
                self._saved = None
                self._ducked_level = None

    # ── Internals ───────────────────────────────────────────────────

    def _cancel_ramp(self) -> None:
        with self._lock:
            thread = self._ramp_thread
            if thread is None or not thread.is_alive():
                return
            self._cancel.set()
            # The ramp moved the volume off the ducked level; force a re-set.
            self._ducked_level = None
        thread.join(timeout=0.5)

    def _ramp(self, start: int, target: int) -> None:
        delay = self._ramp_ms / 1000 / self._steps
        for i in range(1, self._steps + 1):
            if self._cancel.is_set():
                return
            level = round(start + (target - start) * i / self._steps)
            self._backend.set(level)
            if i < self._steps:
                time.sleep(delay)
        with self._lock:
            # Only clear if nobody re-ducked while we were finishing.
            if not self._cancel.is_set():
                logger.debug("Restored volume to %d%%", target)
                self._saved = None
                self._ducked_level = None
