"""Low-latency PCM playback through sounddevice."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

_FRAME_MS = 20  # write granularity; bounds how quickly abort() takes effect


def list_output_devices() -> list[tuple[int, str, bool]]:
    """``(index, name, is_default)`` for every device with output channels."""
    default = sd.default.device[1]
    return [
        (i, d["name"], i == default)
        for i, d in enumerate(sd.query_devices())
        if d["max_output_channels"] > 0
    ]


def resolve_output_device(device: str | None) -> int | None:
    """Resolve a name substring or index string to a device index (None = default)."""
    if not device:
        return None
    try:
        return int(device)
    except ValueError:
        needle = device.lower()
        for i, name, _ in list_output_devices():
            if needle in name.lower():
                return i
        logger.warning("Output device %r not found, using default", device)
        return None


class AudioPlayer:
    """Plays int16 mono chunks. One ``OutputStream`` per :meth:`play` call,
    opened at the chunks' native rate — no resampling needed.

    Thread-safe: :meth:`abort` may be called from any thread while
    :meth:`play` is blocking on another.
    """

    def __init__(self, device: str | None = None) -> None:
        self._device = device
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()
        self._abort = threading.Event()

    def set_device(self, device: str | None) -> None:
        self._device = device

    def play(
        self,
        chunks: Iterable[np.ndarray],
        sample_rate: int,
        gain: float = 1.0,
        on_first_audio: Callable[[], None] | None = None,
    ) -> bool:
        """Block until all chunks have played. Returns False if aborted."""
        self._abort.clear()
        frame = max(1, sample_rate * _FRAME_MS // 1000)
        stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="int16",
            device=resolve_output_device(self._device),
        )
        with self._lock:
            self._stream = stream
        started = False
        try:
            stream.start()
            for chunk in chunks:
                if self._abort.is_set():
                    return False
                if chunk.size == 0:
                    continue
                if gain != 1.0:
                    chunk = np.clip(chunk.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
                if not started:
                    started = True
                    if on_first_audio:
                        on_first_audio()
                for i in range(0, chunk.size, frame):
                    if self._abort.is_set():
                        return False
                    stream.write(np.ascontiguousarray(chunk[i:i + frame]))
            if self._abort.is_set():
                return False
            stream.stop()  # drain
            return True
        except sd.PortAudioError as e:
            if self._abort.is_set():
                return False
            logger.error("Playback failed: %s", e)
            return False
        finally:
            with self._lock:
                self._stream = None
            try:
                if stream.active:
                    stream.abort()
                stream.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def abort(self) -> None:
        """Stop immediately, discarding buffered audio."""
        self._abort.set()
        with self._lock:
            stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:  # pragma: no cover
                pass
