"""Audio capture via sounddevice with a thread-safe buffer."""

from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

from vocal.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Thread-safe accumulator for audio chunks."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._sample_rate = sample_rate

    def append(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(chunk)

    def flush(self) -> np.ndarray:
        """Return concatenated audio and clear the buffer."""
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)
            result = np.concatenate(self._chunks)
            self._chunks.clear()
            return result

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()

    @property
    def duration_seconds(self) -> float:
        """Approximate duration of buffered audio."""
        with self._lock:
            return sum(c.shape[0] for c in self._chunks) / self._sample_rate


def resolve_device(device: str | None) -> int | None:
    """Resolve a device name or index string to a sounddevice device index."""
    if not device:
        return None
    try:
        return int(device)
    except ValueError:
        for i, dev in enumerate(sd.query_devices()):
            if device.lower() in dev["name"].lower():
                return i
        logger.warning("Device %r not found, using default", device)
        return None


class AudioCapture:
    """Manages a sounddevice InputStream and feeds an AudioBuffer."""

    def __init__(self, config: AudioConfig, buffer: AudioBuffer) -> None:
        self._config = config
        self._buffer = buffer
        self._recording = False
        self._stream: sd.InputStream | None = None

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio status: %s", status)
        if self._recording:
            self._buffer.append(indata[:, 0].copy())

    def start(self) -> None:
        """Open the audio stream (does not begin recording to buffer)."""
        device_index = resolve_device(self._config.device)

        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._config.block_size,
            callback=self._callback,
            device=device_index,
        )
        self._stream.start()
        logger.info("Audio stream opened (device=%s, rate=%d)", device_index, self._config.sample_rate)

    def stop(self) -> None:
        """Close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def recording(self) -> bool:
        return self._recording

    @recording.setter
    def recording(self, value: bool) -> None:
        self._recording = value
