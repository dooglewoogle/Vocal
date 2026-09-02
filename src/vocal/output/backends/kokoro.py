"""Kokoro backend (``kokoro-onnx`` >= 0.6, 82M params, ONNX, CPU-capable).

``model`` for :meth:`load` is a directory containing ``kokoro*.onnx`` and
``voices*.bin``, or the ``.onnx`` file with the voices file alongside.
``style`` is the speaker name, e.g. ``af_sarah``.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from vocal.output.backends.base import (
    BackendUnavailable,
    Synthesis,
    TTSBackend,
    find_one,
    float_to_int16,
)

logger = logging.getLogger(__name__)

DEFAULT_STYLE = "af_sarah"


class KokoroBackend(TTSBackend):
    name = "kokoro"

    def __init__(self) -> None:
        super().__init__()
        self._kokoro = None
        self._style = DEFAULT_STYLE
        self._loaded = False

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("kokoro_onnx") is not None

    def load(self, model: Path | None, style: str | None = None) -> None:
        if model is None:
            raise ValueError("Kokoro needs a model path")
        try:
            from kokoro_onnx import Kokoro
        except ImportError as e:  # pragma: no cover
            raise BackendUnavailable("pip install 'vocal[tts-kokoro]'") from e

        directory = model if model.is_dir() else model.parent
        onnx = model if model.is_file() else find_one(directory, ".onnx")
        voices = find_one(directory, ".bin")

        logger.info("Loading Kokoro %s (voices %s)", onnx.name, voices.name)
        self._kokoro = Kokoro(str(onnx), str(voices))
        self._style = style or DEFAULT_STYLE
        self._loaded = True

    def synthesize(self, text: str) -> Synthesis:
        assert self._kokoro is not None
        samples, rate = self._kokoro.create(text, voice=self._style, speed=self.speed, lang="en-us")
        return Synthesis(sample_rate=int(rate), chunks=iter([float_to_int16(samples)]))

    def unload(self) -> None:
        self._kokoro = None
        self._loaded = False
