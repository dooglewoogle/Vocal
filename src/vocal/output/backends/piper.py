"""Piper backend (``piper-tts`` >= 1.7, ONNX, CPU, embeds espeak-ng).

``model`` for :meth:`load` is the ``.onnx`` file (its ``.onnx.json`` must
sit alongside) or a directory containing exactly one such pair.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from vocal.output.backends.base import BackendUnavailable, Synthesis, TTSBackend, find_one

logger = logging.getLogger(__name__)


class PiperBackend(TTSBackend):
    name = "piper"

    def __init__(self) -> None:
        super().__init__()
        self._voice = None
        self._sample_rate = 22050
        self._loaded = False

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("piper") is not None

    def load(self, model: Path | None, style: str | None = None) -> None:
        if model is None:
            raise ValueError("Piper needs a model path")
        try:
            from piper import PiperVoice
        except ImportError as e:  # pragma: no cover - exercised via is_available
            raise BackendUnavailable("pip install 'vocal[tts-piper]'") from e

        onnx = find_one(model, ".onnx") if model.is_dir() else model
        config = onnx.with_name(onnx.name + ".json")
        if not config.exists():
            raise FileNotFoundError(f"Piper config missing: {config}")

        logger.info("Loading Piper voice %s", onnx.name)
        self._voice = PiperVoice.load(str(onnx), config_path=str(config))
        self._sample_rate = int(self._voice.config.sample_rate)
        self._loaded = True

    def synthesize(self, text: str) -> Synthesis:
        assert self._voice is not None
        from piper import SynthesisConfig

        syn = SynthesisConfig(length_scale=1.0 / max(self.speed, 0.1))

        def _chunks() -> Iterator[np.ndarray]:
            for chunk in self._voice.synthesize(text, syn_config=syn):
                yield np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)

        return Synthesis(sample_rate=self._sample_rate, chunks=_chunks())

    def unload(self) -> None:
        self._voice = None
        self._loaded = False
