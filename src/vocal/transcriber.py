"""Whisper transcription wrapper using faster-whisper."""

from __future__ import annotations

import logging
import time

import numpy as np

from vocal.config import ModelConfig, VADConfig

logger = logging.getLogger(__name__)

VALID_MODELS = frozenset({
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    "distil-large-v2", "distil-large-v3",
    "distil-small.en", "distil-medium.en",
})


class Transcriber:
    """Wraps faster_whisper.WhisperModel for single-call transcription."""

    def __init__(self, model_config: ModelConfig, vad_config: VADConfig, sample_rate: int = 16000) -> None:
        self._model_config = model_config
        self._vad_config = vad_config
        self._sample_rate = sample_rate
        self._model = None

    def load(self) -> None:
        """Load the Whisper model. Call once at startup."""
        from faster_whisper import WhisperModel

        model_name = self._model_config.size
        if model_name not in VALID_MODELS:
            raise ValueError(
                f"Unknown model: {model_name!r}. "
                f"Valid models: {', '.join(sorted(VALID_MODELS))}"
            )

        cpu_threads = self._model_config.cpu_threads or 0
        logger.info(
            "Loading model %s (compute=%s, threads=%s)...",
            self._model_config.size,
            self._model_config.compute_type,
            cpu_threads or "auto",
        )
        t0 = time.monotonic()
        self._model = WhisperModel(
            self._model_config.size,
            device="cpu",
            compute_type=self._model_config.compute_type,
            cpu_threads=cpu_threads,
        )
        elapsed = time.monotonic() - t0
        logger.info("Model loaded in %.1fs", elapsed)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 audio array (16 kHz mono). Returns text."""
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        if audio.size == 0:
            return ""

        duration = audio.size / self._sample_rate
        if duration < 0.5:
            logger.debug("Audio too short (%.2fs), skipping", duration)
            return ""

        logger.debug("Transcribing %.1fs of audio...", duration)
        t0 = time.monotonic()

        vad_params = {}
        if self._vad_config.enabled:
            vad_params = dict(
                threshold=self._vad_config.threshold,
                min_silence_duration_ms=self._vad_config.min_silence_duration_ms,
                speech_pad_ms=self._vad_config.speech_pad_ms,
            )

        segments, info = self._model.transcribe(
            audio,
            language=self._model_config.language,
            beam_size=self._model_config.beam_size,
            vad_filter=self._vad_config.enabled,
            vad_parameters=vad_params or None,
        )

        # Must fully consume the lazy generator
        text = " ".join(seg.text for seg in segments)

        elapsed = time.monotonic() - t0
        rtf = elapsed / duration if duration > 0 else 0
        logger.info("Transcribed in %.2fs (RTF=%.2f): %r", elapsed, rtf, text[:80])

        return text
