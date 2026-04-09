"""Streaming VAD using faster-whisper's bundled Silero VAD v6 ONNX model."""

from __future__ import annotations

import numpy as np


WINDOW_SAMPLES = 512       # 32ms @ 16kHz
CONTEXT_SAMPLES = 64       # context prefix expected by the ONNX model
INPUT_SIZE = WINDOW_SAMPLES + CONTEXT_SAMPLES  # 576


class StreamingVAD:
    """Per-window speech probability inference with persistent LSTM state.

    Uses the Silero VAD v6 ONNX session bundled with faster-whisper.
    Call process_window() with exactly 512 samples to get a speech probability.
    Hidden states persist across calls — call reset() between utterances.
    """

    def __init__(self) -> None:
        from faster_whisper.vad import get_vad_model

        self._session = get_vad_model().session
        self._h = np.zeros((1, 1, 128), dtype="float32")
        self._c = np.zeros((1, 1, 128), dtype="float32")
        self._context = np.zeros(CONTEXT_SAMPLES, dtype="float32")

    def reset(self) -> None:
        """Reset hidden states and context for next utterance."""
        self._h[:] = 0
        self._c[:] = 0
        self._context[:] = 0

    def process_window(self, samples: np.ndarray) -> float:
        """Process exactly 512 float32 samples. Returns speech probability [0, 1]."""
        assert samples.shape == (WINDOW_SAMPLES,), (
            f"Expected {WINDOW_SAMPLES} samples, got {samples.shape}"
        )

        # Build input: 64 context + 512 window = 576
        inp = np.concatenate([self._context, samples]).reshape(1, INPUT_SIZE)

        output, self._h, self._c = self._session.run(
            None,
            {"input": inp, "h": self._h, "c": self._c},
        )

        # Update context to the last 64 samples of this window
        self._context = samples[-CONTEXT_SAMPLES:].copy()

        return float(output[0])


class SpeechDetector:
    """State machine that detects speech start/end from a stream of VAD probabilities.

    Matches the Silero convention: threshold with hysteresis via neg_threshold,
    minimum silence duration before ending, minimum speech duration filter.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        neg_threshold: float | None = None,
        min_silence_duration_ms: int = 600,
        min_speech_duration_ms: int = 250,
        sample_rate: int = 16000,
    ) -> None:
        self._threshold = threshold
        self._neg_threshold = (
            neg_threshold if neg_threshold is not None
            else max(threshold - 0.15, 0.01)
        )

        # Convert ms to window counts
        ms_per_window = (WINDOW_SAMPLES / sample_rate) * 1000  # 32ms
        self._min_silence_windows = int(min_silence_duration_ms / ms_per_window)
        self._min_speech_windows = int(min_speech_duration_ms / ms_per_window)

        self._triggered = False
        self._speech_start_window = 0
        self._temp_end: int | None = None  # window where silence began
        self._window_index = 0

    def reset(self) -> None:
        """Reset state machine for next utterance."""
        self._triggered = False
        self._speech_start_window = 0
        self._temp_end = None
        self._window_index = 0

    def process(self, speech_prob: float) -> tuple[str, int]:
        """Feed one VAD probability. Returns (event, window_index).

        Matches Silero's batch algorithm: silence is measured as elapsed
        time from the first below-neg_threshold frame. Frames between
        neg_threshold and threshold don't reset the timer — only frames
        above threshold do.

        Events:
          "speech_start" — speech just began (window_index = start position)
          "speech_end"   — speech just ended (window_index = end position)
          "continue"     — no state change
        """
        idx = self._window_index
        self._window_index += 1

        # Above threshold while silence timer is running → cancel silence
        if speech_prob >= self._threshold and self._temp_end is not None:
            self._temp_end = None

        if speech_prob >= self._threshold and not self._triggered:
            self._triggered = True
            self._speech_start_window = idx
            self._temp_end = None
            return ("speech_start", idx)

        if not self._triggered:
            return ("continue", idx)

        # Below neg_threshold → start or continue silence timer
        if speech_prob < self._neg_threshold:
            if self._temp_end is None:
                self._temp_end = idx

        # Check if enough silence has elapsed (measured from temp_end)
        if self._temp_end is not None:
            silence_windows = idx - self._temp_end
            if silence_windows >= self._min_silence_windows:
                speech_duration_windows = self._temp_end - self._speech_start_window
                self._triggered = False
                self._temp_end = None

                # Reject if speech was too short
                if speech_duration_windows < self._min_speech_windows:
                    return ("continue", idx)

                return ("speech_end", idx)

        return ("continue", idx)
