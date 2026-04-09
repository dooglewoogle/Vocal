"""LiveDictationEngine — always-on VAD-driven dictation."""

from __future__ import annotations

import logging
import queue
from collections import deque

import numpy as np
import sounddevice as sd

from vocal.audio import resolve_device
from vocal.base_engine import BaseDictationEngine
from vocal.config import VocalConfig
from vocal.vad import WINDOW_SAMPLES, SpeechDetector, StreamingVAD

logger = logging.getLogger(__name__)


class LiveDictationEngine(BaseDictationEngine):
    """Always-on dictation engine using streaming VAD to detect speech boundaries."""

    def __init__(self, config: VocalConfig) -> None:
        super().__init__(config)

        # VAD
        self._vad = StreamingVAD()
        self._detector = SpeechDetector(
            threshold=config.vad.threshold,
            min_silence_duration_ms=config.live.min_silence_duration_ms,
            min_speech_duration_ms=config.live.min_speech_duration_ms,
        )

        # Audio buffering
        sample_rate = config.audio.sample_rate
        pad_chunks = int(config.vad.speech_pad_ms * sample_rate / 1000 / WINDOW_SAMPLES)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(pad_chunks, 1))
        self._utterance_chunks: list[np.ndarray] = []
        self._in_speech = False
        self._max_speech_chunks = int(config.live.max_speech_duration_s * sample_rate / WINDOW_SAMPLES)
        self._raw_queue: queue.Queue[np.ndarray | None] = queue.Queue()

        # Audio stream (created in run())
        self._stream: sd.InputStream | None = None

    # ── Audio callback ──────────────────────────────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Sounddevice callback — RT thread. Must be fast."""
        if status:
            logger.warning("Audio status: %s", status)
        chunk = indata[:, 0].copy()
        try:
            self._raw_queue.put_nowait(chunk)
        except queue.Full:
            logger.warning("Raw audio queue full, dropping chunk")

    # ── VAD worker ──────────────────────────────────────────────────

    def _vad_worker(self) -> None:
        """Thread: consume raw audio, run VAD, detect speech boundaries."""
        window_buf = np.array([], dtype=np.float32)

        while not self._shutdown.is_set():
            try:
                chunk = self._raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                break

            if self._in_speech:
                self._utterance_chunks.append(chunk)
                if len(self._utterance_chunks) >= self._max_speech_chunks:
                    self._flush_utterance()
                    window_buf = np.array([], dtype=np.float32)
                    continue
            else:
                self._preroll.append(chunk)

            window_buf = np.concatenate([window_buf, chunk])

            while len(window_buf) >= WINDOW_SAMPLES:
                window = window_buf[:WINDOW_SAMPLES]
                window_buf = window_buf[WINDOW_SAMPLES:]

                prob = self._vad.process_window(window)
                event, _ = self._detector.process(prob)

                if event == "speech_start":
                    self._in_speech = True
                    self._utterance_chunks = list(self._preroll)
                    self._preroll.clear()
                    print("\U0001f399  Speech detected...", flush=True)

                elif event == "speech_end" and self._in_speech:
                    self._flush_utterance()

    def _flush_utterance(self) -> None:
        """Extract buffered speech audio and send to transcription."""
        if not self._utterance_chunks:
            self._in_speech = False
            return

        audio = np.concatenate(self._utterance_chunks)
        duration = audio.size / self._config.audio.sample_rate

        if duration >= 0.5:
            print(f"\u23f3 Transcribing {duration:.1f}s...", flush=True)
            self._transcription_queue.put(audio)
        else:
            logger.debug("Speech too short (%.2fs), skipping", duration)

        self._utterance_chunks.clear()
        self._in_speech = False
        self._vad.reset()
        self._detector.reset()

    # ── Overrides ───────────────────────────────────────────────────

    def _sentinel_queues(self) -> list[queue.Queue]:
        return [self._raw_queue] + super()._sentinel_queues()

    def _cleanup_resources(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ── Entry point ─────────────────────────────────────────────────

    def run(self) -> None:
        """Load model, start all threads, block until shutdown."""
        self._transcriber.load()

        self._stream = sd.InputStream(
            samplerate=self._config.audio.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=WINDOW_SAMPLES,
            callback=self._audio_callback,
            device=resolve_device(self._config.audio.device),
        )
        self._stream.start()

        self._start_workers((self._vad_worker, "vad"))
        self._install_signal_handlers()

        print(
            "\nVocal live mode \u2014 listening for speech. Press Ctrl+C to stop.\n",
            flush=True,
        )

        self._shutdown.wait()
