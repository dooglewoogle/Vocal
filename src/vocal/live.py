"""LiveDictationEngine — always-on VAD-driven dictation."""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from vocal.audio import resolve_device
from vocal.base_engine import BaseDictationEngine
from vocal.config import VocalConfig
from vocal.hotkey import create_listener
from vocal.phrasebook import Phrasebook
from vocal.state import DictationState
from vocal.vad import WINDOW_SAMPLES, SpeechDetector, StreamingVAD

logger = logging.getLogger(__name__)


class LiveDictationEngine(BaseDictationEngine):
    """Always-on dictation engine using streaming VAD to detect speech boundaries."""

    def __init__(
        self,
        config: VocalConfig,
        phrasebook: Phrasebook | None = None,
        phrasebook_seed: bool = False,
        phrasebook_replace: bool = False,
        on_state_change: Callable[[DictationState], None] | None = None,
        on_shutdown_requested: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            config, phrasebook, phrasebook_seed, phrasebook_replace,
            on_state_change=on_state_change,
            on_shutdown_requested=on_shutdown_requested,
        )

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

        # Pause support via hotkey
        # Semantics are inverted vs the main engine: the hotkey's "start
        # recording" action pauses live listening, and "stop recording"
        # resumes it.  In PTT mode this means hold-to-mute.
        self._paused = threading.Event()  # set = paused
        self._listener = create_listener(
            config.hotkey,
            on_start=self._on_pause,
            on_stop=self._on_unpause,
        )

    # ── Pause/unpause callbacks ────────────────────────────────────

    def toggle_pause(self) -> None:
        """Public toggle for external callers (tray menu, IPC, tests)."""
        if self._paused.is_set():
            self._on_unpause()
        else:
            self._on_pause()

    def _on_pause(self) -> None:
        """Called by hotkey listener to pause live listening."""
        self._paused.set()
        # Flush any in-progress utterance so partial audio isn't left dangling
        if self._in_speech:
            self._flush_utterance()
        self._idle_state = DictationState.SLEEPING
        self._set_state(DictationState.SLEEPING)
        print("\u23f8  Paused", flush=True)

    def _on_unpause(self) -> None:
        """Called by hotkey listener to resume live listening."""
        self._vad.reset()
        self._detector.reset()
        self._preroll.clear()
        self._paused.clear()
        self._idle_state = DictationState.LISTENING
        self._set_state(DictationState.LISTENING)
        print("\u25b6  Listening...", flush=True)

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
        if self._paused.is_set():
            return
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
                    self._set_state(DictationState.RECORDING)
                    print("\U0001f399  Speech detected...", flush=True)

                elif event == "speech_end" and self._in_speech:
                    self._flush_utterance()

    def _flush_utterance(self) -> None:
        """Extract buffered speech audio and send to transcription."""
        if not self._utterance_chunks:
            self._in_speech = False
            self._set_state(self._idle_state)
            return

        audio = np.concatenate(self._utterance_chunks)
        duration = audio.size / self._config.audio.sample_rate

        if duration >= 0.5:
            print(f"\u23f3 Transcribing {duration:.1f}s...", flush=True)
            self._set_state(DictationState.TRANSCRIBING)
            self._transcription_queue.put(audio)
        else:
            logger.debug("Speech too short (%.2fs), skipping", duration)
            self._set_state(self._idle_state)

        self._utterance_chunks.clear()
        self._in_speech = False
        self._vad.reset()
        self._detector.reset()

    # ── Runtime switching ─────────────────────────────────────────

    def switch_device(self, device_index: int | None) -> None:
        """Switch audio input device. Pauses briefly during the swap."""
        was_paused = self._paused.is_set()
        if not was_paused:
            self._on_pause()

        if self._stream:
            self._stream.stop()
            self._stream.close()

        self._config.audio.device = str(device_index) if device_index is not None else None

        self._stream = sd.InputStream(
            samplerate=self._config.audio.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=WINDOW_SAMPLES,
            callback=self._audio_callback,
            device=device_index,
        )
        self._stream.start()
        logger.info("Switched audio device to %s", device_index)

        if not was_paused:
            self._on_unpause()

    # ── Overrides ───────────────────────────────────────────────────

    def _sentinel_queues(self) -> list[queue.Queue]:
        return [self._raw_queue] + super()._sentinel_queues()

    def _cleanup_resources(self) -> None:
        self._listener.stop()
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

        key = self._config.hotkey.key
        mode = self._config.hotkey.mode
        print(
            f"\nVocal live mode \u2014 listening for speech. "
            f"Press {key} ({mode}) to pause/resume. Ctrl+C to stop.\n",
            flush=True,
        )

        try:
            self._listener.run()
        except Exception:
            logger.exception("Hotkey listener crashed")
        finally:
            self.shutdown()
