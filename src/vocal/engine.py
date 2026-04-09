"""DictationEngine — hotkey-driven dictation via audio capture and transcription."""

from __future__ import annotations

import enum
import logging
import threading

from vocal.audio import AudioBuffer, AudioCapture
from vocal.base_engine import BaseDictationEngine
from vocal.config import VocalConfig
from vocal.hotkey import create_listener
from vocal.phrasebook import Phrasebook

logger = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


class DictationEngine(BaseDictationEngine):
    """Hotkey-driven dictation: press to record, release to transcribe."""

    def __init__(
        self,
        config: VocalConfig,
        phrasebook: Phrasebook | None = None,
        phrasebook_seed: bool = False,
        phrasebook_replace: bool = False,
    ) -> None:
        super().__init__(config, phrasebook, phrasebook_seed, phrasebook_replace)
        self._state = State.IDLE
        self._state_lock = threading.Lock()

        # Audio
        self._buffer = AudioBuffer(sample_rate=config.audio.sample_rate)
        self._audio = AudioCapture(config.audio, self._buffer)

        # Hotkey
        self._listener = create_listener(
            config.hotkey,
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop,
        )

    # ── State management ────────────────────────────────────────────

    def _set_state(self, state: State) -> None:
        with self._state_lock:
            self._state = state
            logger.debug("State \u2192 %s", state.value)

    def _on_transcription_complete(self) -> None:
        self._set_state(State.IDLE)

    # ── Recording callbacks ─────────────────────────────────────────

    def _on_recording_start(self) -> None:
        """Called by hotkey listener when recording should begin."""
        with self._state_lock:
            if self._state != State.IDLE:
                logger.warning("Cannot start recording in state %s", self._state.value)
                return
            self._buffer.clear()
            self._audio.recording = True
            self._state = State.RECORDING

        print("\U0001f399  Recording...", flush=True)

    def _on_recording_stop(self) -> None:
        """Called by hotkey listener when recording should end."""
        with self._state_lock:
            if self._state != State.RECORDING:
                logger.warning("Cannot stop recording in state %s", self._state.value)
                return
            self._audio.recording = False
            audio = self._buffer.flush()
            self._state = State.TRANSCRIBING

        duration = audio.size / self._config.audio.sample_rate
        if duration < 0.5:
            print("\u23ed  Too short, skipping.", flush=True)
            self._set_state(State.IDLE)
            return

        print(f"\u23f3 Transcribing {duration:.1f}s of audio...", flush=True)
        self._transcription_queue.put(audio)

    # ── Resource cleanup ────────────────────────────────────────────

    def _cleanup_resources(self) -> None:
        self._listener.stop()
        self._audio.recording = False
        self._audio.stop()

    # ── Entry point ─────────────────────────────────────────────────

    def run(self) -> None:
        """Start all threads and block on the hotkey listener (main thread)."""
        self._transcriber.load()
        self._audio.start()
        self._start_workers()
        self._install_signal_handlers()

        print(
            f"\nVocal ready \u2014 press {self._config.hotkey.key} "
            f"({self._config.hotkey.mode} mode) to dictate.\n",
            flush=True,
        )

        try:
            self._listener.run()
        except Exception:
            logger.exception("Hotkey listener crashed")
        finally:
            self.shutdown()
