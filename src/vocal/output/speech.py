"""Speech controller: FIFO of utterances, sentence-streamed synthesis, interrupt."""

from __future__ import annotations

import copy
import logging
import queue
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass

from vocal.config import SpeechConfig, copy_into
from vocal.notify import notify
from vocal.output.backends import BackendUnavailable, TTSBackend, resolve_backend
from vocal.output.models import VoiceNotFoundError, get_voice, resolve_model_path
from vocal.output.playback import AudioPlayer

logger = logging.getLogger(__name__)

_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation / newlines so the first sentence can
    start playing while later ones synthesize. Whitespace-only parts dropped."""
    return [p.strip() for p in _SPLIT.split(text) if p and p.strip()]


@dataclass
class Utterance:
    text: str
    voice: str | None = None


class SpeechController:
    """Owns the TTS backend, a worker thread, and the playback queue.

    ``on_speech_start`` fires when the first audio of a speaking run is
    about to play; ``on_speech_end`` when the queue drains or is stopped.
    A "run" is any stretch of back-to-back utterances, so callers get one
    start/end pair rather than one per sentence.
    """

    def __init__(
        self,
        config: SpeechConfig,
        player: AudioPlayer | None = None,
        backend: TTSBackend | None = None,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_end: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._player = player or AudioPlayer(config.device)
        self._backend = backend  # injected for tests; otherwise resolved lazily
        self._backend_voice: str | None = None  # voice currently loaded
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self._queue: queue.Queue[Utterance | None] = queue.Queue()
        self._abort = threading.Event()  # drop current utterance
        self._speaking = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._voice_lock = threading.Lock()
        self._state_lock = threading.Lock()  # guards _busy / _idle vs queue
        self._busy = False  # worker is processing an utterance
        self._thread: threading.Thread | None = None
        self._shutdown = False

    # ── Public API ───────────────────────────────────────────────────

    @property
    def voice(self) -> str:
        return self._config.voice

    @property
    def backend_name(self) -> str:
        if self._backend is not None:
            return self._backend.name
        try:
            return get_voice(self._config.voice).backend
        except VoiceNotFoundError:
            return self._config.backend

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def queue_length(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker, name="tts-worker", daemon=True)
        self._thread.start()

    def say(self, text: str, interrupt: bool = False, voice: str | None = None) -> None:
        """Enqueue ``text``. ``interrupt`` flushes everything first."""
        text = text.strip()
        if not text:
            return
        if interrupt:
            self.stop()
        self.start()
        with self._state_lock:
            self._idle.clear()
            self._queue.put(Utterance(text, voice))

    def stop(self) -> None:
        """Flush the queue and halt playback. Blocks until the worker is idle
        unless called from the worker itself."""
        self._drain_queue()
        self._abort.set()
        self._player.abort()
        with self._state_lock:
            if not self._busy and self._queue.empty():
                self._idle.set()
        if threading.current_thread() is not self._thread:
            self._idle.wait(timeout=5.0)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been spoken (or stopped)."""
        return self._idle.wait(timeout)

    def set_voice(self, name: str) -> None:
        """Switch voice (downloading if needed) on the worker thread."""
        get_voice(name)  # validate now so callers get an immediate error
        new = copy.copy(self._config)
        new.voice = name
        self.apply_config(new)

    def apply_config(self, new: SpeechConfig) -> None:
        """Adopt ``new`` in place on this controller (other components hold a
        reference to it and to our config object).

        Voice / backend / model_path changes invalidate the loaded voice so the
        next utterance reloads; a device change re-targets the player. speed and
        volume are read per utterance and need nothing.
        """
        with self._voice_lock:
            old = self._config
            reload = (old.voice, old.backend, old.model_path) != (new.voice, new.backend, new.model_path)
            device_changed = old.device != new.device
            copy_into(old, new)
        if reload:
            # Any queued utterance without an explicit voice picks up the new one;
            # force a reload on next synthesis.
            self._backend_voice = None
        if device_changed:
            self._player.set_device(new.device)

    def shutdown(self) -> None:
        self._shutdown = True
        self.stop()
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._backend is not None:
            self._backend.unload()

    # ── Worker ───────────────────────────────────────────────────────

    def _drain_queue(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _worker(self) -> None:
        while not self._shutdown:
            item = self._queue.get()
            if item is None:
                break
            with self._state_lock:
                self._busy = True
            self._abort.clear()
            try:
                self._speak(item)
            except Exception:
                logger.exception("Speech failed")
            finally:
                if self._queue.empty():
                    self._end_run()
                with self._state_lock:
                    self._busy = False
                    if self._queue.empty():
                        self._idle.set()
        self._end_run()
        self._idle.set()

    def _begin_run(self) -> None:
        if not self._speaking.is_set():
            self._speaking.set()
            if self.on_speech_start:
                try:
                    self.on_speech_start()
                except Exception:
                    logger.exception("on_speech_start raised")

    def _end_run(self) -> None:
        if self._speaking.is_set():
            self._speaking.clear()
            if self.on_speech_end:
                try:
                    self.on_speech_end()
                except Exception:
                    logger.exception("on_speech_end raised")

    def _ensure_backend(self, voice: str) -> TTSBackend | None:
        """Load backend/voice if not already loaded. Returns None on failure."""
        if self._backend is not None and self._backend_voice == voice and self._backend.loaded:
            self._backend.speed = self._config.speed
            return self._backend
        try:
            path, spec = resolve_model_path(
                voice, self._config.model_path, self._config.auto_download,
                progress=lambda msg: (logger.info("%s", msg), notify("Vocal", msg, icon="audio-speakers")),
            )
            if self._backend is None or self._backend.name != spec.backend:
                if self._backend is not None:
                    self._backend.unload()
                self._backend = resolve_backend(spec.backend)
            self._backend.speed = self._config.speed
            self._backend.load(path, spec.style)
            self._backend_voice = voice
            return self._backend
        except (VoiceNotFoundError, BackendUnavailable, FileNotFoundError, ValueError) as e:
            logger.error("Cannot load voice %r: %s", voice, e)
            notify("Vocal — speech unavailable", str(e), urgency="critical", icon="dialog-error")
            return None

    def _speak(self, item: Utterance) -> None:
        with self._voice_lock:
            voice = item.voice or self._config.voice
        backend = self._ensure_backend(voice)
        if backend is None:
            return
        gain = max(0, min(100, self._config.volume)) / 100.0
        for sentence in split_sentences(item.text):
            if self._abort.is_set():
                return
            synthesis = backend.synthesize(sentence)
            ok = self._player.play(
                synthesis.chunks, synthesis.sample_rate, gain=gain,
                on_first_audio=self._begin_run,
            )
            if not ok:
                return
