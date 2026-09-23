"""Speech controller: FIFO of utterances, pipelined synthesis + playback, interrupt."""

from __future__ import annotations

import copy
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vocal.config import SpeechConfig, copy_into
from vocal.notify import notify
from vocal.output.backends import BACKENDS, BackendUnavailable, TTSBackend, resolve_backend
from vocal.output.models import (
    DEFAULT_VOICE,
    VoiceNotFoundError,
    VoiceSpec,
    get_voice,
    resolve_model_path,
    resolve_voice,
)
from vocal.output.playback import AudioPlayer

logger = logging.getLogger(__name__)

_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")
_LOOKAHEAD = 2  # sentences fully handed to the play thread ahead of the one playing
_SENTENCE_GAP_MS = 300  # breath between consecutive sentences of a run; none before the first
_GAP_POLL_S = 0.02  # how often the gap wait re-checks for stop()


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation / newlines so the first sentence can
    start playing while later ones synthesize. Whitespace-only parts dropped."""
    return [p.strip() for p in _SPLIT.split(text) if p and p.strip()]


@dataclass
class Utterance:
    text: str
    voice: str | None = None


@dataclass(frozen=True)
class SayResult:
    """What :meth:`SpeechController.say` did with the requested voice."""

    voice: str  # canonical voice that will speak (the default when falling back)
    fallback: str | None = None  # why the requested voice was not used


@dataclass
class _Pending:
    """One sentence's audio: filled by the synth thread, drained by the play thread.

    ``epoch`` is the controller's stop counter at synthesis time; a pending whose
    epoch is behind the current one was cancelled and must not be played.
    """

    epoch: int
    sample_rate: int
    gain: float
    chunks: queue.Queue[np.ndarray | None] = field(default_factory=queue.Queue)

    def __iter__(self) -> Iterator[np.ndarray]:
        while (chunk := self.chunks.get()) is not None:
            yield chunk

    def close(self) -> None:
        self.chunks.put(None)


class SpeechController:
    """Owns the TTS backend, two worker threads, and the playback queue.

    ``tts-synth`` pulls utterances, splits them into sentences and renders each
    into a :class:`_Pending`; ``tts-play`` plays pendings in order. Up to
    ``_LOOKAHEAD`` sentences sit between them, so the next sentence (and the next
    queued request) renders while the current one is audible.

    ``on_speech_start`` fires when the first audio of a speaking run is
    about to play; ``on_speech_end`` when the queue drains or is stopped.
    A "run" is any stretch of back-to-back utterances, so callers get one
    start/end pair rather than one per sentence.

    Bookkeeping: ``_epoch`` increments on every :meth:`stop`; both threads
    compare against it instead of sharing a flag one of them would have to
    clear. ``_work`` counts outstanding units of the *current* epoch — one per
    utterance not yet fully synthesized plus one per pending not yet played or
    discarded. :meth:`stop` zeroes it, and a thread only retires a unit if its
    epoch is still current, so a stale render still grinding on the synth
    thread (Kokoro cannot be interrupted mid-sentence) does not delay idle or
    the run's end. ``_playing`` covers the one thing that must finish first:
    the play thread returning from an aborted ``play()``.

    Voices: each utterance may name its own voice. An unknown name, or one
    whose model cannot be loaded, falls back to the default voice (the
    configured one, or ``DEFAULT_VOICE`` if that is itself unknown). One
    backend per engine stays loaded, so alternating voices only reloads when
    the model file changes; Kokoro speakers share one model.
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
        # One backend per engine, created lazily. An injected (test) backend serves every engine.
        self._backends: dict[str, TTSBackend] = dict.fromkeys(BACKENDS, backend) if backend else {}
        self._warned_config_voice: str | None = None
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self._queue: queue.Queue[Utterance | None] = queue.Queue()
        self._sentences: queue.Queue[_Pending | None] = queue.Queue(maxsize=_LOOKAHEAD)
        self._speaking = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._voice_lock = threading.Lock()
        self._state_lock = threading.Lock()  # guards _epoch, _work, _speaking transitions, _idle
        self._epoch = 0
        self._work = 0  # fresh units outstanding; see class docstring
        self._playing = False  # play thread is inside player.play()
        self._synth: threading.Thread | None = None
        self._play: threading.Thread | None = None
        self._shutdown = False

    # ── Public API ───────────────────────────────────────────────────

    @property
    def voice(self) -> str:
        """The default voice: the configured one if it resolves, else ``DEFAULT_VOICE``."""
        found = resolve_voice(self._config.voice)
        return found[0] if found else DEFAULT_VOICE

    @property
    def backend_name(self) -> str:
        engine = get_voice(self.voice).backend
        backend = self._backends.get(engine)
        return backend.name if backend is not None else engine

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def queue_length(self) -> int:
        """Utterances not yet picked up for synthesis (sentences already
        rendering ahead of playback are not counted)."""
        return self._queue.qsize()

    def start(self) -> None:
        if self._synth is not None:
            return
        self._synth = threading.Thread(target=self._synth_loop, name="tts-synth", daemon=True)
        self._play = threading.Thread(target=self._play_loop, name="tts-play", daemon=True)
        self._synth.start()
        self._play.start()

    def say(self, text: str, interrupt: bool = False, voice: str | None = None) -> SayResult:
        """Enqueue ``text``. ``interrupt`` flushes everything first.

        ``voice`` may be any name :func:`resolve_voice` accepts; an unknown one
        falls back to the default voice (reported in the result, never raised).
        """
        found = resolve_voice(voice)
        if found is not None:
            result = SayResult(found[0])
        elif voice and voice.strip():
            result = SayResult(self.voice, f"unknown voice {voice!r}")
            logger.warning("Unknown voice %r; using the default voice %s", voice, result.voice)
        else:
            result = SayResult(self.voice)
        text = text.strip()
        if not text:
            return result
        if interrupt:
            self.stop()
        self.start()
        with self._state_lock:
            self._work += 1
            self._idle.clear()
            self._queue.put(Utterance(text, found[0] if found else None))
        return result

    def stop(self) -> None:
        """Flush the queue and halt playback. Blocks until idle unless called
        from one of the worker threads (e.g. inside a callback)."""
        with self._state_lock:
            # Same critical section as the play thread's freshness check +
            # player.reset(), so a stale sentence can never wipe this abort.
            self._epoch += 1
            self._work = 0  # everything outstanding just became stale
            self._player.abort()
        self._drain(self._queue)
        self._drain(self._sentences)
        self._settle()
        if not self._on_worker_thread():
            self._idle.wait(timeout=5.0)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been spoken (or stopped)."""
        return self._idle.wait(timeout)

    def set_voice(self, name: str) -> None:
        """Make ``name`` the default voice; it loads (downloading if needed) on next use."""
        get_voice(name)  # validate now so callers get an immediate error
        new = copy.copy(self._config)
        new.voice = name
        self.apply_config(new)

    def apply_config(self, new: SpeechConfig) -> None:
        """Adopt ``new`` in place on this controller (other components hold a
        reference to it and to our config object).

        Voice, model_path, speed and volume are read per utterance (a backend
        reloads only if the model it needs changed); a device change re-targets
        the player.
        """
        with self._voice_lock:
            old = self._config
            device_changed = old.device != new.device
            copy_into(old, new)
        if device_changed:
            self._player.set_device(new.device)

    def shutdown(self) -> None:
        self._shutdown = True
        self.stop()
        me = threading.current_thread()
        if self._synth is not None:
            self._queue.put(None)
            if me is not self._synth:
                self._synth.join(timeout=5.0)
            # No producer left: clear anything still queued so the sentinel fits.
            self._drain(self._sentences)
            self._sentences.put_nowait(None)
            if self._play is not None and me is not self._play:
                self._play.join(timeout=5.0)
            self._synth = self._play = None
        for backend in {id(b): b for b in self._backends.values()}.values():
            backend.unload()

    # ── Shared bookkeeping ───────────────────────────────────────────

    def _on_worker_thread(self) -> bool:
        return threading.current_thread() in (self._synth, self._play)

    def _stale(self, epoch: int) -> bool:
        return self._epoch != epoch

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        """Discard queued items (shutdown sentinels are kept)."""
        sentinels = 0
        try:
            while True:
                if q.get_nowait() is None:
                    sentinels += 1
        except queue.Empty:
            pass
        for _ in range(sentinels):
            q.put_nowait(None)

    def _retire(self, epoch: int) -> None:
        """Retire one unit counted under ``epoch``; a no-op if stop() has
        since wiped that epoch. Then see whether we have gone quiet."""
        with self._state_lock:
            if not self._stale(epoch):
                self._work -= 1
        self._settle()

    def _settle(self) -> None:
        """If nothing fresh is outstanding and nothing is playing: end the run
        (callback outside the lock) and flag idle if that is still true."""
        with self._state_lock:
            quiet = self._work == 0 and not self._playing
            ending = quiet and self._speaking.is_set()
            if ending:
                self._speaking.clear()
        if ending and self.on_speech_end:
            try:
                self.on_speech_end()
            except Exception:
                logger.exception("on_speech_end raised")
        with self._state_lock:
            if self._work == 0 and not self._playing:
                self._idle.set()

    def _begin_run(self) -> None:
        with self._state_lock:
            if self._speaking.is_set():
                return
            self._speaking.set()
        if self.on_speech_start:
            try:
                self.on_speech_start()
            except Exception:
                logger.exception("on_speech_start raised")

    # ── Synth thread ─────────────────────────────────────────────────

    def _synth_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            with self._state_lock:
                epoch = self._epoch
            try:
                self._synthesize(item, epoch)
            except Exception:
                logger.exception("Speech synthesis failed")
            finally:
                self._retire(epoch)

    def _default_voice(self) -> str:
        """``voice``, warning once per bad configured value."""
        configured = self._config.voice
        if resolve_voice(configured) is None and configured != self._warned_config_voice:
            self._warned_config_voice = configured
            logger.warning("Configured voice %r is unknown; using %s", configured, DEFAULT_VOICE)
        return self.voice

    def _locate(self, voice: str, model_path: str | None) -> tuple[Path | None, VoiceSpec]:
        """Model path and spec for ``voice``, downloading if allowed (blocks)."""
        return resolve_model_path(
            voice, model_path, self._config.auto_download,
            progress=lambda msg: (logger.info("%s", msg), notify("Vocal", msg, icon="audio-speakers")),
        )

    def _ensure_backend(self, voice: str, default: str) -> TTSBackend | None:
        """The engine for ``voice`` with its model loaded, or None on failure.

        ``model_path`` only overrides the default voice: it points at one model
        and would be wrong for any other voice a request names.
        """
        try:
            path, spec = self._locate(voice, self._config.model_path if voice == default else None)
            backend = self._backends.get(spec.backend)
            if backend is None:
                backend = self._backends[spec.backend] = resolve_backend(spec.backend)
            backend.speed = self._config.speed
            backend.load(path, spec.style)  # no-op / speaker switch when the model is already loaded
            return backend
        except (VoiceNotFoundError, BackendUnavailable, FileNotFoundError, ValueError) as e:
            logger.error("Cannot load voice %r: %s", voice, e)
            if voice == default:  # a requested voice falls back instead
                notify("Vocal — speech unavailable", str(e), urgency="critical", icon="dialog-error")
            return None

    def _synthesize(self, item: Utterance, epoch: int) -> None:
        with self._voice_lock:
            default = self._default_voice()
        voice = item.voice or default
        backend = self._ensure_backend(voice, default)  # may block on a download
        if backend is None and voice != default:
            logger.warning("Voice %s unavailable; falling back to %s", voice, default)
            backend = self._ensure_backend(default, default)
        if backend is None:
            return
        gain = max(0, min(100, self._config.volume)) / 100.0
        for sentence in split_sentences(item.text):
            if self._stale(epoch):
                return
            synthesis = backend.synthesize(sentence)
            pending = _Pending(epoch, synthesis.sample_rate, gain)
            with self._state_lock:
                if self._stale(epoch):
                    return
                self._work += 1  # counted before it is visible to the play thread
            if not self._offer(pending, epoch):
                self._retire(epoch)  # never queued, so nobody else will retire it
                return
            try:
                # Hand off first, then stream chunks in: the play thread starts on
                # the first chunk while the rest of the sentence is still rendering.
                for chunk in synthesis.chunks:
                    if self._stale(epoch):
                        return
                    pending.chunks.put(chunk)
            finally:
                pending.close()  # always, or a play thread blocked on chunks.get() hangs

    def _offer(self, pending: _Pending, epoch: int) -> bool:
        """Put onto the bounded sentence queue; give up if cancelled meanwhile."""
        while True:
            try:
                self._sentences.put(pending, timeout=0.05)
                return True
            except queue.Full:
                if self._stale(epoch) or self._shutdown:
                    return False

    # ── Play thread ──────────────────────────────────────────────────

    def _gap(self, epoch: int) -> bool:
        """Pause between sentences. Returns False if stop() arrived meanwhile."""
        deadline = time.monotonic() + _SENTENCE_GAP_MS / 1000
        while time.monotonic() < deadline:
            if self._stale(epoch):
                return False
            time.sleep(_GAP_POLL_S)
        return True

    def _play_loop(self) -> None:
        while True:
            pending = self._sentences.get()
            if pending is None:
                break
            try:
                # Mid-run (something already played, no stop since): breathe first.
                fresh = not self._speaking.is_set() or self._gap(pending.epoch)
                with self._state_lock:
                    fresh = fresh and not self._stale(pending.epoch)
                    if fresh:
                        self._player.reset()
                        self._playing = True
                if fresh:
                    try:
                        self._player.play(
                            pending, pending.sample_rate, gain=pending.gain,
                            on_first_audio=self._begin_run,
                        )
                    finally:
                        with self._state_lock:
                            self._playing = False
            except Exception:
                logger.exception("Playback failed")
            finally:
                self._retire(pending.epoch)
