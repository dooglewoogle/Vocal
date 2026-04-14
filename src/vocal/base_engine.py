"""BaseDictationEngine — shared infrastructure for dictation engines."""

from __future__ import annotations

import logging
import queue
import signal
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from vocal.config import VocalConfig
from vocal.notify import notify
from vocal.output import inject_text
from vocal.phrasebook import Phrasebook
from vocal.postprocess import postprocess
from vocal.state import DictationState
from vocal.transcriber import Transcriber

logger = logging.getLogger(__name__)


class BaseDictationEngine(ABC):
    """Base class providing shared transcription pipeline, shutdown, and signal handling."""

    #: State to return to after an utterance finishes transcribing.
    #: Live mode overrides to LISTENING (or SLEEPING if paused); hotkey
    #: mode stays LISTENING.
    _idle_state: DictationState = DictationState.LISTENING

    def __init__(
        self,
        config: VocalConfig,
        phrasebook: Phrasebook | None = None,
        phrasebook_seed: bool = False,
        phrasebook_replace: bool = False,
        on_state_change: Callable[[DictationState], None] | None = None,
        on_shutdown_requested: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._shutdown = threading.Event()
        self._phrasebook = phrasebook if phrasebook_replace else None
        self._state_callback = on_state_change
        self._shutdown_callback = on_shutdown_requested
        self._engine_thread: threading.Thread | None = None

        self._seed_phrasebook = phrasebook if phrasebook_seed else None
        self._transcriber = Transcriber(
            config.model, config.vad,
            sample_rate=config.audio.sample_rate,
            phrasebook=self._seed_phrasebook,
        )
        self._model_loading = threading.Lock()
        self._transcription_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._output_queue: queue.Queue[str | None] = queue.Queue()

        self._threads: list[threading.Thread] = []

        # Canonical state. Subclasses own the lock; base just reads/writes under it.
        self._state: DictationState = DictationState.LISTENING
        self._state_lock = threading.Lock()

    # ── State management ────────────────────────────────────────────

    def _set_state(self, state: DictationState) -> None:
        """Atomically update state and fire the on-change hook."""
        with self._state_lock:
            if self._state == state:
                return
            prev = self._state
            self._state = state
        logger.debug("State %s -> %s", prev.value, state.value)
        try:
            self._on_state_change(state)
        except Exception:
            logger.exception("State-change hook raised")

    def _on_state_change(self, state: DictationState) -> None:
        """Called outside the state lock on every transition. Notifies observer."""
        if self._state_callback is not None:
            try:
                self._state_callback(state)
            except Exception:
                logger.exception("State-change observer raised")

    @property
    def current_state(self) -> DictationState:
        """Read the current state. Safe from any thread."""
        with self._state_lock:
            return self._state

    # ── Runtime switching ──────────────────────────────────────────

    def switch_model(self, model_name: str) -> None:
        """Switch Whisper model in the background. Thread-safe, non-blocking."""
        if model_name == self._config.model.size:
            return
        if not self._model_loading.acquire(blocking=False):
            logger.info("Model switch already in progress, ignoring")
            return

        def _load() -> None:
            try:
                from vocal.config import ModelConfig
                new_cfg = ModelConfig(
                    size=model_name,
                    compute_type=self._config.model.compute_type,
                    beam_size=self._config.model.beam_size,
                    cpu_threads=self._config.model.cpu_threads,
                    language=self._config.model.language,
                )
                new_transcriber = Transcriber(
                    new_cfg, self._config.vad,
                    sample_rate=self._config.audio.sample_rate,
                    phrasebook=self._seed_phrasebook,
                )
                print(f"\u2935  Loading model {model_name}...", flush=True)
                new_transcriber.load()
                # Atomic swap — old transcriber stays alive until in-flight work finishes
                self._transcriber = new_transcriber
                self._config.model.size = model_name
                print(f"\u2705 Switched to model {model_name}", flush=True)
                notify("Vocal", f"Model switched to {model_name}")
            except Exception:
                logger.exception("Failed to load model %s", model_name)
                notify("Vocal", f"Failed to load model {model_name}", urgency="critical")
            finally:
                self._model_loading.release()

        threading.Thread(target=_load, name="model-loader", daemon=True).start()

    # ── Shared workers ──────────────────────────────────────────────

    def _transcription_worker(self) -> None:
        """Thread: pull audio from queue, transcribe, push text to output queue."""
        while not self._shutdown.is_set():
            try:
                audio = self._transcription_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if audio is None:
                break

            try:
                text = self._transcriber.transcribe(audio)
                text = postprocess(text, self._config.postprocess, self._phrasebook)

                if text:
                    self._output_queue.put(text)
                else:
                    logger.debug("Empty transcription after postprocessing")
            except Exception:
                # Per-utterance failures should not kill the engine — log and
                # move on so the user can dictate again.
                logger.exception("Transcription failed for utterance")
            finally:
                self._on_transcription_complete()

    def _on_transcription_complete(self) -> None:
        """Hook called after each transcription. Default: return to idle state."""
        self._set_state(self._idle_state)

    def _output_worker(self) -> None:
        """Thread: pull text from queue and inject into active window."""
        while not self._shutdown.is_set():
            try:
                text = self._output_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:
                break

            try:
                inject_text(text, self._config.output)
                print(f"\u2705 {text}", flush=True)
            except Exception:
                logger.exception("Failed to inject text")

    # ── Thread management ───────────────────────────────────────────

    def _start_workers(self, *extras: tuple[Callable[[], None], str]) -> None:
        """Start transcription + output workers, plus any extra (target, name) pairs."""
        targets: list[tuple[Callable[[], None], str]] = [
            (self._transcription_worker, "transcription"),
            (self._output_worker, "output"),
        ]
        targets.extend(extras)

        self._threads = [
            threading.Thread(target=fn, name=name, daemon=True)
            for fn, name in targets
        ]
        for t in self._threads:
            t.start()

    def _install_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM to trigger shutdown.

        Only safe to call on the main thread; no-op otherwise. When the engine
        runs on a background thread (tray mode), the main thread owns signal
        handling — see cli.install_shutdown_handlers().
        """
        if threading.current_thread() is not threading.main_thread():
            logger.debug("Skipping signal install on non-main thread")
            return

        def handle_signal(signum, frame):
            print("\nShutting down...", flush=True)
            self.shutdown()

        signal.signal(signal.SIGINT, handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handle_signal)

    def _request_shutdown(self) -> None:
        """Fire the external shutdown-requested callback once. Idempotent."""
        if self._shutdown.is_set():
            return
        # Don't flip _shutdown here — shutdown() owns that. Just notify.
        cb = self._shutdown_callback
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.exception("on_shutdown_requested callback raised")

    # ── Shutdown ────────────────────────────────────────────────────

    def _sentinel_queues(self) -> list[queue.Queue]:
        """Queues that need a None sentinel to unblock workers. Override to extend."""
        return [self._transcription_queue, self._output_queue]

    def _cleanup_resources(self) -> None:
        """Override to stop streams, listeners, or other I/O resources."""

    def shutdown(self) -> None:
        """Clean shutdown of all threads and resources."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()

        self._cleanup_resources()

        for q in self._sentinel_queues():
            q.put(None)

        for t in self._threads:
            if t.is_alive():
                t.join(timeout=5.0)

        logger.info("Shutdown complete")

    # ── Entry point ─────────────────────────────────────────────────

    @abstractmethod
    def run(self) -> None:
        """Start the engine and block until shutdown."""

    def start(self) -> threading.Thread:
        """Start run() on a background thread. Non-blocking; returns the thread.

        Used by tray mode: the tray owns the main thread, so the engine
        runs its blocking loop off-main. If run() exits for any reason —
        graceful shutdown, listener crash, worker exception — the
        on_shutdown_requested callback fires so the main thread can tear
        down the tray loop and exit cleanly.
        """
        if self._engine_thread is not None:
            raise RuntimeError("Engine already started")

        def _body() -> None:
            try:
                self.run()
            except Exception as exc:
                logger.exception("Engine thread crashed")
                notify("Vocal error", str(exc), urgency="critical")
            finally:
                self._request_shutdown()

        self._engine_thread = threading.Thread(
            target=_body, name="engine", daemon=True,
        )
        self._engine_thread.start()
        return self._engine_thread
