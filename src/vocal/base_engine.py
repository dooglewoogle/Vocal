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
from vocal.output import inject_text
from vocal.postprocess import postprocess
from vocal.transcriber import Transcriber

logger = logging.getLogger(__name__)


class BaseDictationEngine(ABC):
    """Base class providing shared transcription pipeline, shutdown, and signal handling."""

    def __init__(self, config: VocalConfig) -> None:
        self._config = config
        self._shutdown = threading.Event()

        self._transcriber = Transcriber(
            config.model, config.vad, sample_rate=config.audio.sample_rate,
        )
        self._transcription_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._output_queue: queue.Queue[str | None] = queue.Queue()

        self._threads: list[threading.Thread] = []

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

            text = self._transcriber.transcribe(audio)
            text = postprocess(text, self._config.postprocess)

            if text:
                self._output_queue.put(text)
            else:
                logger.debug("Empty transcription after postprocessing")

            self._on_transcription_complete()

    def _on_transcription_complete(self) -> None:
        """Hook for subclasses to react after each transcription."""

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
        """Register SIGINT/SIGTERM to trigger shutdown."""
        def handle_signal(signum, frame):
            print("\nShutting down...", flush=True)
            self.shutdown()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

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
