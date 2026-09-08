"""VocalApp — the daemon's control object.

Owns the config, the (rebuildable) dictation engine, the speech controller,
the HTTP speech server and the stream ducker, and exposes a small command
surface plus three event signals. The tray icon and the GUI window are thin
views over it and never touch the engine or speech objects directly.

Threading: commands may be called from any thread. Long-running ones
(``apply_config`` rebuilds the engine and reloads the Whisper model) block
the caller, so UI code must invoke them off the UI thread. Events are
emitted on whichever thread produced them (audio worker, TTS worker, timer).
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path

from vocal.config import (
    CONFIG_PATH,
    VocalConfig,
    copy_into,
    save_config,
)
from vocal.input.base_engine import BaseDictationEngine
from vocal.input.phrasebook import Phrasebook, load_phrasebook
from vocal.output.speech import SpeechController
from vocal.state import DictationState

logger = logging.getLogger(__name__)


class Signal:
    """Minimal multi-listener callback. Listener exceptions are logged, not raised."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._listeners: list[Callable[..., None]] = []
        self._lock = threading.Lock()

    def connect(self, fn: Callable[..., None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def disconnect(self, fn: Callable[..., None]) -> None:
        with self._lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def emit(self, *args: object) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(*args)
            except Exception:
                logger.exception("%s listener raised", self._name)


def changed_paths(old: dict, new: dict, prefix: str = "") -> set[str]:
    """Dotted paths whose values differ between two nested dicts."""
    out: set[str] = set()
    for key in set(old) | set(new):
        a, b = old.get(key), new.get(key)
        path = f"{prefix}{key}"
        if isinstance(a, dict) and isinstance(b, dict):
            out |= changed_paths(a, b, path + ".")
        elif a != b:
            out.add(path)
    return out


def _any_under(paths: Iterable[str], *prefixes: str) -> bool:
    return any(p == pre.rstrip(".") or p.startswith(pre) for p in paths for pre in prefixes)


class VocalApp:
    """See module docstring."""

    def __init__(
        self,
        config: VocalConfig,
        *,
        config_path: Path | None = None,
        cli_overridden: Iterable[str] = (),
        # Injection points for tests; production uses the real classes.
        engine_factory: Callable[..., BaseDictationEngine] | None = None,
        speech_factory: Callable[..., SpeechController] | None = None,
        server_factory: Callable[..., object] | None = None,
        ducker_factory: Callable[[int], object] | None = None,
        ducker_available: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.config_path: Path = config_path or CONFIG_PATH
        self.cli_overridden: set[str] = set(cli_overridden)

        self.on_state = Signal("on_state")  # (DictationState)
        self.on_speaking = Signal("on_speaking")  # (bool)
        self.on_transcript = Signal("on_transcript")  # (str)
        self.on_rebuild = Signal("on_rebuild")  # (bool) True = started, False = finished

        self._engine_factory = engine_factory or self._default_engine_factory
        self._server_factory = server_factory
        self._ducker_factory = ducker_factory
        self._ducker_available = ducker_available

        self._engine: BaseDictationEngine | None = None
        self._engine_gen = 0
        self._rebuild_lock = threading.Lock()
        self._rebuilding = threading.Event()
        self._phrasebook: Phrasebook | None = None

        self._server: object | None = None
        self._ducker: object | None = None
        self._release_timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()

        self._shutdown_started = threading.Event()
        self._quit_loop: Callable[[], None] | None = None

        make_speech = speech_factory or SpeechController
        self.speech: SpeechController = make_speech(
            config.output.speech,
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
        )

    # ── Read-only state ──────────────────────────────────────────────

    @property
    def state(self) -> DictationState:
        engine = self._engine
        return engine.current_state if engine is not None else DictationState.LOADING

    @property
    def is_speaking(self) -> bool:
        return self.speech.is_speaking

    @property
    def is_rebuilding(self) -> bool:
        return self._rebuilding.is_set()

    @property
    def phrasebook(self) -> Phrasebook | None:
        return self._phrasebook

    # ── Commands ─────────────────────────────────────────────────────

    def toggle_pause(self) -> None:
        engine = self._engine
        toggle = getattr(engine, "toggle_pause", None)
        if callable(toggle):
            toggle()
        else:
            logger.info("Pause requested — not supported in %s mode", self.config.input.engine)

    def say(self, text: str, *, interrupt: bool = False, voice: str | None = None) -> None:
        self.speech.say(text, interrupt=interrupt, voice=voice)

    def stop_speaking(self) -> None:
        self.speech.stop()

    def set_voice(self, name: str) -> list[str]:
        """Make ``name`` the default voice, persisting it."""
        import copy

        new = copy.deepcopy(self.config)
        new.output.speech.voice = name
        return self.apply_config(new)

    def set_phrasebook(self, phrasebook: Phrasebook | None) -> None:
        """Hot-swap the phrasebook on the running engine (no model reload)."""
        self._phrasebook = phrasebook
        engine = self._engine
        if engine is not None:
            pb_cfg = self.config.input.phrasebook
            engine.set_phrasebook(phrasebook, seed=pb_cfg.seed, replace=pb_cfg.replace)

    def apply_config(self, new: VocalConfig) -> list[str]:
        """Persist ``new`` to the config file and apply it to the running daemon.

        Returns human-readable notes about what happened. Blocks while the
        engine rebuilds (seconds) — call off the UI thread.
        """
        changed = changed_paths(asdict(self.config), asdict(new))
        notes: list[str] = []
        save_config(new, self.config_path)
        notes.append(f"Saved {self.config_path}")
        if not changed:
            return notes
        copy_into(self.config, new)
        logger.info("Config applied; changed: %s", ", ".join(sorted(changed)))

        if "log_level" in changed:
            logging.getLogger().setLevel(self.config.log_level.upper())
            notes.append(f"Log level → {self.config.log_level}")

        input_changed = {p for p in changed if p.startswith("input.")}
        if input_changed and input_changed <= {"input.phrasebook.seed", "input.phrasebook.replace"}:
            self.set_phrasebook(self._load_phrasebook())
            notes.append("Phrasebook settings applied")
        elif input_changed:
            self._rebuild_engine()
            notes.append("Dictation restarted with the new settings")

        if _any_under(changed, "output.speech."):
            self.speech.apply_config(self.config.output.speech)
            self._sync_ducker()
            notes.append("Speech settings applied")

        if _any_under(changed, "output.server."):
            self._stop_server()
            self._start_server()
            notes.append("Speech server restarted" if self._server else "Speech server disabled")
        return notes

    def request_shutdown(self) -> None:
        """Idempotent; safe from any thread. Ends the active main loop."""
        if self._shutdown_started.is_set():
            return
        self._shutdown_started.set()
        logger.info("Shutdown requested")
        quit_loop = self._quit_loop
        if quit_loop is not None:
            try:
                quit_loop()
            except Exception:
                logger.exception("quit_loop raised")

    # ── Lifecycle (used by run_* in cli/gui) ─────────────────────────

    def start(self, quit_loop: Callable[[], None]) -> None:
        """Build and start engine, ducker and server. ``quit_loop`` ends the
        caller's main loop when :meth:`request_shutdown` fires."""
        self._quit_loop = quit_loop
        self._sync_ducker()
        self._engine = self._make_engine()
        # Accept /say only once the engine exists, so the very first utterance
        # can suppress the mic too (engine construction loads the VAD model).
        self._start_server()
        self._engine.start()
        # Engines only emit on *change*; announce the initial (LOADING) state so
        # tray and window don't claim "Listening" while the model is still loading.
        self.on_state.emit(self._engine.current_state)

    def shutdown(self) -> None:
        """Tear everything down in dependency order. Idempotent."""
        self._shutdown_started.set()
        self._stop_server()
        try:
            self.speech.shutdown()
        except Exception:
            logger.exception("Speech shutdown raised")
        self._cancel_release()
        ducker = self._ducker
        if ducker is not None:
            ducker.restore()  # type: ignore[attr-defined]
        engine = self._engine
        if engine is not None:
            self._engine_gen += 1  # drop any late callbacks
            engine.shutdown()
        logger.info("Engine shutdown complete")

    def install_signal_handlers(self, *, glib: bool) -> None:
        """Route SIGINT/SIGTERM to :meth:`request_shutdown`.

        ``glib=True`` (headless mode, pystray owns the main thread): use
        GLib.unix_signal_add so the handler fires inside pystray's loop.
        ``glib=False`` (GUI mode): plain signal.signal — the Tk pump wakes the
        interpreter every 50 ms so Python-level handlers run promptly.
        """
        if glib and sys.platform == "linux":
            try:
                import gi
                gi.require_version("GLib", "2.0")
                from gi.repository import GLib

                def _glib_handler(*_args: object) -> bool:
                    self.request_shutdown()
                    return False

                GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, _glib_handler)
                GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, _glib_handler)
                return
            except (ImportError, ValueError) as e:
                logger.warning("GLib signal install failed (%s); using signal.signal", e)

        def _py_handler(_signum: int, _frame: object) -> None:
            self.request_shutdown()

        signal.signal(signal.SIGINT, _py_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _py_handler)

    # ── Engine ───────────────────────────────────────────────────────

    def _load_phrasebook(self) -> Phrasebook | None:
        pb = self.config.input.phrasebook
        if not (pb.seed or pb.replace):
            return None
        return load_phrasebook()

    @staticmethod
    def _default_engine_factory(mode: str, **kw: object) -> BaseDictationEngine:
        if mode == "live":
            from vocal.input.live import LiveDictationEngine
            kw.pop("on_before_record", None)
            return LiveDictationEngine(**kw)  # type: ignore[arg-type]
        from vocal.input.engine import DictationEngine
        return DictationEngine(**kw)  # type: ignore[arg-type]

    def _make_engine(self) -> BaseDictationEngine:
        """Construct (not start) an engine for the configured mode.

        Callbacks carry a generation token: once a newer engine exists, events
        from this one are dropped, so a dying engine cannot stamp its state
        over the replacement or trigger a daemon shutdown while it exits.
        """
        self._engine_gen += 1
        gen = self._engine_gen

        def guarded(fn: Callable[..., None]) -> Callable[..., None]:
            def wrapper(*args: object) -> None:
                if self._engine_gen == gen:
                    fn(*args)
            return wrapper

        mode = self.config.input.engine
        pb_cfg = self.config.input.phrasebook
        self._phrasebook = self._load_phrasebook()
        logger.info("Starting in %s mode", mode)
        return self._engine_factory(
            mode,
            config=self.config,
            phrasebook=self._phrasebook,
            phrasebook_seed=pb_cfg.seed,
            phrasebook_replace=pb_cfg.replace,
            on_state_change=guarded(self.on_state.emit),
            on_shutdown_requested=guarded(self._on_engine_exited),
            on_transcript=guarded(self.on_transcript.emit),
            # Hotkey mode: pressing the key while speech plays cuts the speech
            # first, synchronously, so the mic never records it.
            on_before_record=self.stop_speaking,
        )

    def _on_engine_exited(self) -> None:
        """The current engine's run() returned (crash or listener exit): quit."""
        self.request_shutdown()

    def _rebuild_engine(self) -> None:
        with self._rebuild_lock:
            self._rebuilding.set()
            self.on_rebuild.emit(True)
            try:
                self._cancel_release()
                old = self._engine
                self._engine_gen += 1  # silence the old engine before stopping it
                if old is not None:
                    old.shutdown()
                new = self._make_engine()
                self._engine = new
                if self.speech.is_speaking and self.config.output.speech.pause_input:
                    new.suppress_input()
                new.start()
                self.on_state.emit(new.current_state)  # LOADING until the model is in
            except Exception:
                logger.exception("Engine rebuild failed")
                raise
            finally:
                self._rebuilding.clear()
                self.on_rebuild.emit(False)

    # ── Speech ↔ input coupling ──────────────────────────────────────

    def _cancel_release(self) -> None:
        with self._timer_lock:
            t = self._release_timer
            self._release_timer = None
        if t is not None:
            t.cancel()

    def _release_input(self) -> None:
        with self._timer_lock:
            self._release_timer = None
        engine = self._engine
        if engine is not None:
            engine.release_input()

    def _on_speech_start(self) -> None:
        self._cancel_release()
        if self.config.output.speech.pause_input:
            engine = self._engine
            if engine is not None:
                engine.suppress_input()
        ducker = self._ducker
        if ducker is not None:
            # pactl round-trips take tens of ms; keep them off the audio path.
            threading.Thread(target=ducker.duck, name="stream-duck", daemon=True).start()  # type: ignore[attr-defined]
        self.on_speaking.emit(True)

    def _on_speech_end(self) -> None:
        ducker = self._ducker
        if ducker is not None:
            threading.Thread(target=ducker.restore, name="stream-restore", daemon=True).start()  # type: ignore[attr-defined]
        self.on_speaking.emit(False)
        cfg = self.config.output.speech
        if cfg.pause_input:
            self._cancel_release()
            t = threading.Timer(cfg.pause_input_tail_ms / 1000.0, self._release_input)
            t.daemon = True
            with self._timer_lock:
                self._release_timer = t
            t.start()

    def _sync_ducker(self) -> None:
        """Create / replace / drop the StreamDucker to match config."""
        cfg = self.config.output.speech
        current = self._ducker
        if current is not None and (not cfg.duck or getattr(current, "_amount", None) != cfg.duck_amount):
            current.restore()  # type: ignore[attr-defined]
            self._ducker = None
            current = None
        if cfg.duck and current is None:
            available = self._ducker_available
            factory = self._ducker_factory
            if factory is None:
                from vocal.volume import StreamDucker
                factory = StreamDucker
                available = available or StreamDucker.available
            if available is None or available():
                self._ducker = factory(cfg.duck_amount)
            else:
                logger.warning("output.speech.duck requested but per-stream ducking needs pactl; ignoring")

    # ── Server ───────────────────────────────────────────────────────

    def _start_server(self) -> None:
        cfg = self.config.output.server
        if not cfg.enabled or self._server is not None:
            return
        factory = self._server_factory
        if factory is None:
            from vocal.output.server import SpeechServer
            factory = SpeechServer
        try:
            server = factory(self.speech, cfg.host, cfg.port)
            server.start()  # type: ignore[attr-defined]
            self._server = server
        except OSError as e:
            logger.error("Speech server failed to start: %s", e)

    def _stop_server(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            try:
                server.stop()  # type: ignore[attr-defined]
            except Exception:
                logger.exception("Speech server stop raised")
