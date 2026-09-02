"""CLI entry point for Vocal.

    vocal [flags]                 run the daemon (dictation + speech server, tray)
    vocal say [--interrupt] TEXT  speak text via the daemon, or in-process if none
    vocal stop                    stop speaking
    vocal status                  daemon speech status
    vocal models list|download|remove
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from vocal.config import CONFIG_DIR, CONFIG_PATH, ConfigError, VocalConfig, load_config
from vocal.state import DictationState
from vocal.input.phrasebook import Phrasebook
from vocal.utils import (
    check_dependencies,
    check_tray_dependencies,
    log_startup_banner,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ── Argument parsing ────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vocal",
        description="Local CPU-only dictation and speech — speak and text appears in the "
                    "active window; send text and it is read aloud.",
    )
    # ── Input (dictation) flags — all valid without a subcommand ──
    parser.add_argument(
        "--model", type=str, default=None,
        help="Whisper model size (tiny.en, base.en, small.en, medium.en)",
    )
    parser.add_argument(
        "--compute-type", type=str, default=None,
        help="Compute type (int8, float32)",
    )
    parser.add_argument(
        "--beam-size", type=int, default=None,
        help="Beam size for decoding (1=greedy, 3=default, 5=thorough)",
    )
    parser.add_argument(
        "--key", type=str, default=None,
        help="Hotkey name (e.g., PAUSE, F18, SCROLLLOCK)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["toggle", "ptt"], default=None,
        help="Hotkey mode: toggle or push-to-talk",
    )
    parser.add_argument(
        "--duck", action="store_true", default=None,
        help="Lower system output volume while recording in hotkey mode",
    )
    parser.add_argument(
        "--duck-amount", type=int, default=None,
        help="Percent to reduce volume by when ducking, 0-100 (default: 50)",
    )
    parser.add_argument(
        "--output", type=str, choices=["clipboard", "xdotool"], default=None,
        help="Text injection method",
    )
    parser.add_argument(
        "--hotkey-backend", type=str, choices=["auto", "evdev", "pynput"], default=None,
        help="Hotkey listener backend",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help=f"Path to config TOML file (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio devices and exit",
    )
    engine_group = parser.add_mutually_exclusive_group()
    engine_group.add_argument(
        "--live", action="store_true", default=False,
        help="Live VAD-driven dictation (the default — flag accepted for clarity)",
    )
    engine_group.add_argument(
        "--hotkey", action="store_true", default=False,
        help="Hotkey-driven dictation (press to record, release to transcribe)",
    )
    parser.add_argument(
        "--silence-ms", type=int, default=None,
        help="Min silence duration in ms before ending an utterance (live mode, default: 600)",
    )
    parser.add_argument(
        "--phrasebook", action="store_true",
        help="Seed Whisper with phrasebook terms to bias decoding toward known vocabulary "
             "(reads from ~/.config/vocal/phrasebook.toml)",
    )
    parser.add_argument(
        "--phrasebook-replace", action="store_true",
        help="Apply phrasebook replacement rules to fix common mishearings after transcription "
             "(reads from ~/.config/vocal/phrasebook.toml)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Benchmark all Whisper model sizes on this hardware and exit",
    )
    parser.add_argument(
        "--latency-target", type=float, default=2.0,
        help="Max acceptable latency in seconds for benchmark recommendation (default: 2.0)",
    )
    parser.add_argument(
        "--benchmark-mic", action="store_true",
        help="Use live mic input for benchmark instead of synthetic audio",
    )
    # ── Output (speech) flags ──
    parser.add_argument(
        "--no-server", action="store_true",
        help="Do not start the localhost speech server",
    )
    parser.add_argument(
        "--voice", type=str, default=None,
        help="TTS voice name (see `vocal models list`)",
    )
    parser.add_argument(
        "--tts-backend", type=str, choices=["piper", "kokoro", "system"], default=None,
        help="TTS backend used with output.speech.model_path",
    )

    # ── Subcommands ──
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    parser.set_defaults(command=None)

    say = sub.add_parser("say", help="Speak text (via the running daemon, or in-process)")
    say.add_argument("text", nargs="*", help="Text to speak; omit or use '-' to read stdin")
    say.add_argument("--interrupt", "-i", action="store_true", help="Cut off current speech first")
    say.add_argument("--voice", type=str, default=None, help="Voice name for this utterance")

    sub.add_parser("stop", help="Stop speaking and clear the queue")
    sub.add_parser("status", help="Show daemon speech status")

    models = sub.add_parser("models", help="Manage TTS voice models")
    msub = models.add_subparsers(dest="models_command", metavar="ACTION")
    models.set_defaults(models_command="list")
    msub.add_parser("list", help="List known voices and whether they are downloaded")
    dl = msub.add_parser("download", help="Download a voice")
    dl.add_argument("name")
    rm = msub.add_parser("remove", help="Delete a downloaded voice")
    rm.add_argument("name")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


# ── Helpers ─────────────────────────────────────────────────────────


def list_audio_devices() -> None:
    """Print available audio input and output devices."""
    import sounddevice as sd

    print("Available audio input devices:\n")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{default}")
            print(f"       channels={dev['max_input_channels']}, rate={dev['default_samplerate']}")
    print("\nAvailable audio output devices:\n")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            default = " (default)" if i == sd.default.device[1] else ""
            print(f"  [{i}] {dev['name']}{default}")
            print(f"       channels={dev['max_output_channels']}, rate={dev['default_samplerate']}")
    print()


def _load_config_or_exit(args: argparse.Namespace) -> VocalConfig:
    config_path = Path(args.config) if args.config else None
    try:
        config = load_config(config_path)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)
    if getattr(args, "voice", None):
        config.output.speech.voice = args.voice
    if getattr(args, "tts_backend", None):
        config.output.speech.backend = args.tts_backend
    if getattr(args, "no_server", False):
        config.output.server.enabled = False
    return config


def _install_shutdown_handlers(on_shutdown: Callable[[], None]) -> None:
    """Wire SIGINT/SIGTERM to the shutdown callback.

    On Linux the tray runs a GTK main loop; plain signal.signal handlers
    won't be delivered promptly (GLib doesn't yield to Python's handler
    between iterations). GLib.unix_signal_add routes signals through the
    same loop that pystray is using, so they fire cleanly.

    On other platforms, fall back to signal.signal — pystray's Cocoa /
    Win32 backends handle this adequately for now.
    """
    if sys.platform == "linux":
        try:
            import gi
            gi.require_version("GLib", "2.0")
            from gi.repository import GLib

            def _glib_handler(*_args: object) -> bool:
                on_shutdown()
                return False  # GLib removes the source after False

            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, _glib_handler)
            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, _glib_handler)
            logger.debug("Installed GLib signal handlers for SIGINT/SIGTERM")
            return
        except (ImportError, ValueError) as e:
            logger.warning("GLib signal install failed (%s); using signal.signal", e)

    def _py_handler(_signum: int, _frame: object) -> None:
        on_shutdown()

    signal.signal(signal.SIGINT, _py_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _py_handler)


def _fail_missing(missing: list[str], missing_tray: list[str]) -> None:
    """Print missing-dep guidance and exit with non-zero status."""
    if missing:
        print(f"Missing system dependencies: {', '.join(missing)}", file=sys.stderr)
        if sys.platform == "linux":
            print("Install with: sudo apt install " + " ".join(missing), file=sys.stderr)
        elif sys.platform == "darwin":
            print("These should be built into macOS. Check your PATH.", file=sys.stderr)
    if missing_tray:
        print(
            "\nMissing tray dependencies:\n  " + "\n  ".join(missing_tray),
            file=sys.stderr,
        )
        if sys.platform == "linux":
            print(
                "\nOn GNOME, also install the 'AppIndicator and KStatusNotifierItem "
                "Support' extension — vanilla GNOME has no built-in tray.",
                file=sys.stderr,
            )
    sys.exit(1)


def _resolve_initial_mode(args: argparse.Namespace) -> str:
    """Pick the starting engine: "live" or "hotkey".

    --hotkey / --live are explicit. Without either, --mode or --duck imply
    hotkey mode, since both are meaningless as dictation controls in live mode.
    """
    if args.hotkey:
        return "hotkey"
    if args.live:
        if args.duck:
            logger.warning("--duck has no effect in live mode")
        return "live"
    if args.mode or args.duck:
        logger.info("Inferred hotkey mode from --mode/--duck (pass --live to override)")
        return "hotkey"
    return "live"


# ── Speech subcommands ──────────────────────────────────────────────


def _say_text(args: argparse.Namespace) -> str:
    if not args.text or args.text == ["-"]:
        return sys.stdin.read()
    return " ".join(args.text)


def _cmd_say(args: argparse.Namespace) -> int:
    from vocal.output import client

    text = _say_text(args).strip()
    if not text:
        print("Nothing to say.", file=sys.stderr)
        return 1
    try:
        if client.say(text, interrupt=args.interrupt, voice=args.voice):
            return 0
    except client.DaemonError as e:
        print(f"Daemon rejected request: {e}", file=sys.stderr)
        return 1

    # No daemon — synthesize in this process.
    from vocal.output.speech import SpeechController

    config = _load_config_or_exit(args)
    setup_logging(args.log_level or "WARNING")
    logger.info("No vocal daemon running; speaking in-process")
    controller = SpeechController(config.output.speech)
    try:
        controller.say(text)
        controller.wait()
    except KeyboardInterrupt:
        controller.stop()
    finally:
        controller.shutdown()
    return 0


def _cmd_stop(_args: argparse.Namespace) -> int:
    from vocal.output import client

    if client.stop():
        return 0
    print("No vocal daemon running.", file=sys.stderr)
    return 1


def _cmd_status(_args: argparse.Namespace) -> int:
    from vocal.output import client

    info = client.status()
    if info is None:
        print("No vocal daemon running.", file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2))
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from vocal.output.models import (
        VOICES,
        VoiceNotFoundError,
        download_voice,
        is_downloaded,
        models_dir,
        remove_voice,
    )

    action = args.models_command
    try:
        if action == "list":
            print(f"Models directory: {models_dir()}\n")
            width = max(len(n) for n in VOICES)
            for name, spec in VOICES.items():
                mark = "✓" if is_downloaded(spec) else " "
                print(f"  [{mark}] {name:<{width}}  {spec.backend:<7} {spec.description}")
            print("\n[✓] = downloaded. Fetch with: vocal models download NAME")
        elif action == "download":
            download_voice(args.name, progress=print)
        elif action == "remove":
            print("Removed." if remove_voice(args.name) else "Nothing to remove.")
    except VoiceNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


# ── Daemon ──────────────────────────────────────────────────────────


def _run_with_tray(
    config: VocalConfig,
    args: argparse.Namespace,
    phrasebook: Phrasebook | None,
) -> None:
    """Main-thread flow: construct tray + engine + speech, wire shutdown, run."""
    from vocal.input.audio import resolve_device
    from vocal.input.base_engine import BaseDictationEngine
    from vocal.output.speech import SpeechController
    from vocal.tray import TrayIcon

    shutdown_started = threading.Event()
    switching_mode = threading.Event()
    # Hold a reference to the engine so menu callbacks built before the engine
    # exists can resolve it later.
    holder: dict[str, object] = {}

    def request_shutdown() -> None:
        if switching_mode.is_set():
            return  # suppress during mode switch
        if shutdown_started.is_set():
            return
        shutdown_started.set()
        logger.info("Shutdown requested; stopping tray")
        tray.stop()

    # ── Engine factory ──────────────────────────────────────────────

    def _make_engine(mode: str) -> BaseDictationEngine:
        if mode == "live":
            from vocal.input.live import LiveDictationEngine
            return LiveDictationEngine(
                config, phrasebook, args.phrasebook, args.phrasebook_replace,
                on_state_change=tray.set_state,
                on_shutdown_requested=request_shutdown,
            )
        else:
            from vocal.input.engine import DictationEngine
            return DictationEngine(
                config, phrasebook, args.phrasebook, args.phrasebook_replace,
                on_state_change=tray.set_state,
                on_shutdown_requested=request_shutdown,
                # Pressing the hotkey while speech plays cuts the speech first,
                # synchronously, so the mic never records it and hotkey-mode
                # ducking never fights active playback.
                on_before_record=lambda: speech.stop(),
            )

    # ── Tray callbacks ──────────────────────────────────────────────

    def on_toggle_pause() -> None:
        engine = holder.get("engine")
        toggle = getattr(engine, "toggle_pause", None)
        if callable(toggle):
            toggle()
        else:
            logger.info("Pause requested — not supported in this mode")

    def on_select_device(device_index: int | None) -> None:
        engine = holder.get("engine")
        switch = getattr(engine, "switch_device", None)
        if callable(switch):
            switch(device_index)

    def on_select_model(model_name: str) -> None:
        engine = holder.get("engine")
        switch = getattr(engine, "switch_model", None)
        if callable(switch):
            switch(model_name)

    def on_switch_mode(mode: str) -> None:
        switching_mode.set()
        try:
            old_engine = holder.get("engine")
            if old_engine:
                old_engine.shutdown()  # type: ignore[union-attr]

            new_engine = _make_engine(mode)
            holder["engine"] = new_engine
            new_engine.start()
            tray.set_state(DictationState.LISTENING)
            logger.info("Switched to %s mode", mode)
        except Exception:
            logger.exception("Mode switch to %s failed", mode)
        finally:
            switching_mode.clear()

    def on_open_phrasebook() -> None:
        from vocal.input.phrasebook import PHRASEBOOK_PATH
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if not PHRASEBOOK_PATH.exists():
            PHRASEBOOK_PATH.write_text(
                "# Vocal Phrasebook — custom vocabulary and corrections\n"
                "#\n"
                "# [replacements]\n"
                '# "mishearing" = "correct term"\n',
            )
        if sys.platform == "linux":
            subprocess.Popen(["xdg-open", str(PHRASEBOOK_PATH)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(PHRASEBOOK_PATH)])
        else:
            import os
            os.startfile(str(PHRASEBOOK_PATH))  # type: ignore[attr-defined]

    # ── Build tray + engine + speech ────────────────────────────────

    initial_mode = _resolve_initial_mode(args)
    if initial_mode == "hotkey":
        logger.info("Starting in hotkey mode (%s, key=%s)", config.input.hotkey.mode, config.input.hotkey.key)
    else:
        logger.info("Starting in live mode")

    tray = TrayIcon(
        on_toggle_pause=on_toggle_pause,
        on_quit=request_shutdown,
        on_select_device=on_select_device,
        on_select_model=on_select_model,
        on_switch_mode=on_switch_mode,
        on_open_phrasebook=on_open_phrasebook,
        on_select_voice=lambda v: speech.set_voice(v),
        on_stop_speaking=lambda: speech.stop(),
        current_model=config.input.model.size,
        current_mode=initial_mode,
        current_device=resolve_device(config.input.audio.device),
        current_voice=config.output.speech.voice,
    )

    # ── Speech output ↔ input coupling ──────────────────────────────
    # While speaking: suppress the mic (so live mode doesn't transcribe the
    # speaker) and duck other apps' streams (not ours — StreamDucker is
    # per-stream; the master-volume Ducker would silence the speech too).

    speech_cfg = config.output.speech
    stream_ducker = None
    if speech_cfg.duck:
        from vocal.volume import StreamDucker
        if StreamDucker.available():
            stream_ducker = StreamDucker(speech_cfg.duck_amount)
        else:
            logger.warning("output.speech.duck requested but per-stream ducking needs pactl; ignoring")

    release_timer: list[threading.Timer | None] = [None]

    def _cancel_release() -> None:
        t = release_timer[0]
        if t is not None:
            t.cancel()
            release_timer[0] = None

    def _release_input() -> None:
        release_timer[0] = None
        engine = holder.get("engine")
        release = getattr(engine, "release_input", None)
        if callable(release):
            release()

    def on_speech_start() -> None:
        _cancel_release()
        if speech_cfg.pause_input:
            engine = holder.get("engine")
            suppress = getattr(engine, "suppress_input", None)
            if callable(suppress):
                suppress()
        if stream_ducker is not None:
            # pactl round-trips take tens of ms; keep them off the audio path.
            threading.Thread(target=stream_ducker.duck, name="stream-duck", daemon=True).start()
        tray.set_speaking(True)

    def on_speech_end() -> None:
        if stream_ducker is not None:
            threading.Thread(target=stream_ducker.restore, name="stream-restore", daemon=True).start()
        tray.set_speaking(False)
        if speech_cfg.pause_input:
            _cancel_release()
            t = threading.Timer(speech_cfg.pause_input_tail_ms / 1000.0, _release_input)
            t.daemon = True
            release_timer[0] = t
            t.start()

    speech = SpeechController(
        speech_cfg,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
    )
    server = None
    if config.output.server.enabled:
        from vocal.output.server import SpeechServer
        try:
            server = SpeechServer(speech, config.output.server.host, config.output.server.port)
            server.start()
        except OSError as e:
            logger.error("Speech server failed to start: %s", e)
            server = None

    engine = _make_engine(initial_mode)
    holder["engine"] = engine

    _install_shutdown_handlers(request_shutdown)

    engine.start()
    try:
        tray.run()  # blocks main; returns when tray.stop() is called
    finally:
        if server is not None:
            server.stop()
        speech.shutdown()
        _cancel_release()
        if stream_ducker is not None:
            stream_ducker.restore()
        engine.shutdown()
        logger.info("Engine shutdown complete")


def main() -> None:
    args = parse_args()

    if args.command == "say":
        sys.exit(_cmd_say(args))
    if args.command == "stop":
        sys.exit(_cmd_stop(args))
    if args.command == "status":
        sys.exit(_cmd_status(args))
    if args.command == "models":
        sys.exit(_cmd_models(args))

    if args.list_devices:
        list_audio_devices()
        return

    if args.benchmark:
        from vocal.input.benchmark import run_benchmark
        run_benchmark(
            latency_target=args.latency_target,
            compute_type=args.compute_type or "int8",
            use_mic=args.benchmark_mic,
        )
        return

    config = _load_config_or_exit(args)

    # Apply CLI overrides
    if args.model:
        config.input.model.size = args.model
    if args.compute_type:
        config.input.model.compute_type = args.compute_type
    if args.beam_size is not None:
        config.input.model.beam_size = args.beam_size
    if args.key:
        config.input.hotkey.key = args.key
    if args.mode:
        config.input.hotkey.mode = args.mode
    if args.duck:
        config.input.hotkey.duck = True
    if args.duck_amount is not None:
        config.input.hotkey.duck_amount = args.duck_amount
    if args.output:
        config.input.inject.method = args.output
    if args.hotkey_backend:
        config.input.hotkey.backend = args.hotkey_backend
    if args.log_level:
        config.log_level = args.log_level

    if args.silence_ms is not None:
        config.input.live.min_silence_duration_ms = args.silence_ms

    log_path = setup_logging(config.log_level)
    log_startup_banner(log_path)

    # Check system + tray dependencies together — fail fast with one message.
    missing = check_dependencies(config.input.inject.method)
    missing_tray = check_tray_dependencies()
    if missing or missing_tray:
        _fail_missing(missing, missing_tray)

    if config.input.hotkey.duck:
        from vocal.volume import candidate_tools, detect_backend
        backend = detect_backend()
        if backend is None:
            logger.warning(
                "Ducking requested but no volume tool found (tried: %s); disabling",
                ", ".join(candidate_tools()) or "none for this platform",
            )
            config.input.hotkey.duck = False
        else:
            logger.info("Ducking enabled: -%d%% via %s", config.input.hotkey.duck_amount, backend.tool)

    # Load phrasebook if either flag is set
    phrasebook = None
    if args.phrasebook or args.phrasebook_replace:
        from vocal.input.phrasebook import load_phrasebook
        phrasebook = load_phrasebook()

    _run_with_tray(config, args, phrasebook)
