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
import sys
from pathlib import Path

from vocal.config import CONFIG_PATH, ConfigError, VocalConfig, load_config
from vocal.utils import (
    check_dependencies,
    check_gui_dependencies,
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
        help="Hotkey name (e.g., PAUSE, F18, SCROLLLOCK); hold to record, or hold to mute in live mode",
    )
    parser.add_argument(  # 0.3 flag; hotkey is hold-to-talk only now. Accepted so old launchers keep working.
        "--mode", type=str, choices=["toggle", "ptt"], default=None, help=argparse.SUPPRESS,
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
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without the settings window (tray icon only)",
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
    from vocal.input.audio import list_input_devices
    from vocal.output.playback import list_output_devices

    print("Available audio input devices:\n")
    for i, name, is_default in list_input_devices():
        print(f"  [{i}] {name}{' (default)' if is_default else ''}")
    print("\nAvailable audio output devices:\n")
    for i, name, is_default in list_output_devices():
        print(f"  [{i}] {name}{' (default)' if is_default else ''}")
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
    if getattr(args, "no_server", False):
        config.output.server.enabled = False
    return config


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


def _resolve_initial_mode(args: argparse.Namespace) -> str | None:
    """Engine requested on the command line: "live", "hotkey", or None.

    --hotkey / --live are explicit. Without either, --duck implies hotkey
    mode, since recording-time ducking is meaningless in live mode.
    None means nothing was asked for, so the config file's ``input.engine``
    (or its default) applies.
    """
    if args.hotkey:
        return "hotkey"
    if args.live:
        if args.duck:
            logger.warning("--duck has no effect in live mode")
        return "live"
    if args.duck or getattr(args, "mode", None):
        logger.info("Inferred hotkey mode from --duck/--mode (pass --live to override)")
        return "hotkey"
    return None


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


def _apply_cli_overrides(config: VocalConfig, args: argparse.Namespace) -> set[str]:
    """Copy command-line flags onto ``config``. Returns the dotted paths touched,
    so the GUI can tell the user which values did not come from the file."""
    touched: set[str] = set()

    def put(path: str, value: object) -> None:
        obj = config
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        touched.add(path)

    if args.model:
        put("input.model.size", args.model)
    if args.compute_type:
        put("input.model.compute_type", args.compute_type)
    if args.beam_size is not None:
        put("input.model.beam_size", args.beam_size)
    if args.key:
        put("input.hotkey.key", args.key)
    if args.duck:
        put("input.hotkey.duck", True)
    if args.duck_amount is not None:
        put("input.hotkey.duck_amount", args.duck_amount)
    if args.output:
        put("input.inject.method", args.output)
    if args.hotkey_backend:
        put("input.hotkey.backend", args.hotkey_backend)
    if args.silence_ms is not None:
        put("input.live.min_silence_duration_ms", args.silence_ms)
    if getattr(args, "mode", None):
        logger.warning("--mode is ignored since 0.4: the hotkey is always hold-to-talk")
    if args.phrasebook:
        put("input.phrasebook.seed", True)
    if args.phrasebook_replace:
        put("input.phrasebook.replace", True)
    mode = _resolve_initial_mode(args)
    if mode is not None:
        put("input.engine", mode)
    if args.log_level:
        put("log_level", args.log_level)
    # Speech flags are applied in _load_config_or_exit (shared with `vocal say`).
    if getattr(args, "voice", None):
        touched.add("output.speech.voice")
    if getattr(args, "no_server", False):
        touched.add("output.server.enabled")
    return touched


def _run_daemon(config: VocalConfig, args: argparse.Namespace, overridden: set[str], headless: bool) -> None:
    from vocal.app import VocalApp

    config_path = Path(args.config) if args.config else CONFIG_PATH
    app = VocalApp(config, config_path=config_path, cli_overridden=overridden)
    if headless:
        from vocal.tray import TrayIcon

        tray = TrayIcon(
            on_toggle_pause=app.toggle_pause,
            on_stop_speaking=app.stop_speaking,
            on_quit=app.request_shutdown,
        )
        app.on_state.connect(tray.set_state)
        app.on_speaking.connect(tray.set_speaking)
        app.start(quit_loop=tray.stop)
        app.install_signal_handlers(glib=True)
        try:
            tray.run()  # blocks main; returns when tray.stop() is called
        finally:
            app.shutdown()
        return

    from vocal.gui.window import run_gui

    run_gui(app)


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
    overridden = _apply_cli_overrides(config, args)

    log_path = setup_logging(config.log_level)
    log_startup_banner(log_path)

    headless = bool(args.headless)
    if not headless:
        missing_gui = check_gui_dependencies()
        if missing_gui:
            print(
                "Settings window unavailable (" + "; ".join(missing_gui) + "); running headless.",
                file=sys.stderr,
            )
            headless = True

    # Check system + tray dependencies together — fail fast with one message.
    missing = check_dependencies(config.input.inject.method)
    missing_tray = check_tray_dependencies()
    if missing or (missing_tray and headless):
        _fail_missing(missing, missing_tray)
    if missing_tray:
        print(
            "Tray icon unavailable:\n  " + "\n  ".join(missing_tray)
            + "\nRunning with the window only (closing it quits).",
            file=sys.stderr,
        )

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

    _run_daemon(config, args, overridden, headless)
