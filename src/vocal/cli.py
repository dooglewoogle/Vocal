"""CLI entry point for Vocal."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Callable

from vocal.config import CONFIG_PATH, VocalConfig, load_config
from vocal.phrasebook import Phrasebook
from vocal.utils import (
    check_dependencies,
    check_tray_dependencies,
    log_startup_banner,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vocal",
        description="Local CPU-only dictation — speak and text appears in the active window.",
    )
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
    return parser.parse_args()


def list_audio_devices() -> None:
    """Print available audio input devices."""
    import sounddevice as sd

    print("Available audio input devices:\n")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{default}")
            print(f"       channels={dev['max_input_channels']}, rate={dev['default_samplerate']}")
    print()


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


def _run_with_tray(
    config: VocalConfig,
    args: argparse.Namespace,
    phrasebook: Phrasebook | None,
) -> None:
    """Main-thread flow: construct tray + engine, wire shutdown, run."""
    from vocal.tray import TrayIcon

    shutdown_started = threading.Event()
    # Hold a reference to the engine so menu callbacks built before the engine
    # exists can resolve it later.
    holder: dict[str, object] = {}

    def request_shutdown() -> None:
        if shutdown_started.is_set():
            return
        shutdown_started.set()
        logger.info("Shutdown requested; stopping tray")
        tray.stop()

    def on_toggle_pause() -> None:
        engine = holder.get("engine")
        toggle = getattr(engine, "toggle_pause", None)
        if callable(toggle):
            toggle()
        else:
            logger.info("Pause requested — not supported in this mode")

    tray = TrayIcon(
        on_toggle_pause=on_toggle_pause,
        on_quit=request_shutdown,
    )

    from vocal.base_engine import BaseDictationEngine

    # Default is live mode unless --hotkey is explicitly passed.
    use_live = not args.hotkey

    engine: BaseDictationEngine
    if use_live:
        from vocal.live import LiveDictationEngine
        engine = LiveDictationEngine(
            config, phrasebook, args.phrasebook, args.phrasebook_replace,
            on_state_change=tray.set_state,
            on_shutdown_requested=request_shutdown,
        )
    else:
        from vocal.engine import DictationEngine
        engine = DictationEngine(
            config, phrasebook, args.phrasebook, args.phrasebook_replace,
            on_state_change=tray.set_state,
            on_shutdown_requested=request_shutdown,
        )
    holder["engine"] = engine

    _install_shutdown_handlers(request_shutdown)

    engine.start()
    try:
        tray.run()  # blocks main; returns when tray.stop() is called
    finally:
        engine.shutdown()
        logger.info("Engine shutdown complete")


def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.benchmark:
        from vocal.benchmark import run_benchmark
        run_benchmark(
            latency_target=args.latency_target,
            compute_type=args.compute_type or "int8",
            use_mic=args.benchmark_mic,
        )
        return

    # Load config
    from pathlib import Path
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # Apply CLI overrides
    if args.model:
        config.model.size = args.model
    if args.compute_type:
        config.model.compute_type = args.compute_type
    if args.beam_size is not None:
        config.model.beam_size = args.beam_size
    if args.key:
        config.hotkey.key = args.key
    if args.mode:
        config.hotkey.mode = args.mode
    if args.output:
        config.output.method = args.output
    if args.hotkey_backend:
        config.hotkey.backend = args.hotkey_backend
    if args.log_level:
        config.log_level = args.log_level

    if args.silence_ms is not None:
        config.live.min_silence_duration_ms = args.silence_ms

    log_path = setup_logging(config.log_level)
    log_startup_banner(log_path)

    # Check system + tray dependencies together — fail fast with one message.
    missing = check_dependencies(config.output.method)
    missing_tray = check_tray_dependencies()
    if missing or missing_tray:
        _fail_missing(missing, missing_tray)

    # Load phrasebook if either flag is set
    phrasebook = None
    if args.phrasebook or args.phrasebook_replace:
        from vocal.phrasebook import load_phrasebook
        phrasebook = load_phrasebook()

    _run_with_tray(config, args, phrasebook)
