"""CLI entry point for Vocal."""

from __future__ import annotations

import argparse
import sys

from vocal.config import load_config, CONFIG_PATH
from vocal.utils import check_dependencies, setup_logging


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
    parser.add_argument(
        "--live", action="store_true",
        help="Live mode: always-on VAD-driven dictation (no hotkey needed)",
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

    setup_logging(config.log_level)

    # Check system dependencies
    missing = check_dependencies(config.output.method)
    if missing:
        print(f"Missing system dependencies: {', '.join(missing)}", file=sys.stderr)
        if sys.platform == "linux":
            print("Install with: sudo apt install " + " ".join(missing), file=sys.stderr)
        elif sys.platform == "darwin":
            print("These should be built into macOS. Check your PATH.", file=sys.stderr)
        sys.exit(1)

    # Load phrasebook if either flag is set
    phrasebook = None
    if args.phrasebook or args.phrasebook_replace:
        from vocal.phrasebook import load_phrasebook
        phrasebook = load_phrasebook()

    # Run the engine
    if args.live:
        from vocal.live import LiveDictationEngine
        engine = LiveDictationEngine(
            config, phrasebook, args.phrasebook, args.phrasebook_replace,
        )
    else:
        from vocal.engine import DictationEngine
        engine = DictationEngine(
            config, phrasebook, args.phrasebook, args.phrasebook_replace,
        )

    engine.run()
