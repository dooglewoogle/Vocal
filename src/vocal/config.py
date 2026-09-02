"""Configuration dataclasses and TOML loading.

Layout (TOML tables in brackets):

    [input.model] [input.audio] [input.hotkey] [input.inject]
    [input.vad]   [input.live]  [input.postprocess]
    [output.speech] [output.server]
    log_level = "INFO"

``input`` is the speech-to-text side, ``output`` the text-to-speech side.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore[assignment]


def _default_config_dir() -> Path:
    """Return the platform-appropriate configuration directory."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "vocal"
    elif sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "vocal"
    else:
        return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "vocal"


CONFIG_DIR = _default_config_dir()
CONFIG_PATH = CONFIG_DIR / "config.toml"


class ConfigError(Exception):
    """Raised for a config file that cannot be applied (bad types, legacy layout)."""


# ── Input (speech-to-text) ──────────────────────────────────────────


@dataclass
class ModelConfig:
    size: str = "small.en"
    compute_type: str = "int8"
    beam_size: int = 3
    cpu_threads: int = 0
    language: str = "en"


@dataclass
class AudioConfig:
    device: str | None = None
    sample_rate: int = 16000
    block_size: int = 1024


@dataclass
class HotkeyConfig:
    backend: str = "auto"
    key: str = "PAUSE"
    mode: str = "toggle"
    duck: bool = False
    duck_amount: int = 50


@dataclass
class InjectConfig:
    """How transcribed text is delivered to the active window."""

    method: str = "clipboard"
    xdotool_delay: int = 8


@dataclass
class VADConfig:
    enabled: bool = True
    threshold: float = 0.5
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 200


@dataclass
class LiveConfig:
    min_silence_duration_ms: int = 600
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = 30.0


@dataclass
class PostprocessConfig:
    strip_leading_space: bool = True
    capitalize_first: bool = True
    remove_filler_words: bool = True
    remove_hallucinations: bool = True


@dataclass
class InputConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    inject: InjectConfig = field(default_factory=InjectConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)


# ── Output (text-to-speech) ─────────────────────────────────────────


@dataclass
class SpeechConfig:
    backend: str = "piper"  # piper | kokoro | system
    voice: str = "piper-en-lessac-medium"  # key in vocal.output.models.VOICES
    model_path: str | None = None  # manual model location; bypasses registry/download
    auto_download: bool = True
    speed: float = 1.0
    volume: int = 100  # digital gain 0-100 applied to synthesized PCM
    device: str | None = None  # sounddevice output device name or index
    pause_input: bool = True  # suppress dictation while speaking
    pause_input_tail_ms: int = 300
    duck: bool = True  # duck other audio streams while speaking
    duck_amount: int = 50


@dataclass
class ServerConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 47821  # falls back to an ephemeral port if taken


@dataclass
class OutputConfig:
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# ── Root ────────────────────────────────────────────────────────────


@dataclass
class VocalConfig:
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    log_level: str = "INFO"


# Top-level tables from the 0.2 flat layout and where they live now.
LEGACY_TABLES: dict[str, str] = {
    "model": "input.model",
    "audio": "input.audio",
    "hotkey": "input.hotkey",
    "output": "input.inject",
    "vad": "input.vad",
    "live": "input.live",
    "postprocess": "input.postprocess",
}


def _check_legacy_layout(data: dict, path: Path) -> None:
    """Refuse a 0.2-style flat config with a message naming every renamed table.

    ``_apply_dict`` silently ignores unknown keys, so without this an old
    config would load as all-defaults with no indication anything was wrong.
    A flat ``[output]`` table is detectable because the new ``[output]``
    only contains ``speech`` / ``server`` sub-tables.
    """
    legacy: list[str] = []
    for old, new in LEGACY_TABLES.items():
        value = data.get(old)
        if not isinstance(value, dict):
            continue
        if old == "output" and set(value) <= {"speech", "server"}:
            continue
        legacy.append(f"  [{old}]  ->  [{new}]")
    if legacy:
        raise ConfigError(
            f"{path} uses the pre-0.3 flat layout. Rename these tables:\n"
            + "\n".join(legacy)
        )


def _apply_dict(obj: object, d: dict) -> None:
    """Apply a dict of overrides onto a dataclass instance with type checking."""
    for key, value in d.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _apply_dict(current, value)
        else:
            if current is not None:
                # Reject bool for int fields — TOML distinguishes them even
                # though Python's bool is a subclass of int.
                if isinstance(value, bool) and isinstance(current, int) and not isinstance(current, bool):
                    raise ConfigError(
                        f"Config key {key!r}: expected {type(current).__name__}, "
                        f"got bool ({value!r})"
                    )
                if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
                    value = float(value)
                if not isinstance(value, type(current)):
                    raise ConfigError(
                        f"Config key {key!r}: expected {type(current).__name__}, "
                        f"got {type(value).__name__} ({value!r})"
                    )
            setattr(obj, key, value)


def load_config(path: Path | None = None) -> VocalConfig:
    """Load config from TOML file, falling back to defaults."""
    config = VocalConfig()
    path = path or CONFIG_PATH

    if path.exists():
        if tomllib is None:
            import warnings
            warnings.warn(
                f"Config file {path} found but cannot be loaded: "
                "install 'tomli' on Python < 3.11 (`pip install tomli`)",
                stacklevel=2,
            )
        else:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            _check_legacy_layout(data, path)
            _apply_dict(config, data)

    return config
