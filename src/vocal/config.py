"""Configuration dataclass and TOML loading."""

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


CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "vocal"
CONFIG_PATH = CONFIG_DIR / "config.toml"


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
    backend: str = "evdev"
    key: str = "PAUSE"
    mode: str = "toggle"


@dataclass
class OutputConfig:
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
class VocalConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    log_level: str = "INFO"


def _apply_dict(obj: object, d: dict) -> None:
    """Apply a flat dict of overrides onto a dataclass instance with type checking."""
    for key, value in d.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _apply_dict(current, value)
        else:
            if current is not None and not isinstance(value, type(current)):
                raise TypeError(
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
            _apply_dict(config, data)

    return config
