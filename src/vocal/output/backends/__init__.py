"""Backend registry and resolution with system-TTS fallback."""

from __future__ import annotations

import logging

from vocal.output.backends.base import BackendUnavailable, Synthesis, TTSBackend

logger = logging.getLogger(__name__)

__all__ = ["BackendUnavailable", "Synthesis", "TTSBackend", "BACKENDS", "resolve_backend"]


def _registry() -> dict[str, type[TTSBackend]]:
    from vocal.output.backends.kokoro import KokoroBackend
    from vocal.output.backends.piper import PiperBackend
    from vocal.output.backends.system import SystemBackend

    return {cls.name: cls for cls in (PiperBackend, KokoroBackend, SystemBackend)}


BACKENDS = ("piper", "kokoro", "system")


def resolve_backend(name: str, *, fallback: bool = True) -> TTSBackend:
    """Instantiate backend ``name``.

    If its package is missing and ``fallback`` is true, fall back to the
    system backend with a warning (mirrors the evdev → pynput fallback in
    the hotkey listener). Raises :class:`BackendUnavailable` if nothing
    usable exists.
    """
    registry = _registry()
    try:
        cls = registry[name]
    except KeyError:
        raise ValueError(f"Unknown TTS backend {name!r}; choose from {', '.join(BACKENDS)}") from None

    if cls.is_available():
        return cls()

    hint = {
        "piper": "pip install 'vocal[tts-piper]'",
        "kokoro": "pip install 'vocal[tts-kokoro]'",
        "system": "install espeak-ng (Linux); say/PowerShell are built in on macOS/Windows",
    }[name]
    if not fallback or name == "system":
        raise BackendUnavailable(f"TTS backend {name!r} is not available — {hint}")

    logger.warning("TTS backend %r not available (%s); falling back to system TTS", name, hint)
    system = registry["system"]
    if not system.is_available():
        raise BackendUnavailable(
            f"TTS backend {name!r} not available ({hint}) and no system TTS tool found either"
        )
    return system()
