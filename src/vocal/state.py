"""Shared engine state — canonical across hotkey and live modes."""

from __future__ import annotations

import enum


class DictationState(enum.Enum):
    """Canonical engine state, used by both engines and consumed by the tray.

    LOADING     — engine constructed, Whisper model still loading. Every
                  engine starts here and moves to LISTENING once ready.
    SLEEPING    — not listening. Live mode when paused; not reachable from
                  hotkey mode (it stays in LISTENING until the hotkey fires).
    LISTENING   — idle, ready to capture. Live mode waiting for VAD to
                  detect speech; hotkey mode waiting for the key.
    RECORDING   — actively capturing audio.
    TRANSCRIBING — audio captured; model is turning it into text.
    """

    LOADING = "loading"
    SLEEPING = "sleeping"
    LISTENING = "listening"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
