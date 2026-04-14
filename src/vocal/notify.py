"""Desktop notifications via notify-send.

This module never raises — callers fire-and-forget. If notify-send is
unavailable (macOS, Windows, or missing libnotify-bin on Linux) calls are
silently logged and dropped.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

_AVAILABLE: bool | None = None  # lazy-detected


def _is_available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = shutil.which("notify-send") is not None
        if not _AVAILABLE:
            logger.debug("notify-send not found; desktop notifications disabled")
    return _AVAILABLE


def notify(
    title: str,
    body: str = "",
    urgency: str = "normal",
    icon: str = "audio-input-microphone",
) -> None:
    """Send a transient desktop notification. Never raises.

    Args:
        title: notification headline
        body: detail text
        urgency: low | normal | critical
        icon: freedesktop icon name or path
    """
    if not _is_available():
        return

    cmd = [
        "notify-send",
        "--app-name=Vocal",
        f"--urgency={urgency}",
        f"--icon={icon}",
        title,
    ]
    if body:
        cmd.append(body)

    try:
        subprocess.run(cmd, timeout=5, check=False)
    except Exception:
        logger.debug("notify-send call failed", exc_info=True)
