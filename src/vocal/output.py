"""Text injection into the active window via clipboard or xdotool."""

from __future__ import annotations

import logging
import subprocess
import time

from vocal.config import OutputConfig

logger = logging.getLogger(__name__)


def _run(cmd: list[str], timeout: float = 5.0, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess with timeout, logging failures."""
    try:
        return subprocess.run(cmd, timeout=timeout, capture_output=True, **kwargs)
    except FileNotFoundError:
        logger.error("Command not found: %s — is it installed?", cmd[0])
        raise
    except subprocess.TimeoutExpired:
        logger.error("Command timed out: %s", " ".join(cmd))
        raise


def inject_clipboard(text: str) -> None:
    """Inject text by copying to clipboard and pasting with Ctrl+V."""
    # 1. Save current clipboard
    old_clipboard = None
    try:
        result = _run(["xclip", "-selection", "clipboard", "-o"], timeout=1.0)
        if result.returncode == 0:
            old_clipboard = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # 2. Set clipboard to our text
    proc = subprocess.Popen(
        ["xclip", "-selection", "clipboard"],
        stdin=subprocess.PIPE,
    )
    proc.communicate(text.encode("utf-8"))
    if proc.returncode != 0:
        logger.warning("xclip set failed (rc=%d)", proc.returncode)
        return

    # 3. Paste
    _run(["xdotool", "key", "--clearmodifiers", "ctrl+v"], timeout=2.0)

    # 4. Restore clipboard after a short delay
    if old_clipboard is not None:
        time.sleep(0.1)
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
        )
        proc.communicate(old_clipboard)
        if proc.returncode != 0:
            logger.warning("xclip restore failed (rc=%d)", proc.returncode)


def inject_xdotool(text: str, delay_ms: int = 8) -> None:
    """Inject text by simulating keystrokes with xdotool."""
    _run(
        ["xdotool", "type", "--clearmodifiers", "--delay", str(delay_ms), "--", text],
        timeout=30.0,
    )


def inject_text(text: str, config: OutputConfig) -> None:
    """Inject text using the configured method."""
    if not text:
        return

    logger.debug("Injecting %d chars via %s", len(text), config.method)

    if config.method == "clipboard":
        inject_clipboard(text)
    elif config.method == "xdotool":
        inject_xdotool(text, delay_ms=config.xdotool_delay)
    else:
        logger.error("Unknown output method: %s", config.method)
