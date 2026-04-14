"""Text injection into the active window — cross-platform."""

from __future__ import annotations

import logging
import subprocess
import sys
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


# ── Linux (xclip + xdotool) ────────────────────────────────────────


def _inject_clipboard_linux(text: str) -> None:
    """Inject text by copying to clipboard and pasting with Ctrl+V (Linux/X11)."""
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


def _inject_xdotool_linux(text: str, delay_ms: int = 8) -> None:
    """Inject text by simulating keystrokes with xdotool (Linux/X11)."""
    _run(
        ["xdotool", "type", "--clearmodifiers", "--delay", str(delay_ms), "--", text],
        timeout=30.0,
    )


# ── macOS (pbcopy/pbpaste + osascript) ─────────────────────────────


def _inject_clipboard_macos(text: str) -> None:
    """Inject text by copying to clipboard and pasting with Cmd+V (macOS)."""
    # 1. Save current clipboard
    old_clipboard = None
    try:
        result = _run(["pbpaste"], timeout=1.0)
        if result.returncode == 0:
            old_clipboard = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # 2. Set clipboard to our text
    proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    proc.communicate(text.encode("utf-8"))
    if proc.returncode != 0:
        logger.warning("pbcopy failed (rc=%d)", proc.returncode)
        return

    # 3. Paste via Cmd+V
    _run(
        [
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down',
        ],
        timeout=2.0,
    )

    # 4. Restore clipboard after a short delay
    if old_clipboard is not None:
        time.sleep(0.1)
        proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        proc.communicate(old_clipboard)
        if proc.returncode != 0:
            logger.warning("pbcopy restore failed (rc=%d)", proc.returncode)


def _inject_xdotool_macos(text: str, delay_ms: int = 8) -> None:
    """Inject text by simulating keystrokes via osascript (macOS)."""
    escaped = (
        text
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    _run(
        [
            "osascript", "-e",
            f'tell application "System Events" to keystroke "{escaped}"',
        ],
        timeout=30.0,
    )


# ── Windows (stubs) ────────────────────────────────────────────────


def _inject_clipboard_windows(text: str) -> None:
    raise NotImplementedError(
        "Windows text injection is not yet implemented. "
        "Contributions welcome — needs pyperclip + pyautogui or win32 APIs."
    )


def _inject_xdotool_windows(text: str, delay_ms: int = 8) -> None:
    raise NotImplementedError(
        "Windows keystroke injection is not yet implemented."
    )


# ── Dispatcher ──────────────────────────────────────────────────────


_CLIPBOARD_DISPATCH = {
    "linux": _inject_clipboard_linux,
    "darwin": _inject_clipboard_macos,
    "win32": _inject_clipboard_windows,
}

_XDOTOOL_DISPATCH = {
    "linux": _inject_xdotool_linux,
    "darwin": _inject_xdotool_macos,
    "win32": _inject_xdotool_windows,
}


def inject_text(text: str, config: OutputConfig) -> None:
    """Inject text using the configured method, dispatching per platform."""
    if not text:
        return

    logger.debug("Injecting %d chars via %s on %s", len(text), config.method, sys.platform)

    if config.method == "clipboard":
        fn = _CLIPBOARD_DISPATCH.get(sys.platform, _inject_clipboard_linux)
        fn(text)
    elif config.method == "xdotool":
        fn_xdo = _XDOTOOL_DISPATCH.get(sys.platform, _inject_xdotool_linux)
        fn_xdo(text, delay_ms=config.xdotool_delay)
    else:
        logger.error("Unknown output method: %s", config.method)
