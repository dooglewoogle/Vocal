"""Utility functions — logging setup, dependency checks."""

from __future__ import annotations

import logging
import logging.handlers
import os
import platform as _platform
import shutil
import sys
import tempfile
from pathlib import Path


LOG_FORMAT_FILE = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
LOG_FORMAT_STREAM = "%(asctime)s %(levelname)s: %(message)s"
LOG_DATEFMT_FILE = "%Y-%m-%d %H:%M:%S"
LOG_DATEFMT_STREAM = "%H:%M:%S"
LOG_ROTATE_BYTES = 1 * 1024 * 1024  # 1 MB per file
LOG_ROTATE_COUNT = 5                # keep 5 rotated files + current = ~6 MB cap


def _default_log_dir() -> Path:
    """Platform-appropriate directory for log files."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "vocal"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "vocal" / "Logs"
        return Path.home() / "AppData" / "Local" / "vocal" / "Logs"
    # Linux / other POSIX: XDG_STATE_HOME
    base = os.environ.get("XDG_STATE_HOME")
    if base:
        return Path(base) / "vocal"
    return Path.home() / ".local" / "state" / "vocal"


def _resolve_log_path() -> Path:
    """Return a writable path for the log file, falling back to tempdir."""
    primary = _default_log_dir() / "vocal.log"
    try:
        primary.parent.mkdir(parents=True, exist_ok=True)
        # Probe writability without truncating existing content
        with open(primary, "a", encoding="utf-8"):
            pass
        return primary
    except OSError:
        return Path(tempfile.gettempdir()) / "vocal.log"


def setup_logging(level: str = "INFO") -> Path | None:
    """Configure root logger with a rotated file handler plus stderr.

    Returns the resolved log file path, or None if file logging could not
    be initialised (stream handler is always configured).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Idempotent: drop any pre-existing handlers so re-calls don't stack up.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(
        logging.Formatter(LOG_FORMAT_STREAM, datefmt=LOG_DATEFMT_STREAM)
    )
    root.addHandler(stream_handler)

    log_path = _resolve_log_path()
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=LOG_ROTATE_BYTES,
            backupCount=LOG_ROTATE_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT_FILE, datefmt=LOG_DATEFMT_FILE)
        )
        root.addHandler(file_handler)
        return log_path
    except OSError as e:
        logging.getLogger(__name__).warning(
            "Could not open log file %s: %s; continuing with stderr only",
            log_path, e,
        )
        return None


def log_startup_banner(log_path: Path | None) -> None:
    """Emit a once-per-run banner with diagnostic info for bug reports."""
    from vocal import __version__

    log = logging.getLogger("vocal")
    log.info("-" * 60)
    log.info("vocal %s starting", __version__)
    log.info("python %s on %s", sys.version.split()[0], _platform.platform())
    if log_path is not None:
        log.info("log file: %s", log_path)
    else:
        log.info("log file: (stderr only — file open failed)")
    log.info("-" * 60)


def check_dependencies(output_method: str = "clipboard") -> list[str]:
    """Check for required system dependencies. Returns list of missing ones."""
    missing = []

    if sys.platform == "linux":
        if shutil.which("xdotool") is None:
            missing.append("xdotool")
        if output_method == "clipboard" and shutil.which("xclip") is None:
            missing.append("xclip")
    elif sys.platform == "darwin":
        if shutil.which("pbcopy") is None:
            missing.append("pbcopy (should be built into macOS)")
        if shutil.which("osascript") is None:
            missing.append("osascript (should be built into macOS)")
    # Windows: no external CLI tools needed

    return missing


def check_tray_dependencies() -> list[str]:
    """Check for tray-mode dependencies. Returns human-readable missing items.

    Linux requires python3-gi + AppIndicator typelib for pystray's preferred
    backend. These must be checked BEFORE importing pystray, because pystray
    crashes at import time if AppIndicator is missing.
    """
    missing: list[str] = []

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow (pip install Pillow)")

    if sys.platform == "linux":
        try:
            import gi  # noqa: F401
        except ImportError:
            missing.append("python3-gi (apt install python3-gi)")
            # Can't check further without gi; pystray will also fail.
            return missing

        # Probe for Ayatana (preferred) then legacy AppIndicator3.
        has_indicator = False
        try:
            gi.require_version("AyatanaAppIndicator3", "0.1")
            has_indicator = True
        except ValueError:
            try:
                gi.require_version("AppIndicator3", "0.1")
                has_indicator = True
            except ValueError:
                pass

        if not has_indicator:
            missing.append(
                "gir1.2-ayatanaappindicator3-0.1 "
                "(apt install gir1.2-ayatanaappindicator3-0.1)"
            )
            # Don't try to import pystray — it will crash.
            return missing

    # Only import pystray after confirming AppIndicator is available.
    try:
        import pystray  # noqa: F401
    except (ImportError, ValueError) as e:
        missing.append(f"pystray (pip install pystray) — {e}")

    return missing
