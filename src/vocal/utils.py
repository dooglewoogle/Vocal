"""Utility functions — logging setup, dependency checks."""

from __future__ import annotations

import logging
import shutil
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


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
