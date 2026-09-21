"""Location and contents of the daemon's runtime file (``server.json``).

Standard library only: the agent hook (``vocal.hooks.say_hook``) imports this on
every agent turn, so nothing here may pull in the speech or audio stack.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def runtime_file_path() -> Path:
    env = os.environ.get("VOCAL_RUNTIME_FILE")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "vocal"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "vocal"
    else:
        rt = os.environ.get("XDG_RUNTIME_DIR")
        base = Path(rt) / "vocal" if rt else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vocal"
    return base / "server.json"


def read_runtime_info(path: Path | None = None) -> dict | None:
    p = path or runtime_file_path()
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "port" not in data:
        return None
    return data
