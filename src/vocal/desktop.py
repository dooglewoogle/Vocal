"""Desktop integration on Linux: app-menu entry, icon, and start-at-login.

``vocal install-desktop`` (or the "Start Vocal at login" checkbox) writes
freedesktop ``.desktop`` files that point at *this* installation's ``vocal``
executable by absolute path, so a venv install works without being on PATH.
"""

from __future__ import annotations

import os
import shutil
import sys
from importlib.resources import files
from pathlib import Path

APP_ID = "vocal"


def supported() -> bool:
    return sys.platform == "linux"


def data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))


def config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))


def applications_file() -> Path:
    return data_home() / "applications" / f"{APP_ID}.desktop"


def autostart_file() -> Path:
    return config_home() / "autostart" / f"{APP_ID}.desktop"


def icon_file() -> Path:
    return data_home() / "icons" / "hicolor" / "64x64" / "apps" / f"{APP_ID}.png"


def vocal_command() -> str:
    """Absolute command that launches this installation of Vocal."""
    exe = Path(sys.executable).with_name("vocal")
    if exe.exists():
        return str(exe)
    found = shutil.which("vocal")
    if found:
        return found
    return f"{sys.executable} -m vocal"


def _entry(exec_cmd: str, autostart: bool) -> str:
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        "Name=Vocal",
        "Comment=Local dictation and text-to-speech",
        f"Exec={exec_cmd}",
        f"Icon={APP_ID}",
        "Categories=Utility;AudioVideo;",
        "StartupNotify=false",
        "Terminal=false",
    ]
    if autostart:
        lines.append("X-GNOME-Autostart-enabled=true")
    return "\n".join(lines) + "\n"


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def install(autostart: bool | None = None) -> list[Path]:
    """Install the icon and app-menu entry. ``autostart`` True/False also
    adds/removes the login entry; ``None`` leaves it as it is. Returns written paths."""
    if not supported():
        raise RuntimeError("Desktop entries are only supported on Linux")
    cmd = vocal_command()
    written: list[Path] = []
    icon = icon_file()
    icon.parent.mkdir(parents=True, exist_ok=True)
    with files("vocal").joinpath("assets", "vocal-awake.png").open("rb") as src:
        icon.write_bytes(src.read())
    written.append(icon)
    written.append(_write(applications_file(), _entry(cmd, autostart=False)))
    if autostart is True:
        written.append(_write(autostart_file(), _entry(cmd, autostart=True)))
    elif autostart is False:
        autostart_file().unlink(missing_ok=True)
    return written


def uninstall() -> list[Path]:
    """Remove everything :func:`install` wrote. Returns the paths removed."""
    removed: list[Path] = []
    for p in (autostart_file(), applications_file(), icon_file()):
        if p.exists():
            p.unlink()
            removed.append(p)
    return removed


def is_autostart_enabled() -> bool:
    return autostart_file().exists()


def set_autostart(enabled: bool) -> None:
    install(autostart=enabled)
