"""macOS privacy permissions (TCC) for the hotkey and text injection.

macOS grants these to the *responsible* app — the terminal Vocal was started
from — not to the Python binary. Everything here imports its pyobjc modules
lazily (pynput depends on them on darwin), so the module imports anywhere.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

# (status key, name in System Settings, Settings pane anchor, why Vocal needs it)
PERMISSIONS = [
    ("input_monitoring", "Input Monitoring", "Privacy_ListenEvent", "needed for the global hotkey"),
    ("accessibility", "Accessibility", "Privacy_Accessibility", "needed to type or paste text"),
]

_SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?"

_TERM_PROGRAMS = {
    "iTerm.app": "iTerm",
    "Apple_Terminal": "Terminal",
    "vscode": "Visual Studio Code",
}

UNKNOWN_APP = "the app you started Vocal from"


def status() -> dict[str, bool | None]:
    """Whether each permission is granted. Never prompts. None = could not check."""
    out: dict[str, bool | None] = {key: None for key, *_ in PERMISSIONS}
    try:
        import Quartz

        out["input_monitoring"] = bool(Quartz.CGPreflightListenEventAccess())
    except (ImportError, AttributeError) as e:
        logger.debug("Input Monitoring check unavailable: %s", e)
    try:
        import HIServices

        out["accessibility"] = bool(HIServices.AXIsProcessTrusted())
    except (ImportError, AttributeError) as e:
        logger.debug("Accessibility check unavailable: %s", e)
    return out


def request(st: dict[str, bool | None]) -> None:
    """Ask macOS to show its permission prompts for what ``st`` reports missing.

    Only call this on an explicit user action: the Accessibility prompt can
    reappear on every call while access is missing.
    """
    try:
        if st.get("input_monitoring") is False:
            import Quartz

            Quartz.CGRequestListenEventAccess()
        if st.get("accessibility") is False:
            import HIServices

            HIServices.AXIsProcessTrustedWithOptions({HIServices.kAXTrustedCheckOptionPrompt: True})
    except (ImportError, AttributeError) as e:
        logger.warning("Could not request macOS permissions: %s", e)


def open_settings(st: dict[str, bool | None]) -> None:
    """Open the System Settings pane for each permission ``st`` reports missing."""
    for key, _name, pane, _why in PERMISSIONS:
        if st.get(key) is False:
            subprocess.run(["open", _SETTINGS_URL + pane], check=False)


def outermost_app(path: str) -> str | None:
    """Name of the outermost ``*.app`` bundle in ``path`` (VS Code helpers live inside VS Code)."""
    for part in path.split("/"):
        if part.endswith(".app"):
            return part[: -len(".app")]
    return None


def _responsible_path() -> str | None:
    """Executable of the process macOS holds responsible for us (private libSystem API)."""
    import ctypes

    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        get_responsible = libc.responsibility_get_pid_responsible_for_pid
    except (OSError, AttributeError):
        return None
    get_responsible.argtypes = [ctypes.c_int]
    get_responsible.restype = ctypes.c_int
    pid = get_responsible(os.getpid())
    if pid <= 0:
        return None
    buf = ctypes.create_string_buffer(4096)  # PROC_PIDPATHINFO_MAXSIZE
    if libc.proc_pidpath(pid, buf, ctypes.sizeof(buf)) <= 0:
        return None
    return buf.value.decode(errors="replace")


def responsible_app(env: dict[str, str] | None = None) -> str:
    """The app to switch on in System Settings, e.g. "iTerm"."""
    if sys.platform == "darwin":
        path = _responsible_path()
        if path:
            return outermost_app(path) or path
    term = (env if env is not None else os.environ).get("TERM_PROGRAM", "")
    if term:
        return _TERM_PROGRAMS.get(term, term)
    return UNKNOWN_APP


def advice(st: dict[str, bool | None], app: str) -> list[str]:
    """Next steps for the user, one sentence per line."""
    missing = [name for key, name, _pane, _why in PERMISSIONS if st.get(key) is False]
    if not missing:
        return [
            "If the hotkey still does nothing, turn off Secure Keyboard Entry in your terminal's menu: "
            "it hides keystrokes from every other app, and so do password fields.",
        ]
    return [
        f"Switch on {app} in System Settings → Privacy & Security → {' and '.join(missing)}.",
        f"Then quit {app} completely (⌘Q) and reopen it: macOS only applies the change to newly started apps.",
    ]
