"""VocalSpike probe: record what macOS lets this process do.

Started by the VocalSpike.app launcher (see README.md). Appends everything to
~/Library/Logs/VocalSpike.log, since an app started with `open` has no terminal.

    probe.py --mode MODE [paste]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vocal import macos_perms  # noqa: E402

LOG = Path.home() / "Library/Logs/VocalSpike.log"
KEY_SECONDS = 15


def log(msg: str) -> None:
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} probe[{os.getpid()}]: {msg}\n")


def announce(text: str) -> None:
    subprocess.run(["say", text], check=False)


def step(name: str, fn) -> None:
    try:
        fn()
    except Exception as e:  # a failing step must not hide the others
        log(f"{name}: FAILED {type(e).__name__}: {e}")


def identity() -> None:
    log(f"executable={sys.executable} real={os.path.realpath(sys.executable)} ppid={os.getppid()} "
        f"TERM_PROGRAM={os.environ.get('TERM_PROGRAM', '')!r}")
    log(f"responsible path={macos_perms._responsible_path()!r} app={macos_perms.responsible_app()!r}")


def permissions() -> None:
    import Quartz

    log(f"status={macos_perms.status()} post_events={bool(Quartz.CGPreflightPostEventAccess())}")


def keyboard() -> None:
    from pynput.keyboard import Listener

    count = 0

    def on_press(_key) -> None:
        nonlocal count
        count += 1

    listener = Listener(on_press=on_press)
    listener.start()
    time.sleep(1.0)
    if not listener.is_alive():
        log("keyboard: event tap REFUSED (listener thread ended)")
        return
    announce(f"Type some keys in any app for {KEY_SECONDS} seconds")
    time.sleep(KEY_SECONDS)
    listener.stop()
    log(f"keyboard: tap created, {count} key presses seen")


def microphone() -> None:
    import numpy as np
    import sounddevice as sd

    audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    rms = float(np.sqrt(np.mean(np.square(audio))))
    log(f"microphone: rms={rms:.5f} ({'silent: probably denied' if rms == 0 else 'got audio'})")


def _post_keys(events: list[tuple[int, bool, int, str | None]]) -> None:
    """Post (keycode, down, flags, unicode text) keyboard events."""
    import Quartz

    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for code, down, flags, text in events:
        ev = Quartz.CGEventCreateKeyboardEvent(src, code, down)
        Quartz.CGEventSetFlags(ev, flags)  # explicit, so held modifiers don't leak in
        if text is not None:
            Quartz.CGEventKeyboardSetUnicodeString(ev, len(text), text)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def paste() -> None:
    import Quartz

    old = subprocess.run(["pbpaste"], capture_output=True).stdout
    subprocess.run(["pbcopy"], input="vocal spike paste test ".encode(), check=True)
    announce("Click into a TextEdit window now")
    time.sleep(5)
    cmd = Quartz.kCGEventFlagMaskCommand
    _post_keys([(9, True, cmd, None), (9, False, cmd, None)])  # kVK_ANSI_V
    time.sleep(0.5)  # CGEventPost is asynchronous: restoring sooner can paste the old clipboard
    subprocess.run(["pbcopy"], input=old, check=False)
    typed = "typed: héllo ✓"
    chunks = [typed[i:i + 20] for i in range(0, len(typed), 20)]
    _post_keys([(0, down, 0, c) for c in chunks for down in (True, False)])
    log("paste: posted Cmd+V then unicode typing; tester reports what appeared in TextEdit")


def main() -> None:
    args = sys.argv[1:]
    mode = args[args.index("--mode") + 1] if "--mode" in args else "direct"
    log(f"==== start mode={mode} args={args}")
    step("identity", identity)
    step("permissions", permissions)
    step("keyboard", keyboard)
    step("microphone", microphone)
    if "paste" in args:
        step("paste", paste)
    log("==== done")
    announce("Probe done")


if __name__ == "__main__":
    main()
