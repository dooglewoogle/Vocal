#!/usr/bin/env python3
"""End-to-end smoke test of the Vocal GUI against the real engine, isolated from
the user's config and any running daemon.

    PYTHONPATH=src .venv/bin/python scripts/gui_smoke.py [--speak-only]

Uses a scratch XDG_CONFIG_HOME under /tmp (hotkey F24, tiny.en, server port
47999), drives every tab through the Tk thread, takes screenshots into
<scratch>/out/, then SIGTERMs itself. Needs a display and working audio.
Prints a results log; exit code is non-zero if the driver hit an error.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SCRATCH = Path(os.environ.get("VOCAL_SMOKE_DIR", "/tmp/vocal-gui-test"))
shutil.rmtree(SCRATCH / "out", ignore_errors=True)
(SCRATCH / "vocal").mkdir(parents=True, exist_ok=True)
(SCRATCH / "rt").mkdir(parents=True, exist_ok=True)
(SCRATCH / "vocal" / "config.toml").write_text(
    '[input]\nengine = "hotkey"\n[input.model]\nsize = "tiny.en"\n'
    '[input.hotkey]\nkey = "F24"\n[output.server]\nport = 47999\n'
)
(SCRATCH / "vocal" / "phrasebook.toml").unlink(missing_ok=True)
os.environ["XDG_CONFIG_HOME"] = str(SCRATCH)
os.environ["VOCAL_RUNTIME_FILE"] = str(SCRATCH / "rt" / "server.json")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vocal.app import VocalApp  # noqa: E402
from vocal.cli import _apply_cli_overrides, parse_args  # noqa: E402
from vocal.config import CONFIG_PATH, load_config  # noqa: E402
from vocal.gui.window import run_gui  # noqa: E402
from vocal.utils import setup_logging  # noqa: E402

T0 = time.monotonic()
OUT = SCRATCH / "out"
OUT.mkdir(exist_ok=True)
RESULTS: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.monotonic() - T0:6.2f}s] {msg}"
    print(line, flush=True)
    RESULTS.append(line)


def shot(name: str) -> None:
    try:
        from PIL import ImageGrab
        ids = subprocess.run(["xdotool", "search", "--name", "^Vocal$"], capture_output=True, text=True).stdout.split()
        best = None
        for wid in ids:
            geo = subprocess.run(["xdotool", "getwindowgeometry", "--shell", wid], capture_output=True, text=True).stdout
            vals = dict(l.split("=") for l in geo.strip().splitlines() if "=" in l)
            cand = (int(vals["X"]), int(vals["Y"]), int(vals["WIDTH"]), int(vals["HEIGHT"]))
            if best is None or cand[2] * cand[3] > best[2] * best[3]:
                best = cand
        x, y, w, h = best
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(OUT / f"{name}.png")
        log(f"screenshot {name} ({w}x{h})")
    except Exception as e:
        log(f"screenshot {name} FAILED: {e!r}")


def http(method: str, path: str, body: dict | None = None) -> dict:
    port = json.loads((SCRATCH / "rt" / "server.json").read_text())["port"]
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def ui(window, fn):
    """Run fn on the Tk thread and wait for it (driver thread only)."""
    done = threading.Event()
    box: dict = {}

    def wrapped():
        try:
            box["r"] = fn()
        except Exception as e:  # noqa: BLE001
            box["e"] = e
        done.set()

    window.call_soon(wrapped)
    done.wait(5)
    if "e" in box:
        raise box["e"]
    return box.get("r")


def driver(window) -> None:
    app: VocalApp = window.app
    try:
        log(f"initial state={app.state.value} label={ui(window, lambda: window.status._state_label.cget('text'))!r}")
        for _ in range(300):
            if app.state.value == "listening":
                break
            time.sleep(0.1)
        time.sleep(0.3)
        log(f"state={app.state.value} label={ui(window, lambda: window.status._state_label.cget('text'))!r} "
            f"rebuilding={app.is_rebuilding} tray={window._has_tray}")
        shot("1-status")

        # ── HTTP say → speaking overlay ──
        r = http("POST", "/say", {"text": "Testing the Vocal window, one two three four five six."})
        t_say = time.monotonic()
        log(f"/say → {r}")
        first_speaking = first_label = None
        for _ in range(200):
            if first_speaking is None and app.is_speaking:
                first_speaking = time.monotonic() - t_say
            if first_label is None and ui(window, lambda: window.status._speaking_label.cget("text")) == "Speaking":
                first_label = time.monotonic() - t_say
            if first_speaking and first_label:
                break
            time.sleep(0.05)
        log(f"is_speaking after {first_speaking}s; label 'Speaking' after {first_label}s; "
            f"stop button={ui(window, lambda: str(window.status._stop_btn.cget('state')))}")
        shot("1b-status-speaking")
        if "--speak-only" in sys.argv:
            time.sleep(1)
            log(f"input suppressed on engine: {getattr(app._engine, '_suppressed', None)}")
            for _ in range(200):
                if not app.is_speaking:
                    break
                time.sleep(0.05)
            time.sleep(0.6)
            log(f"after speech: is_speaking={app.is_speaking} label={ui(window, lambda: window.status._speaking_label.cget('text'))!r} "
                f"released timer={app._release_timer}")
            os.kill(os.getpid(), 15)
            return
        for _ in range(200):
            if not app.is_speaking:
                break
            time.sleep(0.05)
        log(f"after speech: is_speaking={app.is_speaking} label={ui(window, lambda: window.status._speaking_label.cget('text'))!r}")

        # ── Settings: change speed + hotkey key, Save & Apply ──
        ui(window, lambda: window.notebook.select(window.settings))
        ui(window, lambda: window.settings._show_advanced.set(True) or window.settings._toggle_advanced())
        shot("2-settings")
        ui(window, lambda: window.settings._vars["output.speech.speed"].set("1.15"))
        ui(window, lambda: window.settings._vars["input.hotkey.key"].set("F23"))
        ui(window, lambda: window.settings.save_and_apply())
        time.sleep(5)
        cfg_on_disk = load_config(CONFIG_PATH)
        log(f"settings applied: speed={app.config.output.speech.speed} key={app.config.input.hotkey.key} "
            f"disk speed={cfg_on_disk.output.speech.speed} disk key={cfg_on_disk.input.hotkey.key} "
            f"status={ui(window, lambda: window.settings._status.cget('text'))!r}")
        log(f"engine after rebuild: state={app.state.value} rebuilding={app.is_rebuilding}")

        # ── Voices: select default voice, Test ──
        ui(window, lambda: window.notebook.select(window.voices))
        ui(window, lambda: window.voices._voices.selection_set("piper-en-lessac-medium"))
        shot("3-voices")
        ui(window, lambda: window.voices._test())
        time.sleep(3.5)
        log(f"voice test status: {ui(window, lambda: window.voices._voice_status.cget('text'))!r}")
        models_cached = ui(window, lambda: {i: window.voices._models.set(i, 'cached') for i in window.voices._models.get_children()})
        log(f"whisper cached marks: { {k: v for k, v in models_cached.items() if v} }")

        # ── Phrasebook: add rule, save ──
        ui(window, lambda: window.notebook.select(window.phrasebook))
        ui(window, lambda: (window.phrasebook._wrong.set("cooper netties"), window.phrasebook._right.set("Kubernetes")))
        ui(window, lambda: window.phrasebook._add())
        ui(window, lambda: window.phrasebook.save())
        time.sleep(1)
        pb_file = SCRATCH / "vocal" / "phrasebook.toml"
        log(f"phrasebook file: {pb_file.exists()} contains rule={'Kubernetes' in pb_file.read_text() if pb_file.exists() else False}; "
            f"engine pb={app.phrasebook.replacements if app.phrasebook else None}; status={ui(window, lambda: window.phrasebook._status.cget('text'))!r}")
        shot("4-phrasebook")

        # ── Hide to tray, reopen ──
        ui(window, lambda: window._on_close())
        time.sleep(0.5)
        log(f"after close: window state={ui(window, lambda: window.root.state())!r} (expect withdrawn), engine still {app.state.value}")
        window.show_soon()
        time.sleep(0.5)
        log(f"after Open: window state={ui(window, lambda: window.root.state())!r}")

        # ── /status and non-daemon threads ──
        log(f"/status → {http('GET', '/status')}")
        log("non-daemon threads: " + ", ".join(t.name for t in threading.enumerate() if not t.daemon))

        # ── SIGTERM self ──
        log("sending SIGTERM to self")
        t_sig = time.monotonic()
        os.kill(os.getpid(), 15)
        RESULTS.append(f"SIGTERM sent at {t_sig - T0:.2f}")
    except Exception as e:  # noqa: BLE001
        log(f"DRIVER ERROR: {e!r}")
        import traceback
        traceback.print_exc()
        RESULTS.append("FAILED")
        app.request_shutdown()
    finally:
        (OUT / "results.txt").write_text("\n".join(RESULTS))


def main() -> None:
    args = parse_args(["--log-level", "INFO"])
    config = load_config()
    overridden = _apply_cli_overrides(config, args)
    setup_logging("INFO")
    app = VocalApp(config, config_path=CONFIG_PATH, cli_overridden=overridden)
    run_gui(app, on_ready=lambda w: threading.Thread(target=driver, args=(w,), name="driver", daemon=True).start())
    log("run_gui returned; exiting")
    (OUT / "results.txt").write_text("\n".join(RESULTS))
    sys.exit(1 if any(r == "FAILED" or "ERROR" in r for r in RESULTS) else 0)


if __name__ == "__main__":
    main()
