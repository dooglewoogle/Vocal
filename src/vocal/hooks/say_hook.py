"""Speak ``<say>...</say>`` spans from an AI coding agent's output.

Registered by ``vocal install-agents`` as a hook command for Claude Code, Codex
CLI and Gemini CLI. The agent pipes its event JSON to stdin; we dispatch on
``hook_event_name``:

* ``MessageDisplay`` (Claude Code) streams the assistant text in batches of
  completed lines: ``delta`` per stable ``message_id``, ``final: true`` on the
  last batch. Deltas are accumulated in a small temp file so a span split across
  batches is spoken once, as soon as it closes.
* ``Stop`` (Codex CLI) carries ``last_assistant_message``.
* ``AfterAgent`` (Gemini CLI) carries ``prompt_response``.

Each utterance is handed to a detached child (``--speak TEXT``) that POSTs it to
the daemon's ``/say`` route, so the agent is never blocked on the network.
Everything is fire and forget: every failure path exits 0 with empty stdout
(Gemini rejects non-JSON stdout, Claude would show it), and nothing here
imports the speech stack, so the hook costs one bare interpreter start.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from vocal.output.runtime import read_runtime_info

SAY_RE = re.compile(r"<say>(.*?)</say>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```.*?(```|\Z)", re.DOTALL)  # unclosed fence masks to end
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
STALE_SECONDS = 3600
POST_TIMEOUT = 3.0

# hook_event_name -> field holding the complete assistant text
FINAL_TEXT_FIELD = {
    "Stop": "last_assistant_message",
    "AfterAgent": "prompt_response",
}


def spans(text: str) -> list[str]:
    """Completed <say> spans outside code, whitespace-normalised."""
    text = FENCE_RE.sub("", text)
    text = INLINE_CODE_RE.sub("", text)
    return [" ".join(s.split()).replace(":", ",") for s in SAY_RE.findall(text) if s.strip()]


def post(text: str) -> bool:
    """Synchronous POST to the daemon. Runs in the detached child only."""
    info = read_runtime_info()
    if info is None:
        return False
    url = f"http://{info.get('host', '127.0.0.1')}:{info['port']}/say"
    body = json.dumps({"text": text, "interrupt": False}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=POST_TIMEOUT):
            return True
    except Exception:  # noqa: BLE001 - daemon down, refused, timeout: all silent
        return False


def speak(text: str) -> None:
    """Fire and forget: hand ``text`` to a detached copy of this module."""
    subprocess.Popen(
        [sys.executable, "-m", "vocal.hooks.say_hook", "--speak", text],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def state_dir() -> Path:
    d = Path(tempfile.gettempdir()) / f"vocal-say-hook-{os.getuid()}"
    d.mkdir(mode=0o700, exist_ok=True)
    return d


def sweep_stale(d: Path) -> None:
    cutoff = time.time() - STALE_SECONDS
    for p in d.glob("*.json"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def handle_message_display(payload: dict) -> None:
    msg_id = re.sub(r"[^A-Za-z0-9_-]", "", str(payload.get("message_id", "")))
    if not msg_id:
        return
    d = state_dir()
    p = d / f"{payload.get('session_id', 'nosession')}-{msg_id}.json"
    try:
        st = json.loads(p.read_text())
    except (OSError, ValueError):
        st = {"text": "", "spoken": 0}

    st["text"] += payload.get("delta") or ""
    found = spans(st["text"])
    for utterance in found[st["spoken"]:]:
        speak(utterance)
    st["spoken"] = len(found)

    if payload.get("final"):
        p.unlink(missing_ok=True)
        sweep_stale(d)
    else:
        p.write_text(json.dumps(st))


def handle(payload: dict) -> None:
    event = payload.get("hook_event_name")
    if event == "MessageDisplay":
        handle_message_display(payload)
        return
    field = FINAL_TEXT_FIELD.get(event)
    if field:
        for utterance in spans(str(payload.get(field) or "")):
            speak(utterance)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--speak":
        post(argv[1])
        return
    payload = json.loads(sys.stdin.read() or "{}")
    if isinstance(payload, dict):
        handle(payload)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - a hook must never fail the agent's turn
        pass
    sys.exit(0)
