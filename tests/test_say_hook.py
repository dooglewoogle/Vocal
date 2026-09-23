"""The agent hook speaks completed <say> spans and never fails the turn."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vocal.hooks import say_hook

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def spoken(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list:
    """Spoken texts; a span with a voice is recorded as ``(voice, text)``."""
    out: list = []
    monkeypatch.setattr(say_hook, "speak", lambda text, voice=None: out.append((voice, text) if voice else text))
    monkeypatch.setattr(say_hook.tempfile, "gettempdir", lambda: str(tmp_path))
    return out


def test_hook_module_does_not_import_speech_stack() -> None:
    """Fresh interpreter: the hook runs on every agent turn and must stay stdlib-light."""
    code = ("import sys, vocal.hooks.say_hook; "
            "print(sorted(m for m in sys.modules if m.startswith('vocal.')))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.stdout.strip() == "['vocal.hooks', 'vocal.hooks.say_hook', 'vocal.output', 'vocal.output.runtime']"


@pytest.mark.parametrize("text, expected", [
    ("<say>Hello there.</say>", ["Hello there."]),
    ("<SAY>  spaced\n out </SAY>", ["spaced out"]),
    ("<say>Ratio: 1 to 2</say>", ["Ratio, 1 to 2"]),
    ("```\n<say>in a fence</say>\n```\n<say>outside</say>", ["outside"]),
    ("`<say>inline</say>` and <say>real</say>", ["real"]),
    ("```py\n<say>unclosed fence", []),
    ("<say>   </say>", []),
    ("<say>first</say> text <say>second</say>", ["first", "second"]),
])
def test_spans(text: str, expected: list[str]) -> None:
    assert say_hook.spans(text) == [(None, t) for t in expected]


@pytest.mark.parametrize("text, expected", [
    ('<say voice="bf_emma">Hi.</say>', [("bf_emma", "Hi.")]),
    ("<SAY VOICE='piper-en_GB-vctk-medium#12' >Hi.</SAY>", [("piper-en_GB-vctk-medium#12", "Hi.")]),
    ('<say voice="">Hi.</say>', [(None, "Hi.")]),
    ('<say voice="a">one</say> <say>two</say>', [("a", "one"), (None, "two")]),
    ('```\n<say voice="a">fenced</say>\n```', []),
    ('<say voice=bare>unquoted is not a say tag</say>', []),
])
def test_spans_with_voice(text: str, expected: list) -> None:
    assert say_hook.spans(text) == expected


def test_voice_attribute_reaches_speak(spoken: list) -> None:
    say_hook.handle({"hook_event_name": "Stop", "last_assistant_message": '<say voice="am_adam">Yo.</say>'})
    base = {"hook_event_name": "MessageDisplay", "session_id": "s1", "message_id": "m2"}
    say_hook.handle({**base, "delta": '<say voice="bf_'})
    say_hook.handle({**base, "delta": 'emma">Split tag.</say>', "final": True})
    assert spoken == [("am_adam", "Yo."), ("bf_emma", "Split tag.")]


def test_codex_stop_speaks_last_message(spoken: list[str]) -> None:
    say_hook.handle({"hook_event_name": "Stop", "last_assistant_message": "Done. <say>All green.</say>"})
    assert spoken == ["All green."]


def test_gemini_after_agent_speaks_response(spoken: list[str]) -> None:
    say_hook.handle({"hook_event_name": "AfterAgent", "prompt_response": "<say>Yes.</say> Details…"})
    assert spoken == ["Yes."]


def test_unknown_event_is_silent(spoken: list[str]) -> None:
    say_hook.handle({"hook_event_name": "PreToolUse", "last_assistant_message": "<say>no</say>"})
    say_hook.handle({})
    assert spoken == []


def test_message_display_accumulates_deltas_and_speaks_each_once(spoken: list[str]) -> None:
    base = {"hook_event_name": "MessageDisplay", "session_id": "s1", "message_id": "m1"}
    say_hook.handle({**base, "delta": "Intro <say>part one"})
    assert spoken == []
    say_hook.handle({**base, "delta": " done.</say> mid <say>two</say>"})
    assert spoken == ["part one done.", "two"]
    say_hook.handle({**base, "delta": " tail", "final": True})
    assert spoken == ["part one done.", "two"]
    assert list(say_hook.state_dir().glob("*.json")) == []


def test_missing_message_id_is_ignored(spoken: list[str]) -> None:
    say_hook.handle({"hook_event_name": "MessageDisplay", "delta": "<say>x</say>"})
    assert spoken == []


def _run(stdin: str, *argv: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "vocal.hooks.say_hook", *argv], input=stdin, capture_output=True,
        text=True, timeout=20, env={**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env or {})},
    )


@pytest.mark.parametrize("stdin", ["", "not json", "[1, 2]", '{"hook_event_name": "Stop"}'])
def test_process_exits_zero_silently_on_bad_input(stdin: str) -> None:
    r = _run(stdin)
    assert r.returncode == 0 and r.stdout == ""


def test_speak_child_posts_to_daemon(tmp_path: Path) -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    got: list[dict] = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.append({"path": self.path, "body": json.loads(body)})
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):  # silence
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rt = tmp_path / "server.json"
    rt.write_text(json.dumps({"host": "127.0.0.1", "port": srv.server_address[1]}))
    try:
        r = _run("", "--speak", "Hello daemon", env={"VOCAL_RUNTIME_FILE": str(rt)})
        r2 = _run("", "--speak", "Hi Emma", "--voice", "bf_emma", env={"VOCAL_RUNTIME_FILE": str(rt)})
    finally:
        srv.shutdown()
    assert r.returncode == r2.returncode == 0 and r.stdout == r2.stdout == ""
    assert got == [
        {"path": "/say", "body": {"text": "Hello daemon", "interrupt": False}},
        {"path": "/say", "body": {"text": "Hi Emma", "interrupt": False, "voice": "bf_emma"}},
    ]


def test_speak_child_without_daemon_is_silent(tmp_path: Path) -> None:
    r = _run("", "--speak", "nobody home", env={"VOCAL_RUNTIME_FILE": str(tmp_path / "missing.json")})
    assert r.returncode == 0 and r.stdout == "" and r.stderr == ""
