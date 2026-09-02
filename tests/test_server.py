"""HTTP control plane + client against a real server on an ephemeral port."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from vocal.output import client
from vocal.output.server import SpeechServer, read_runtime_info


class StubController:
    def __init__(self) -> None:
        self.said: list[tuple[str, bool, str | None]] = []
        self.stops = 0
        self.is_speaking = False
        self.voice = "piper-en-lessac-medium"
        self.backend_name = "piper"

    @property
    def queue_length(self) -> int:
        return len(self.said)

    def say(self, text: str, interrupt: bool = False, voice: str | None = None) -> None:
        if voice == "bad":
            raise ValueError("unknown voice")
        self.said.append((text, interrupt, voice))

    def stop(self) -> None:
        self.stops += 1


@pytest.fixture
def server(tmp_path: Path):
    ctl = StubController()
    srv = SpeechServer(ctl, port=0, runtime_file=tmp_path / "server.json")
    srv.start()
    yield srv, ctl, tmp_path / "server.json"
    srv.stop()


def _call(srv: SpeechServer, method: str, path: str, body: dict | list | None = None, origin: str | None = None):
    headers = {"Content-Type": "application/json"}
    if origin is not None:
        headers["Origin"] = origin
    req = urllib.request.Request(
        f"http://127.0.0.1:{srv.port}{path}", method=method, headers=headers,
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_runtime_file_written_0600_and_removed(server) -> None:
    srv, _, rt = server
    info = read_runtime_info(rt)
    assert info["port"] == srv.port and info["pid"] == os.getpid() and "token" not in info
    if os.name == "posix":
        assert (rt.stat().st_mode & 0o777) == 0o600
    srv.stop()
    assert not rt.exists()


def test_browser_origin_rejected(server) -> None:
    srv, ctl, _ = server
    assert _call(srv, "GET", "/health", origin="https://evil.example")[0] == 403
    assert _call(srv, "POST", "/say", {"text": "x"}, origin="http://localhost:3000")[0] == 403
    assert _call(srv, "POST", "/say", {"text": "x"}, origin="null")[0] == 403
    assert ctl.said == []


def test_health_and_status(server) -> None:
    srv, ctl, _ = server
    assert _call(srv, "GET", "/health") == (200, {"ok": True})
    status, body = _call(srv, "GET", "/status")
    assert status == 200
    assert body == {"speaking": False, "queue": 0, "voice": ctl.voice, "backend": "piper"}


def test_say_enqueues(server) -> None:
    srv, ctl, _ = server
    status, body = _call(srv, "POST", "/say", {"text": "hello", "interrupt": True, "voice": "system"})
    assert status == 202 and body["ok"] is True
    assert ctl.said == [("hello", True, "system")]


def test_say_validation(server) -> None:
    srv, _, _ = server
    assert _call(srv, "POST", "/say", {"text": ""})[0] == 400
    assert _call(srv, "POST", "/say", {"text": 5})[0] == 400
    assert _call(srv, "POST", "/say", {"text": "x", "voice": 1})[0] == 400
    assert _call(srv, "POST", "/say", {"text": "x", "voice": "bad"})[0] == 400
    assert _call(srv, "POST", "/say", ["not", "an", "object"])[0] == 400


def test_stop_and_404(server) -> None:
    srv, ctl, _ = server
    assert _call(srv, "POST", "/stop", {}) == (200, {"ok": True})
    assert ctl.stops == 1
    assert _call(srv, "GET", "/nope")[0] == 404
    assert _call(srv, "POST", "/nope", {})[0] == 404


def test_client_roundtrip(server) -> None:
    srv, ctl, rt = server
    assert client.say("via client", runtime_file=rt) is True
    assert ctl.said == [("via client", False, None)]
    assert client.stop(runtime_file=rt) is True
    assert client.status(runtime_file=rt)["backend"] == "piper"
    with pytest.raises(client.DaemonError, match="400"):
        client.say("x", voice="bad", runtime_file=rt)


def test_client_without_daemon(tmp_path: Path) -> None:
    missing = tmp_path / "none.json"
    assert client.say("x", runtime_file=missing) is False
    assert client.status(runtime_file=missing) is None
    # stale runtime file pointing at a closed port
    stale = tmp_path / "stale.json"
    srv = SpeechServer(StubController(), port=0, runtime_file=stale)
    srv.start()
    srv.stop()
    stale.write_text(json.dumps({"host": "127.0.0.1", "port": srv.port}))
    assert client.say("x", runtime_file=stale) is False


def test_port_fallback_to_ephemeral(tmp_path: Path) -> None:
    a = SpeechServer(StubController(), port=0, runtime_file=tmp_path / "a.json")
    b = SpeechServer(StubController(), port=a.port, runtime_file=tmp_path / "b.json")
    try:
        assert b.port != a.port
    finally:
        a.stop()
        b.stop()
