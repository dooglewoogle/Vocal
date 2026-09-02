"""Localhost HTTP control plane for text-to-speech.

    POST /say     {"text": "...", "interrupt": false, "voice": null}
    POST /stop
    GET  /status  -> {"speaking", "queue", "voice", "backend"}
    GET  /health  -> {"ok": true}

The server binds to loopback only and needs no credentials. Requests that
carry an ``Origin`` header are refused: browsers always add one to
cross-origin requests, so this stops a web page from driving the speaker
while leaving curl/scripts untouched. The bound port is written to a
runtime file so local clients (and ``vocal say``) can find the daemon
without configuration.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vocal.output.speech import SpeechController

logger = logging.getLogger(__name__)

_MAX_BODY = 1 << 20  # 1 MiB


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


class _Handler(BaseHTTPRequestHandler):
    server: SpeechServer  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # route to logging
        logger.debug("http: " + format, *args)

    # ── helpers ──

    def _send(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        # Browsers send Origin on every cross-origin request; curl and
        # scripts don't. Refusing it blocks web pages, nothing else.
        return self.headers.get("Origin") is None

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > _MAX_BODY:
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"})
            return None
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
            return None
        if not isinstance(data, dict):
            self._send(HTTPStatus.BAD_REQUEST, {"error": "JSON object expected"})
            return None
        return data

    # ── routes ──

    def do_GET(self) -> None:
        if not self._authorized():
            return self._send(HTTPStatus.FORBIDDEN, {"error": "browser origins are not allowed"})
        ctl = self.server.controller
        if self.path == "/health":
            return self._send(HTTPStatus.OK, {"ok": True})
        if self.path == "/status":
            return self._send(HTTPStatus.OK, {
                "speaking": ctl.is_speaking, "queue": ctl.queue_length,
                "voice": ctl.voice, "backend": ctl.backend_name,
            })
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            return self._send(HTTPStatus.FORBIDDEN, {"error": "browser origins are not allowed"})
        ctl = self.server.controller
        if self.path == "/stop":
            ctl.stop()
            return self._send(HTTPStatus.OK, {"ok": True})
        if self.path == "/say":
            data = self._read_json()
            if data is None:
                return
            text = data.get("text")
            if not isinstance(text, str) or not text.strip():
                return self._send(HTTPStatus.BAD_REQUEST, {"error": "'text' (non-empty string) required"})
            voice = data.get("voice")
            if voice is not None and not isinstance(voice, str):
                return self._send(HTTPStatus.BAD_REQUEST, {"error": "'voice' must be a string"})
            try:
                ctl.say(text, interrupt=bool(data.get("interrupt", False)), voice=voice)
            except Exception as e:  # unknown voice etc.
                return self._send(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return self._send(HTTPStatus.ACCEPTED, {"ok": True, "queue": ctl.queue_length})
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})


class SpeechServer(ThreadingHTTPServer):
    """Threaded HTTP server bound to loopback; owns the runtime file."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        controller: SpeechController,
        host: str = "127.0.0.1",
        port: int = 47821,
        runtime_file: Path | None = None,
    ) -> None:
        self.controller = controller
        self._runtime_file = runtime_file or runtime_file_path()
        self._thread: threading.Thread | None = None
        try:
            super().__init__((host, port), _Handler)
        except OSError as e:
            logger.warning("Port %d unavailable (%s); using an ephemeral port", port, e)
            super().__init__((host, 0), _Handler)

    @property
    def port(self) -> int:
        return self.server_address[1]

    @property
    def host(self) -> str:
        return str(self.server_address[0])

    def start(self) -> None:
        self._write_runtime_file()
        self._thread = threading.Thread(target=self.serve_forever, name="tts-http", daemon=True)
        self._thread.start()
        logger.info("Speech server listening on http://%s:%d (runtime file %s)",
                    self.host, self.port, self._runtime_file)

    def stop(self) -> None:
        if self._thread is not None:
            self.shutdown()
            self._thread.join(timeout=5.0)
            self._thread = None
        self.server_close()
        try:
            self._runtime_file.unlink()
        except OSError:
            pass

    def _write_runtime_file(self) -> None:
        self._runtime_file.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"host": self.host, "port": self.port, "pid": os.getpid()})
        tmp = self._runtime_file.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        os.replace(tmp, self._runtime_file)
