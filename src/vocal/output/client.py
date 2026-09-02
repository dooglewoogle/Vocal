"""Thin urllib client for a running vocal daemon. Never raises for
"no daemon" — returns ``None``/``False`` so callers can fall back."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

from vocal.output.server import read_runtime_info

logger = logging.getLogger(__name__)


class DaemonError(RuntimeError):
    """The daemon answered with an error (4xx/5xx)."""


def _request(method: str, path: str, body: dict | None = None, timeout: float = 2.0,
             runtime_file: Path | None = None) -> dict | None:
    info = read_runtime_info(runtime_file)
    if info is None:
        return None
    url = f"http://{info.get('host', '127.0.0.1')}:{info['port']}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {info['token']}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read() or b"{}").get("error", e.reason)
        except ValueError:
            msg = e.reason
        raise DaemonError(f"{e.code}: {msg}") from None
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        logger.debug("Daemon unreachable at %s: %s", url, e)
        return None


def say(text: str, interrupt: bool = False, voice: str | None = None, **kw) -> bool:
    body: dict = {"text": text, "interrupt": interrupt}
    if voice:
        body["voice"] = voice
    return _request("POST", "/say", body, **kw) is not None


def stop(**kw) -> bool:
    return _request("POST", "/stop", {}, **kw) is not None


def status(**kw) -> dict | None:
    return _request("GET", "/status", **kw)
