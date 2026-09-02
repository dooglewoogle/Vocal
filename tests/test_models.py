"""Voice registry paths, download plumbing (mocked), overrides."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vocal.output import models
from vocal.output.models import (
    VOICES,
    VoiceNotFoundError,
    download_voice,
    get_voice,
    is_downloaded,
    remove_voice,
    resolve_model_path,
    voice_dir,
)


@pytest.fixture(autouse=True)
def _models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VOCAL_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path / "models"


def test_registry_sanity() -> None:
    for spec in VOICES.values():
        assert spec.backend in ("piper", "kokoro", "system")
        if spec.source == "hf":
            assert spec.repo_id and spec.files
        if spec.source == "http":
            assert len(spec.urls) == len(spec.files) > 0
    assert get_voice("system").source == "none"
    with pytest.raises(VoiceNotFoundError, match="Unknown voice"):
        get_voice("nope")


def test_voice_dir_layout(_models_dir: Path) -> None:
    piper = get_voice("piper-en-lessac-medium")
    assert voice_dir(piper) == _models_dir / "piper" / "piper-en-lessac-medium"
    # kokoro voices share one model directory
    assert voice_dir(get_voice("kokoro-af_sarah")) == voice_dir(get_voice("kokoro-am_adam"))
    assert voice_dir(get_voice("kokoro-af_sarah")).name == "kokoro-v1"


def test_hf_download_flattens(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = get_voice("piper-en-lessac-medium")
    calls: list[tuple[str, str]] = []

    def fake_hf_hub_download(repo_id: str, filename: str, local_dir: str) -> str:
        calls.append((repo_id, filename))
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"model")
        (Path(local_dir) / ".cache").mkdir(exist_ok=True)
        return str(p)

    import types, sys
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_hf_hub_download))

    assert not is_downloaded(spec)
    dest = download_voice(spec.name)
    assert dest == voice_dir(spec)
    assert sorted(p.name for p in dest.iterdir()) == sorted(spec.basenames)
    assert is_downloaded(spec)
    assert [f for _, f in calls] == list(spec.files)

    # second call is a no-op
    calls.clear()
    download_voice(spec.name)
    assert calls == []

    assert remove_voice(spec.name) is True
    assert not is_downloaded(spec)
    assert remove_voice(spec.name) is False


def test_http_download_with_checksum(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"kokoro-bytes"
    good = hashlib.sha256(payload).hexdigest()
    spec = get_voice("kokoro-af_sarah")
    pinned = models.VoiceSpec(**{**spec.__dict__, "sha256": {spec.basenames[0]: good}})
    monkeypatch.setitem(models.VOICES, spec.name, pinned)

    import io

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(models.urllib.request, "urlopen", lambda url, timeout: Resp(payload))
    dest = download_voice(spec.name)
    assert (dest / spec.basenames[0]).read_bytes() == payload
    assert not list(dest.glob("*.part"))


def test_http_checksum_mismatch_removes_file(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = get_voice("kokoro-af_sarah")
    pinned = models.VoiceSpec(**{**spec.__dict__, "sha256": {spec.basenames[0]: "0" * 64}})
    monkeypatch.setitem(models.VOICES, spec.name, pinned)
    import io

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(models.urllib.request, "urlopen", lambda url, timeout: Resp(b"x"))
    with pytest.raises(VoiceNotFoundError, match="Checksum mismatch"):
        download_voice(spec.name)
    assert not (voice_dir(spec) / spec.basenames[0]).exists()


def test_resolve_model_path_override(tmp_path: Path) -> None:
    manual = tmp_path / "custom.onnx"
    manual.write_bytes(b"")
    path, spec = resolve_model_path("piper-en-amy-low", str(manual), auto_download=False)
    assert path == manual and spec.backend == "piper"
    with pytest.raises(VoiceNotFoundError, match="model_path does not exist"):
        resolve_model_path("piper-en-amy-low", str(tmp_path / "missing"), auto_download=False)


def test_resolve_model_path_no_auto_download() -> None:
    with pytest.raises(VoiceNotFoundError, match="vocal models download"):
        resolve_model_path("piper-en-amy-low", None, auto_download=False)


def test_resolve_model_path_system() -> None:
    path, spec = resolve_model_path("system", None, auto_download=False)
    assert path is None and spec.backend == "system"


def test_resolve_model_path_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(models, "download_voice", lambda name, progress=None: called.append(name))
    monkeypatch.setattr(models, "is_downloaded", lambda spec: False)
    path, spec = resolve_model_path("piper-en-amy-low", None, auto_download=True)
    assert called == ["piper-en-amy-low"] and path == voice_dir(spec)
