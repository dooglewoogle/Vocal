"""Voice registry paths, download plumbing (mocked), overrides."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from vocal.config import SpeechConfig
from vocal.output import models
from vocal.output.models import (
    DEFAULT_VOICE,
    KOKORO_LANGUAGES,
    KOKORO_STYLES,
    VOICES,
    VoiceNotFoundError,
    download_voice,
    get_voice,
    is_downloaded,
    remove_voice,
    resolve_model_path,
    resolve_voice,
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


def test_registry_lists_every_upstream_voice() -> None:
    by_backend = Counter(spec.backend for spec in VOICES.values())
    assert by_backend == {"piper": 177, "kokoro": 54, "system": 1}
    for spec in VOICES.values():
        if spec.backend == "piper":
            assert spec.language and spec.language_name and spec.size_bytes > 0
            assert set(spec.md5) == set(spec.basenames)  # every Piper file is pinned
    assert get_voice(DEFAULT_VOICE).language == "en_GB"
    assert SpeechConfig().voice == DEFAULT_VOICE


@pytest.mark.parametrize("name, canonical, style", [
    ("kokoro-af_sarah", "kokoro-af_sarah", "af_sarah"),
    ("af_sarah", "kokoro-af_sarah", "af_sarah"),
    ("KOKORO-BF_EMMA", "kokoro-bf_emma", "bf_emma"),
    ("piper-en_US-lessac-medium", "piper-en_US-lessac-medium", None),
    ("en_us-lessac-medium", "piper-en_US-lessac-medium", None),
    ("piper-en_GB-vctk-medium#12", "piper-en_GB-vctk-medium#12", "12"),
    (" system ", "system", None),
])
def test_resolve_voice(name: str, canonical: str, style: str | None) -> None:
    found = resolve_voice(name)
    assert found is not None
    assert found[0] == canonical and found[1].style == style


@pytest.mark.parametrize("name", [
    None, "", "  ", "nope",
    "piper-en-lessac-medium",  # pre-0.5 name: no aliases
    "piper-en_GB-vctk-medium#109",  # speaker out of range
    "piper-en_US-lessac-medium#0",  # single-speaker model
    "kokoro-af_sarah#1", "piper-en_GB-vctk-medium#x",
])
def test_resolve_voice_rejects(name: str | None) -> None:
    assert resolve_voice(name) is None


def test_speaker_variant_shares_model_dir() -> None:
    assert voice_dir(get_voice("piper-en_GB-vctk-medium#3")) == voice_dir(get_voice("piper-en_GB-vctk-medium"))


def test_kokoro_languages() -> None:
    assert {s[0] for s in KOKORO_STYLES} == set(KOKORO_LANGUAGES)
    assert get_voice("bf_emma").language == "en_GB"
    assert get_voice("ef_dora").language_name == "Spanish (Spain)"


def test_voice_dir_layout(_models_dir: Path) -> None:
    piper = get_voice("piper-en_US-lessac-medium")
    assert voice_dir(piper) == _models_dir / "piper" / "piper-en_US-lessac-medium"
    # kokoro voices share one model directory
    assert voice_dir(get_voice("kokoro-af_sarah")) == voice_dir(get_voice("kokoro-am_adam"))
    assert voice_dir(get_voice("kokoro-af_sarah")).name == "kokoro-v1"


def _fake_hf(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, str]], payload: bytes = b"model") -> None:
    def fake_hf_hub_download(repo_id: str, filename: str, local_dir: str) -> str:
        calls.append((repo_id, filename))
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
        (Path(local_dir) / ".cache").mkdir(exist_ok=True)
        return str(p)

    import sys
    import types
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_hf_hub_download))


def test_hf_download_md5_mismatch_removes_file(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = get_voice("piper-en_US-lessac-medium")  # real upstream md5 pins
    _fake_hf(monkeypatch, [])
    with pytest.raises(VoiceNotFoundError, match="Checksum mismatch"):
        download_voice(spec.name)
    assert not is_downloaded(spec)


def test_hf_download_flattens(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = get_voice("piper-en_US-lessac-medium")
    digest = hashlib.md5(b"model").hexdigest()
    spec = models.VoiceSpec(**{**spec.__dict__, "md5": {b: digest for b in spec.basenames}})
    monkeypatch.setitem(models.VOICES, spec.name, spec)
    calls: list[tuple[str, str]] = []
    _fake_hf(monkeypatch, calls)

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
    path, spec = resolve_model_path("piper-en_US-amy-low", str(manual), auto_download=False)
    assert path == manual and spec.backend == "piper"
    with pytest.raises(VoiceNotFoundError, match="model_path does not exist"):
        resolve_model_path("piper-en_US-amy-low", str(tmp_path / "missing"), auto_download=False)


def test_resolve_model_path_no_auto_download() -> None:
    with pytest.raises(VoiceNotFoundError, match="vocal models download"):
        resolve_model_path("piper-en_US-amy-low", None, auto_download=False)


def test_resolve_model_path_system() -> None:
    path, spec = resolve_model_path("system", None, auto_download=False)
    assert path is None and spec.backend == "system"


def test_resolve_model_path_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(models, "download_voice", lambda name, progress=None: called.append(name))
    monkeypatch.setattr(models, "is_downloaded", lambda spec: False)
    path, spec = resolve_model_path("piper-en_US-amy-low", None, auto_download=True)
    assert called == ["piper-en_US-amy-low"] and path == voice_dir(spec)
