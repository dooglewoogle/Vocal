"""Voice registry, on-disk layout, and downloading.

Layout: ``<models_dir>/<backend>/<model_id>/<file>`` where ``models_dir``
is ``$VOCAL_MODELS_DIR`` or the platform cache dir (``~/.cache/vocal/models``
on Linux). Files are stored flat under the voice directory regardless of
their path inside the source repo.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


class VoiceNotFoundError(RuntimeError):
    """Voice name is not in the registry, or its files are missing and
    downloading is disabled or failed."""


@dataclass(frozen=True)
class VoiceSpec:
    name: str
    backend: str
    sample_rate: int
    license: str
    description: str = ""
    source: str = "hf"  # "hf" | "http" | "none"
    repo_id: str | None = None
    files: tuple[str, ...] = ()  # repo-relative paths (hf) or basenames (http)
    urls: tuple[str, ...] = ()  # for source="http", parallel to ``files``
    sha256: Mapping[str, str] = field(default_factory=dict)  # basename -> hex digest
    model_id: str | None = None  # shared directory; defaults to ``name``
    style: str | None = None  # speaker for multi-speaker models

    @property
    def dir_name(self) -> str:
        return self.model_id or self.name

    @property
    def basenames(self) -> tuple[str, ...]:
        return tuple(Path(f).name for f in self.files)


def _piper(name: str, lang_dir: str, voice: str, quality: str, desc: str, lang_code: str = "en_US") -> VoiceSpec:
    stem = f"{lang_code}-{voice}-{quality}"
    base = f"{lang_dir}/{voice}/{quality}/{stem}"
    return VoiceSpec(
        name=name, backend="piper", sample_rate=22050 if quality != "low" else 16000,
        license="MIT (voice) / GPL-3.0 (engine)", description=desc,
        source="hf", repo_id="rhasspy/piper-voices",
        files=(f"{base}.onnx", f"{base}.onnx.json"),
    )


_KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"


def _kokoro(style: str, desc: str) -> VoiceSpec:
    return VoiceSpec(
        name=f"kokoro-{style}", backend="kokoro", sample_rate=24000,
        license="Apache-2.0", description=desc,
        # Canonical files for kokoro-onnx live on GitHub Releases, not HF.
        # (onnx-community/Kokoro-82M-v1.0-ONNX on HF is an incompatible export.)
        source="http",
        files=("kokoro-v1.0.onnx", "voices-v1.0.bin"),
        urls=(f"{_KOKORO_RELEASE}/kokoro-v1.0.onnx", f"{_KOKORO_RELEASE}/voices-v1.0.bin"),
        model_id="kokoro-v1", style=style,
    )


VOICES: dict[str, VoiceSpec] = {
    v.name: v
    for v in (
        _piper("piper-en-lessac-medium", "en/en_US", "lessac", "medium", "US English, neutral — fast default"),
        _piper("piper-en-amy-low", "en/en_US", "amy", "low", "US English, smallest/fastest"),
        _piper("piper-en-gb-alan-medium", "en/en_GB", "alan", "medium", "British English", lang_code="en_GB"),
        _kokoro("af_sarah", "US English female, higher quality"),
        _kokoro("am_adam", "US English male, higher quality"),
        _kokoro("bf_emma", "British English female, higher quality"),
        VoiceSpec(
            name="system", backend="system", sample_rate=0, license="n/a",
            description="OS text-to-speech (espeak-ng / say / SAPI)", source="none",
        ),
    )
}


# ── Paths ───────────────────────────────────────────────────────────


def models_dir() -> Path:
    env = os.environ.get("VOCAL_MODELS_DIR")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "vocal" / "models"


def voice_dir(spec: VoiceSpec) -> Path:
    return models_dir() / spec.backend / spec.dir_name


def get_voice(name: str) -> VoiceSpec:
    try:
        return VOICES[name]
    except KeyError:
        raise VoiceNotFoundError(
            f"Unknown voice {name!r}. Known: {', '.join(sorted(VOICES))}"
        ) from None


def is_downloaded(spec: VoiceSpec) -> bool:
    if spec.source == "none":
        return True
    d = voice_dir(spec)
    return all((d / b).exists() for b in spec.basenames)


# ── Download ────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _verify(spec: VoiceSpec, path: Path) -> None:
    expected = spec.sha256.get(path.name)
    if not expected:
        logger.debug("No sha256 pin for %s; skipping verification", path.name)
        return
    actual = _sha256(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise VoiceNotFoundError(f"Checksum mismatch for {path.name}: {actual} != {expected}")


def _fetch_hf(spec: VoiceSpec, dest: Path, progress: ProgressFn) -> None:
    from huggingface_hub import hf_hub_download

    for rel in spec.files:
        target = dest / Path(rel).name
        if target.exists():
            continue
        progress(f"Downloading {Path(rel).name} from {spec.repo_id}")
        got = Path(hf_hub_download(spec.repo_id, rel, local_dir=str(dest)))
        if got.resolve() != target.resolve():
            shutil.move(str(got), str(target))
    # hf_hub_download(local_dir=...) leaves the repo-relative folders + a
    # .cache dir behind; flatten.
    for extra in (dest / Path(spec.files[0]).parts[0], dest / ".cache"):
        if extra.is_dir() and extra != dest:
            shutil.rmtree(extra, ignore_errors=True)


def _fetch_http(spec: VoiceSpec, dest: Path, progress: ProgressFn) -> None:
    for name, url in zip(spec.basenames, spec.urls):
        target = dest / name
        if target.exists():
            continue
        progress(f"Downloading {name}")
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        tmp.replace(target)


def download_voice(name: str, progress: ProgressFn | None = None) -> Path:
    """Ensure ``name``'s files are present; return its directory."""
    spec = get_voice(name)
    report = progress or (lambda msg: logger.info("%s", msg))
    dest = voice_dir(spec)
    if spec.source == "none":
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if spec.source == "hf":
            _fetch_hf(spec, dest, report)
        elif spec.source == "http":
            _fetch_http(spec, dest, report)
        else:
            raise VoiceNotFoundError(f"Voice {name!r} has unknown source {spec.source!r}")
    except VoiceNotFoundError:
        raise
    except Exception as e:
        raise VoiceNotFoundError(f"Download of {name!r} failed: {e}") from e
    for b in spec.basenames:
        _verify(spec, dest / b)
    report(f"Voice {name} ready in {dest}")
    return dest


def remove_voice(name: str) -> bool:
    spec = get_voice(name)
    d = voice_dir(spec)
    if spec.source == "none" or not d.exists():
        return False
    shutil.rmtree(d)
    return True


def resolve_model_path(voice: str, model_path: str | None, auto_download: bool,
                       progress: ProgressFn | None = None) -> tuple[Path | None, VoiceSpec]:
    """Return ``(path_for_backend.load, spec)`` honouring a manual override.

    A configured ``model_path`` bypasses the registry (the backend is still
    taken from ``voice``'s spec) and must exist.
    """
    spec = get_voice(voice)
    if model_path:
        p = Path(model_path).expanduser()
        if not p.exists():
            raise VoiceNotFoundError(f"output.speech.model_path does not exist: {p}")
        return p, spec
    if spec.source == "none":
        return None, spec
    if not is_downloaded(spec):
        if not auto_download:
            raise VoiceNotFoundError(
                f"Voice {voice!r} is not downloaded; run `vocal models download {voice}` "
                "or set output.speech.auto_download = true"
            )
        download_voice(voice, progress)
    return voice_dir(spec), spec
