"""Voice registry, on-disk layout, and downloading.

Layout: ``<models_dir>/<backend>/<model_id>/<file>`` where ``models_dir``
is ``$VOCAL_MODELS_DIR`` or the platform cache dir (``~/.cache/vocal/models``
on Linux). Files are stored flat under the voice directory regardless of
their path inside the source repo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
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
    license: str
    description: str = ""
    language: str = ""  # locale code, e.g. "en_GB"
    language_name: str = ""  # e.g. "English (Great Britain)"
    size_bytes: int = 0  # total download; shared by every Kokoro voice
    speakers: int = 1  # >1: multi-speaker Piper model, speaker picked with "#N"
    source: str = "hf"  # "hf" | "http" | "none"
    repo_id: str | None = None
    files: tuple[str, ...] = ()  # repo-relative paths (hf) or basenames (http)
    urls: tuple[str, ...] = ()  # for source="http", parallel to ``files``
    sha256: Mapping[str, str] = field(default_factory=dict)  # basename -> hex digest
    md5: Mapping[str, str] = field(default_factory=dict)  # basename -> hex digest
    model_id: str | None = None  # shared directory; defaults to ``name``
    style: str | None = None  # Kokoro speaker name, or Piper speaker id as a string

    @property
    def dir_name(self) -> str:
        return self.model_id or self.name

    @property
    def basenames(self) -> tuple[str, ...]:
        return tuple(Path(f).name for f in self.files)


def _piper_voices() -> list[VoiceSpec]:
    """Every Piper voice, from the vendored catalogue (scripts/gen_piper_voices.py)."""
    catalogue = json.loads((Path(__file__).with_name("piper_voices.json")).read_text(encoding="utf-8"))
    specs = []
    for v in catalogue:
        speakers = v["speakers"]
        desc = f"{v['quality'].replace('_', '-')} quality"
        if speakers > 1:
            desc += f", {speakers} speakers (#0-#{speakers - 1})"
        specs.append(VoiceSpec(
            name=f"piper-{v['key']}", backend="piper", license="MIT (voice) / GPL-3.0 (engine)",
            description=desc, language=v["lang"], language_name=v["language"],
            size_bytes=sum(size for _, size, _ in v["files"]), speakers=speakers,
            source="hf", repo_id="rhasspy/piper-voices",
            files=tuple(path for path, _, _ in v["files"]),
            md5={Path(path).name: digest for path, _, digest in v["files"]},
        ))
    return specs


_KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
# Pinned from the model-files-v1.0 release assets (sizes 325532387 / 28214398 bytes).
_KOKORO_SHA256 = {
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
_KOKORO_SIZE = 325532387 + 28214398

# First letter of a Kokoro style -> (locale, language name, espeak language).
KOKORO_LANGUAGES: dict[str, tuple[str, str, str]] = {
    "a": ("en_US", "English (United States)", "en-us"),
    "b": ("en_GB", "English (Great Britain)", "en-gb"),
    "e": ("es_ES", "Spanish (Spain)", "es"),
    "f": ("fr_FR", "French (France)", "fr-fr"),
    "h": ("hi_IN", "Hindi (India)", "hi"),
    "i": ("it_IT", "Italian (Italy)", "it"),
    "j": ("ja_JP", "Japanese (Japan)", "ja"),
    "p": ("pt_BR", "Portuguese (Brazil)", "pt-br"),
    "z": ("zh_CN", "Chinese (China)", "cmn"),
}

# Every speaker in voices-v1.0.bin.
KOKORO_STYLES = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
    "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
)


def _kokoro(style: str) -> VoiceSpec:
    locale, language, _ = KOKORO_LANGUAGES[style[0]]
    desc = "female" if style[1] == "f" else "male"
    if style[0] in "jz":
        desc += " (experimental: espeak reads kana and hanzi poorly)"
    return VoiceSpec(
        name=f"kokoro-{style}", backend="kokoro", license="Apache-2.0", description=desc,
        language=locale, language_name=language, size_bytes=_KOKORO_SIZE,
        # Canonical files for kokoro-onnx live on GitHub Releases, not HF.
        # (onnx-community/Kokoro-82M-v1.0-ONNX on HF is an incompatible export.)
        source="http",
        files=("kokoro-v1.0.onnx", "voices-v1.0.bin"),
        urls=(f"{_KOKORO_RELEASE}/kokoro-v1.0.onnx", f"{_KOKORO_RELEASE}/voices-v1.0.bin"),
        sha256=_KOKORO_SHA256,
        model_id="kokoro-v1", style=style,
    )


VOICES: dict[str, VoiceSpec] = {
    v.name: v
    for v in (
        *(_kokoro(s) for s in KOKORO_STYLES),
        *_piper_voices(),
        VoiceSpec(
            name="system", backend="system", license="n/a",
            description="OS text-to-speech (espeak-ng / say / SAPI)", source="none",
        ),
    )
}

DEFAULT_VOICE = "kokoro-bf_emma"

# Lower-cased canonical and bare names ("kokoro-af_sarah", "af_sarah",
# "piper-en_us-lessac-medium", "en_us-lessac-medium") -> canonical name.
_LOOKUP: dict[str, str] = {}
for _name in VOICES:
    _LOOKUP[_name.lower()] = _name
    _LOOKUP[_name.split("-", 1)[-1].lower()] = _name


def resolve_voice(name: str | None) -> tuple[str, VoiceSpec] | None:
    """Look up a voice by canonical or bare name, in any case, with an
    optional ``#N`` speaker suffix for multi-speaker Piper models.

    Returns ``(canonical_name, spec)``, the spec's ``style`` carrying the
    speaker, or None if ``name`` is empty or matches nothing.
    """
    if not name or not name.strip():
        return None
    base, _, speaker = name.strip().partition("#")
    canonical = _LOOKUP.get(base.lower())
    if canonical is None:
        return None
    spec = VOICES[canonical]
    if not speaker:
        return canonical, spec
    if spec.speakers < 2 or not speaker.isdigit() or int(speaker) >= spec.speakers:
        return None
    return f"{canonical}#{int(speaker)}", replace(spec, style=str(int(speaker)))

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


def matches(spec: VoiceSpec, needle: str) -> bool:
    """Case-insensitive substring match over name, language and description."""
    needle = needle.strip().lower()
    haystack = f"{spec.name} {spec.language} {spec.language_name} {spec.description}".lower()
    return needle in haystack


def human_size(n: int) -> str:
    return f"{n / 1e6:.0f} MB" if n else ""


def get_voice(name: str) -> VoiceSpec:
    found = resolve_voice(name)
    if found is None:
        raise VoiceNotFoundError(f"Unknown voice {name!r}; see `vocal models list`")
    return found[1]


def is_downloaded(spec: VoiceSpec) -> bool:
    if spec.source == "none":
        return True
    d = voice_dir(spec)
    return all((d / b).exists() for b in spec.basenames)


# ── Download ────────────────────────────────────────────────────────


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo, usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _verify(spec: VoiceSpec, path: Path) -> None:
    if path.name in spec.sha256:
        algo, expected = "sha256", spec.sha256[path.name]
    elif path.name in spec.md5:
        algo, expected = "md5", spec.md5[path.name]
    else:
        logger.debug("No checksum pin for %s; skipping verification", path.name)
        return
    actual = _digest(path, algo)
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
