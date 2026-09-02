"""System TTS fallback: espeak-ng (Linux), ``say`` (macOS), SAPI (Windows).

Each tool renders to WAV which is decoded with the stdlib and fed through
the same :class:`~vocal.output.playback.AudioPlayer` as the neural
backends, so device selection and stop semantics are identical.
No model files are involved; :meth:`load` ignores its arguments.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from vocal.output.backends.base import BackendUnavailable, Synthesis, TTSBackend

logger = logging.getLogger(__name__)

_BASE_WPM = 175  # espeak-ng / say default speaking rate


def _tool() -> str | None:
    if sys.platform == "linux":
        return "espeak-ng" if shutil.which("espeak-ng") else ("espeak" if shutil.which("espeak") else None)
    if sys.platform == "darwin":
        return "say" if shutil.which("say") else None
    if sys.platform == "win32":
        return "powershell" if shutil.which("powershell") else None
    return None


def _decode_wav(data: bytes) -> tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(data), "rb") as w:
        rate = w.getframerate()
        width = w.getsampwidth()
        channels = w.getnchannels()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"Unsupported WAV sample width {width}")
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return rate, samples


class SystemBackend(TTSBackend):
    name = "system"

    def __init__(self) -> None:
        super().__init__()
        self._tool: str | None = None
        self._loaded = False

    @classmethod
    def is_available(cls) -> bool:
        return _tool() is not None

    def load(self, model: Path | None, style: str | None = None) -> None:
        self._tool = _tool()
        if self._tool is None:
            raise BackendUnavailable(
                "No system TTS tool found (espeak-ng on Linux, say on macOS, PowerShell on Windows)"
            )
        self._loaded = True

    def synthesize(self, text: str) -> Synthesis:
        wpm = int(_BASE_WPM * self.speed)
        if self._tool in ("espeak-ng", "espeak"):
            out = subprocess.run(
                [self._tool, "--stdout", "-s", str(wpm), "--", text],
                capture_output=True, check=True,
            ).stdout
        elif self._tool == "say":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = Path(f.name)
            try:
                subprocess.run(
                    ["say", "-o", str(path), "--data-format=LEI16@22050", "-r", str(wpm), "--", text],
                    check=True, capture_output=True,
                )
                out = path.read_bytes()
            finally:
                path.unlink(missing_ok=True)
        else:  # powershell / SAPI — untested, no Windows machine available
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = Path(f.name)
            try:
                rate = max(-10, min(10, int(round((self.speed - 1.0) * 10))))
                script = (
                    "Add-Type -AssemblyName System.Speech;"
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                    f"$s.Rate = {rate};"
                    f"$s.SetOutputToWaveFile('{path}');"
                    "$s.Speak([Console]::In.ReadToEnd());"
                    "$s.Dispose();"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script],
                    input=text.encode("utf-8"), check=True, capture_output=True,
                )
                out = path.read_bytes()
            finally:
                path.unlink(missing_ok=True)

        rate, samples = _decode_wav(out)
        return Synthesis(sample_rate=rate, chunks=iter([samples]))
