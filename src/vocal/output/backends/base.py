"""Abstract text-to-speech backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class BackendUnavailable(RuntimeError):
    """The backend's Python package or system tool is missing."""


@dataclass
class Synthesis:
    """Result of synthesizing one piece of text.

    ``chunks`` yields int16 mono numpy arrays at ``sample_rate``. Backends
    that synthesize incrementally yield as they go so playback can start
    before the whole utterance is rendered.
    """

    sample_rate: int
    chunks: Iterator[np.ndarray]


class TTSBackend(ABC):
    """One synthesis engine. Instances are single-threaded: the speech
    controller serializes ``load`` / ``synthesize`` on its worker thread."""

    name: str = ""

    def __init__(self) -> None:
        self.speed: float = 1.0

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Cheap import / tool check. Must not load models."""

    @abstractmethod
    def load(self, model: Path | None, style: str | None = None) -> None:
        """Load a voice.

        ``model`` is a file or directory as the backend expects (see each
        backend's docstring); ``None`` for backends with no model files.
        ``style`` selects a speaker for multi-speaker models.
        """

    @abstractmethod
    def synthesize(self, text: str) -> Synthesis:
        """Render ``text``. Never called before ``load`` succeeds."""

    def unload(self) -> None:
        """Release model memory. Default: nothing."""

    @property
    def loaded(self) -> bool:
        return getattr(self, "_loaded", False)


def float_to_int16(samples: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] audio to int16, clipping."""
    return np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)


def find_one(directory: Path, suffix: str) -> Path:
    """Return the single file in ``directory`` ending with ``suffix``.

    ``suffix`` is matched against the full filename so ``.onnx`` does not
    match ``.onnx.json``.
    """
    matches = [p for p in sorted(directory.iterdir()) if p.name.endswith(suffix)]
    if suffix == ".onnx":
        matches = [p for p in matches if not p.name.endswith(".onnx.json")]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *{suffix} in {directory}, found {len(matches)}"
        )
    return matches[0]
