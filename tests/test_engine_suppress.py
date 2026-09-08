"""Live engine drops mic audio while suppressed and resets VAD on release."""

from __future__ import annotations

import numpy as np
import pytest

from vocal.config import VocalConfig
from vocal.input.live import LiveDictationEngine


class _Recorder:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> LiveDictationEngine:
    # No real hotkey listener / model: stub the pieces the constructor touches.
    monkeypatch.setattr("vocal.input.live.create_listener", lambda *a, **k: object())
    eng = LiveDictationEngine(VocalConfig())
    eng._vad = _Recorder()  # type: ignore[assignment]
    eng._detector = _Recorder()  # type: ignore[assignment]
    return eng


def _frame(n: int = 512) -> np.ndarray:
    return np.zeros((n, 1), dtype=np.float32)


def test_audio_dropped_while_suppressed(engine: LiveDictationEngine) -> None:
    engine._audio_callback(_frame(), 512, None, 0)  # type: ignore[arg-type]
    assert engine._raw_queue.qsize() == 1

    engine.suppress_input()
    engine._audio_callback(_frame(), 512, None, 0)  # type: ignore[arg-type]
    assert engine._raw_queue.qsize() == 1  # dropped

    engine.release_input()
    engine._audio_callback(_frame(), 512, None, 0)  # type: ignore[arg-type]
    assert engine._raw_queue.qsize() == 2


def test_release_resets_vad_state(engine: LiveDictationEngine) -> None:
    engine._preroll.append(np.zeros(8, dtype=np.float32))
    engine.suppress_input()
    engine.release_input()
    assert engine._vad.resets == 1 and engine._detector.resets == 1  # type: ignore[attr-defined]
    assert len(engine._preroll) == 0


def test_suppress_release_idempotent(engine: LiveDictationEngine) -> None:
    engine.release_input()  # not suppressed: no-op, no reset
    assert engine._vad.resets == 0  # type: ignore[attr-defined]
    engine.suppress_input()
    engine.suppress_input()
    engine.release_input()
    engine.release_input()
    assert engine._vad.resets == 1  # type: ignore[attr-defined]


def test_suppress_flushes_in_progress_utterance(engine: LiveDictationEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    flushed = []
    monkeypatch.setattr(engine, "_flush_utterance", lambda: flushed.append(True))
    engine._in_speech = True
    engine.suppress_input()
    assert flushed == [True]


def test_hotkey_engine_defaults_are_noops() -> None:
    from vocal.input.base_engine import BaseDictationEngine

    class Dummy(BaseDictationEngine):
        def run(self) -> None:  # pragma: no cover
            pass

    d = Dummy(VocalConfig())
    d.suppress_input()
    d.release_input()


# ── transcript hook + phrasebook hot-swap ──


def test_on_transcript_fires_after_inject(monkeypatch: pytest.MonkeyPatch) -> None:
    from vocal.input.base_engine import BaseDictationEngine

    class Dummy(BaseDictationEngine):
        def run(self) -> None:  # pragma: no cover
            pass

    injected: list[str] = []
    got: list[str] = []
    monkeypatch.setattr("vocal.input.base_engine.inject_text", lambda text, cfg: injected.append(text))
    d = Dummy(VocalConfig(), on_transcript=got.append)
    d._output_queue.put("hello")
    d._output_queue.put(None)
    d._output_worker()
    assert injected == ["hello"] and got == ["hello"]


def test_on_transcript_skipped_when_inject_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from vocal.input.base_engine import BaseDictationEngine

    class Dummy(BaseDictationEngine):
        def run(self) -> None:  # pragma: no cover
            pass

    def boom(text, cfg):
        raise RuntimeError("no display")

    got: list[str] = []
    monkeypatch.setattr("vocal.input.base_engine.inject_text", boom)
    d = Dummy(VocalConfig(), on_transcript=got.append)
    d._output_queue.put("hello")
    d._output_queue.put(None)
    d._output_worker()
    assert got == []


def test_set_phrasebook_updates_seed_and_replace() -> None:
    from vocal.input.base_engine import BaseDictationEngine
    from vocal.input.phrasebook import Phrasebook, _compile_replacements

    class Dummy(BaseDictationEngine):
        def run(self) -> None:  # pragma: no cover
            pass

    d = Dummy(VocalConfig())
    assert d._phrasebook is None and d._transcriber._initial_prompt is None

    rules = {"pie torch": "PyTorch"}
    pb = Phrasebook(replacements=rules, _patterns=_compile_replacements(rules))
    d.set_phrasebook(pb, seed=True, replace=True)
    assert d._phrasebook is pb
    assert d._seed_phrasebook is pb
    assert d._transcriber._initial_prompt == "PyTorch"

    d.set_phrasebook(pb, seed=False, replace=True)
    assert d._phrasebook is pb and d._seed_phrasebook is None
    assert d._transcriber._initial_prompt is None
