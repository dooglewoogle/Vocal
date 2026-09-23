"""SpeechController queueing / interrupt / callbacks with fake backend + player."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from vocal.config import SpeechConfig
from vocal.output.backends.base import Synthesis, TTSBackend
from vocal.output import speech
from vocal.output.models import DEFAULT_VOICE, VoiceNotFoundError, VoiceSpec, get_voice
from vocal.output.speech import SayResult, SpeechController, split_sentences


class FakeBackend(TTSBackend):
    name = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []
        self._loaded = True

    @classmethod
    def is_available(cls) -> bool:
        return True

    def load(self, model: Path | None, style: str | None = None) -> None:
        self._loaded = True

    def synthesize(self, text: str) -> Synthesis:
        self.texts.append(text)
        return Synthesis(sample_rate=16000, chunks=iter([np.zeros(160, dtype=np.int16)]))


class SlowBackend(FakeBackend):
    """Every synthesis after the first blocks for ``delay`` (uninterruptible, like Kokoro)."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay
        self.rendering = threading.Event()  # set while a slow render is in progress

    def synthesize(self, text: str) -> Synthesis:
        out = super().synthesize(text)
        if len(self.texts) > 1:
            self.rendering.set()
            time.sleep(self.delay)
            self.rendering.clear()
        return out


class FakePlayer:
    """Records plays; each play "takes" ``delay`` seconds unless aborted."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.played: list[str] = []
        self.times: list[float] = []
        self.aborts = 0
        self._abort = threading.Event()

    def play(self, chunks: Iterable[np.ndarray], sample_rate: int, gain: float = 1.0,
             on_first_audio=None) -> bool:
        list(chunks)
        if on_first_audio:
            on_first_audio()
        self.played.append(f"{sample_rate}:{gain}")
        self.times.append(time.monotonic())
        if self._abort.wait(self.delay):
            return False
        return True

    def reset(self) -> None:
        self._abort.clear()

    def abort(self) -> None:
        self.aborts += 1
        self._abort.set()


def _controller(player: FakePlayer | None = None, **cfg) -> tuple[SpeechController, FakeBackend, FakePlayer, list[str]]:
    backend = FakeBackend()
    player = player or FakePlayer()
    events: list[str] = []
    ctl = SpeechController(
        SpeechConfig(**cfg), player=player, backend=backend,
        on_speech_start=lambda: events.append("start"),
        on_speech_end=lambda: events.append("end"),
    )
    _offline(ctl)
    return ctl, backend, player, events


def _offline(ctl: SpeechController, fail: tuple[str, ...] = ()) -> list[tuple[str, str | None]]:
    """Resolve voices without touching the models dir; record (voice, model_path)
    per load and raise for voices in ``fail``."""
    located: list[tuple[str, str | None]] = []

    def locate(voice: str, model_path: str | None) -> tuple[Path | None, VoiceSpec]:
        located.append((voice, model_path))
        if voice in fail:
            raise VoiceNotFoundError(f"{voice} is not downloaded")
        return None, get_voice(voice)

    ctl._locate = locate  # type: ignore[method-assign]
    return located


def _wait_idle(ctl: SpeechController, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ctl.queue_length == 0 and not ctl.is_speaking and ctl._idle.is_set():
            return
        time.sleep(0.005)
    raise AssertionError("controller did not go idle")


# ── split_sentences ──


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello world.", ["Hello world."]),
        ("One. Two! Three?", ["One.", "Two!", "Three?"]),
        ("Line one\nLine two", ["Line one", "Line two"]),
        ("  spaced   ", ["spaced"]),
        ("", []),
        ("No terminal punctuation", ["No terminal punctuation"]),
        ("v1.2 is out. Really.", ["v1.2 is out.", "Really."]),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


# ── queueing ──


def test_fifo_order_and_sentence_streaming() -> None:
    ctl, backend, player, events = _controller()
    ctl.say("First one. Still first.")
    ctl.say("Second.")
    _wait_idle(ctl)
    ctl.shutdown()
    assert backend.texts == ["First one.", "Still first.", "Second."]
    assert len(player.played) == 3


def test_start_end_fire_once_per_run() -> None:
    ctl, _, _, events = _controller()
    ctl.say("A. B. C.")
    _wait_idle(ctl)
    assert events == ["start", "end"]
    ctl.say("Again.")
    _wait_idle(ctl)
    ctl.shutdown()
    assert events == ["start", "end", "start", "end"]


def test_empty_text_ignored() -> None:
    ctl, backend, _, events = _controller()
    ctl.say("   ")
    time.sleep(0.05)
    ctl.shutdown()
    assert backend.texts == [] and events == []


def test_gain_from_volume() -> None:
    ctl, _, player, _ = _controller(volume=50)
    ctl.say("Hi.")
    _wait_idle(ctl)
    ctl.shutdown()
    assert player.played == ["16000:0.5"]


def _wait_for(pred, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met in time")


def test_next_sentences_render_while_first_plays() -> None:
    player = FakePlayer(delay=0.5)
    ctl, backend, player, events = _controller(player)
    ctl.say("A. B. C.")
    _wait_for(lambda: ctl.is_speaking)
    # Lookahead: B and C are synthesized while A is still "playing".
    _wait_for(lambda: len(backend.texts) == 3, timeout=0.2)
    assert player.played == ["16000:1.0"]
    player.delay = 0.0
    _wait_idle(ctl)
    ctl.shutdown()
    assert len(player.played) == 3
    assert events == ["start", "end"]


def test_stop_while_synth_blocked_on_full_lookahead() -> None:
    player = FakePlayer(delay=5.0)
    ctl, backend, player, events = _controller(player)
    ctl.say("One. Two. Three. Four. Five. Six.")
    # one playing + _LOOKAHEAD queued + one blocked in put() => 4 synthesized
    _wait_for(lambda: len(backend.texts) == 4)
    t0 = time.time()
    ctl.stop()
    assert time.time() - t0 < 1.0
    assert ctl._idle.is_set() and not ctl.is_speaking
    assert events == ["start", "end"]
    time.sleep(0.1)
    ctl.shutdown()
    assert player.played == ["16000:1.0"]
    assert len(backend.texts) == 4  # synth bailed, did not render the rest


def test_stop_during_lookahead_drops_prerendered_request() -> None:
    player = FakePlayer(delay=5.0)
    ctl, backend, player, events = _controller(player)
    ctl.say("One.")
    ctl.say("Two.")
    _wait_for(lambda: "Two." in backend.texts)
    ctl.stop()
    assert player.played == ["16000:1.0"]
    player.delay = 0.0
    ctl.say("Three.")
    _wait_idle(ctl)
    ctl.shutdown()
    assert player.played == ["16000:1.0", "16000:1.0"]
    assert events == ["start", "end", "start", "end"]


def test_gap_between_sentences_but_not_before_first() -> None:
    ctl, _, player, _ = _controller()
    t0 = time.monotonic()
    ctl.say("A. B.")
    ctl.say("C.")
    _wait_idle(ctl)
    ctl.shutdown()
    assert len(player.times) == 3
    assert player.times[0] - t0 < 0.15  # first sentence: no gap
    assert 0.28 <= player.times[1] - player.times[0] < 0.6  # A -> B
    assert 0.28 <= player.times[2] - player.times[1] < 0.6  # B -> C (across requests)


def test_stop_does_not_wait_for_stale_render() -> None:
    backend = SlowBackend(delay=1.0)
    player = FakePlayer(delay=5.0)
    events: list[str] = []
    ctl = SpeechController(
        SpeechConfig(), player=player, backend=backend,
        on_speech_start=lambda: events.append("start"),
        on_speech_end=lambda: events.append("end"),
    )
    _offline(ctl)
    ctl.say("One. Two.")
    _wait_for(lambda: ctl.is_speaking and backend.rendering.is_set())
    t0 = time.monotonic()
    ctl.stop()
    assert time.monotonic() - t0 < 0.3  # did not wait for "Two." to finish rendering
    assert not ctl.is_speaking and events == ["start", "end"]
    assert ctl._idle.is_set()
    # Stale render finishes in the background and is discarded; new work flows.
    player.delay = 0.0
    backend.delay = 0.0
    ctl.say("Three.")
    _wait_idle(ctl, timeout=3.0)
    ctl.shutdown()
    assert player.played == ["16000:1.0", "16000:1.0"]
    assert backend.texts == ["One.", "Two.", "Three."]
    assert events == ["start", "end", "start", "end"]


def test_stop_during_gap_returns_promptly() -> None:
    ctl, _, player, events = _controller()
    ctl.say("A. B. C.")
    _wait_for(lambda: len(player.played) == 1)
    t0 = time.monotonic()
    ctl.stop()
    assert time.monotonic() - t0 < 0.2
    assert not ctl.is_speaking and events == ["start", "end"]
    time.sleep(0.4)
    ctl.shutdown()
    assert len(player.played) == 1


# ── interrupt / stop ──


def test_stop_flushes_queue_and_aborts_playback() -> None:
    player = FakePlayer(delay=5.0)
    ctl, backend, player, events = _controller(player)
    ctl.say("Long one.")
    ctl.say("Queued.")
    # wait until the first is actually playing
    deadline = time.time() + 2
    while not ctl.is_speaking and time.time() < deadline:
        time.sleep(0.005)
    assert ctl.is_speaking
    ctl.stop()
    assert player.aborts >= 1
    assert ctl.queue_length == 0
    assert not ctl.is_speaking
    assert events == ["start", "end"]
    ctl.shutdown()
    assert player.played == ["16000:1.0"]  # "Queued." may pre-render, never plays


def test_interrupt_replaces_current_speech() -> None:
    player = FakePlayer(delay=5.0)
    ctl, backend, player, events = _controller(player)
    ctl.say("Boring.")
    deadline = time.time() + 2
    while not ctl.is_speaking and time.time() < deadline:
        time.sleep(0.005)
    player.delay = 0.0
    ctl.say("Urgent!", interrupt=True)
    _wait_idle(ctl)
    ctl.shutdown()
    assert backend.texts == ["Boring.", "Urgent!"]
    assert events == ["start", "end", "start", "end"]


def test_set_voice_validates_and_updates() -> None:
    ctl, _, _, _ = _controller()
    with pytest.raises(Exception):
        ctl.set_voice("not-a-voice")
    ctl.set_voice("piper-en_US-amy-low")
    assert ctl.voice == "piper-en_US-amy-low"
    ctl.shutdown()


def test_apply_config_adopts_in_place() -> None:
    from dataclasses import replace

    ctl, _, player, _ = _controller()
    cfg_ref = ctl._config

    ctl.apply_config(replace(cfg_ref, speed=1.5, volume=40, voice="piper-en_US-amy-low"))
    assert ctl._config is cfg_ref and cfg_ref.speed == 1.5 and cfg_ref.volume == 40
    assert ctl.voice == "piper-en_US-amy-low"

    # device change: player re-targeted
    player.set_device = lambda d: setattr(player, "device", d)  # type: ignore[attr-defined]
    ctl.apply_config(replace(cfg_ref, device="USB Audio"))
    assert player.device == "USB Audio"  # type: ignore[attr-defined]
    ctl.shutdown()


# ── Per-request voices ──


def test_say_with_known_voice_uses_it() -> None:
    ctl, _, _, _ = _controller()
    located = _offline(ctl)
    assert ctl.say("Hi.", voice="AF_SARAH") == SayResult("kokoro-af_sarah")
    _wait_idle(ctl)
    ctl.shutdown()
    assert [v for v, _ in located] == ["kokoro-af_sarah"]


@pytest.mark.parametrize("voice", [None, "", "  "])
def test_say_without_voice_uses_default_silently(voice: str | None) -> None:
    ctl, _, _, _ = _controller()
    located = _offline(ctl)
    assert ctl.say("Hi.", voice=voice) == SayResult(DEFAULT_VOICE)
    _wait_idle(ctl)
    ctl.shutdown()
    assert [v for v, _ in located] == [DEFAULT_VOICE]


def test_say_with_unknown_voice_falls_back_to_default() -> None:
    ctl, backend, _, _ = _controller(voice="piper-en_US-amy-low")
    located = _offline(ctl)
    assert ctl.say("Hi.", voice="nonsense") == SayResult("piper-en_US-amy-low", "unknown voice 'nonsense'")
    _wait_idle(ctl)
    ctl.shutdown()
    assert [v for v, _ in located] == ["piper-en_US-amy-low"] and backend.texts == ["Hi."]


def test_unloadable_requested_voice_falls_back_without_critical_notify(monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[str] = []
    monkeypatch.setattr(speech, "notify", lambda title, *a, **k: notes.append(title))
    ctl, backend, _, _ = _controller()
    located = _offline(ctl, fail=("kokoro-am_adam",))
    ctl.say("Hi.", voice="am_adam")
    _wait_idle(ctl)
    ctl.shutdown()
    assert [v for v, _ in located] == ["kokoro-am_adam", DEFAULT_VOICE]
    assert backend.texts == ["Hi."] and notes == []


def test_unloadable_default_notifies(monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[str] = []
    monkeypatch.setattr(speech, "notify", lambda title, *a, **k: notes.append(title))
    ctl, backend, _, _ = _controller()
    _offline(ctl, fail=(DEFAULT_VOICE,))
    ctl.say("Hi.")
    _wait_idle(ctl)
    ctl.shutdown()
    assert backend.texts == [] and notes == ["Vocal — speech unavailable"]


def test_model_path_applies_only_to_default_voice() -> None:
    ctl, _, _, _ = _controller(model_path="/models/custom")
    located = _offline(ctl)
    ctl.say("One.")
    ctl.say("Two.", voice="piper-en_US-amy-low")
    _wait_idle(ctl)
    ctl.shutdown()
    assert located == [(DEFAULT_VOICE, "/models/custom"), ("piper-en_US-amy-low", None)]


def test_unknown_configured_voice_uses_builtin_default() -> None:
    ctl, _, _, _ = _controller(voice="piper-en-lessac-medium")  # pre-0.5 name
    assert ctl.voice == DEFAULT_VOICE
    located = _offline(ctl)
    ctl.say("Hi.")
    _wait_idle(ctl)
    ctl.shutdown()
    assert [v for v, _ in located] == [DEFAULT_VOICE]


def test_one_backend_per_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    def fake_resolve_backend(name: str) -> FakeBackend:
        created.append(name)
        return FakeBackend()

    monkeypatch.setattr(speech, "resolve_backend", fake_resolve_backend)
    ctl = SpeechController(SpeechConfig(), player=FakePlayer())
    _offline(ctl)
    for voice in ("af_sarah", "en_US-amy-low", "am_adam", "en_US-lessac-medium", "bf_emma"):
        ctl.say("Hi.", voice=voice)
    _wait_idle(ctl)
    ctl.shutdown()
    assert created == ["kokoro", "piper"]


# ── Backends: same model reloads nothing, only the speaker changes ──


def test_kokoro_speaker_switch_keeps_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from vocal.output.backends.kokoro import KokoroBackend

    built: list[str] = []
    calls: list[tuple[str, str]] = []

    class Kokoro:
        def __init__(self, onnx: str, voices: str) -> None:
            built.append(onnx)

        def create(self, text: str, voice: str, speed: float, lang: str):
            calls.append((voice, lang))
            return np.zeros(10, dtype=np.float32), 24000

    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=Kokoro))
    for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        (tmp_path / f).write_bytes(b"")
    backend = KokoroBackend()
    for style in ("af_sarah", "bf_emma", "ef_dora"):
        backend.load(tmp_path, style)
        backend.synthesize("Hi.")
    assert len(built) == 1
    assert calls == [("af_sarah", "en-us"), ("bf_emma", "en-gb"), ("ef_dora", "es")]


def test_piper_speaker_id_reaches_synthesis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import sys
    import types

    from vocal.output.backends.piper import PiperBackend

    loads: list[str] = []
    speakers: list[int | None] = []

    class Voice:
        def synthesize(self, text: str, syn_config):
            speakers.append(syn_config.speaker_id)
            return iter(())

    class PiperVoice:
        @staticmethod
        def load(onnx: str, config_path: str) -> Voice:
            loads.append(onnx)
            return Voice()

    class SynthesisConfig:
        def __init__(self, speaker_id=None, length_scale=None) -> None:
            self.speaker_id = speaker_id

    monkeypatch.setitem(sys.modules, "piper", types.SimpleNamespace(PiperVoice=PiperVoice, SynthesisConfig=SynthesisConfig))
    (tmp_path / "v.onnx").write_bytes(b"")
    (tmp_path / "v.onnx.json").write_text(json.dumps({"audio": {"sample_rate": 22050}}))
    backend = PiperBackend()
    for style in (None, "12", "3"):
        backend.load(tmp_path, style)
        list(backend.synthesize("Hi.").chunks)
    assert len(loads) == 1 and speakers == [None, 12, 3]
