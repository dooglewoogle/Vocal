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
from vocal.output.speech import SpeechController, split_sentences


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
    # Pretend the configured voice is what the fake backend has loaded.
    ctl._backend_voice = ctl.voice
    return ctl, backend, player, events


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
    ctl._backend_voice = ctl.voice
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
    ctl.set_voice("piper-en-amy-low")
    assert ctl.voice == "piper-en-amy-low"
    ctl.shutdown()


def test_apply_config_invalidates_loaded_voice_only_when_needed() -> None:
    from dataclasses import replace

    ctl, _, player, _ = _controller()
    cfg_ref = ctl._config
    assert ctl._backend_voice == ctl.voice

    # speed/volume only: no reload, same config object mutated
    ctl.apply_config(replace(cfg_ref, speed=1.5, volume=40))
    assert ctl._backend_voice == ctl.voice
    assert ctl._config is cfg_ref and cfg_ref.speed == 1.5 and cfg_ref.volume == 40

    # voice change: loaded voice invalidated
    ctl.apply_config(replace(cfg_ref, voice="piper-en-amy-low"))
    assert ctl._backend_voice is None and ctl.voice == "piper-en-amy-low"

    # device change: player re-targeted
    player.set_device = lambda d: setattr(player, "device", d)  # type: ignore[attr-defined]
    ctl.apply_config(replace(cfg_ref, device="USB Audio"))
    assert player.device == "USB Audio"  # type: ignore[attr-defined]
    ctl.shutdown()
