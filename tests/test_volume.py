"""Tests for vocal.volume — backends parsing and Ducker behaviour."""

from __future__ import annotations

import time

import pytest

from vocal.volume import (
    AmixerBackend,
    Ducker,
    OsascriptBackend,
    PactlBackend,
    VolumeBackend,
    WpctlBackend,
)


class FakeBackend(VolumeBackend):
    """In-memory volume with a call log. ``level=None`` simulates an unreadable mixer."""

    tool = "fake"

    def __init__(self, level: int | None = 80, set_delay: float = 0.0) -> None:
        self.level = level
        self.sets: list[int] = []
        self.gets = 0
        self.set_delay = set_delay

    def get(self) -> int | None:
        self.gets += 1
        return self.level

    def set(self, percent: int) -> bool:
        if self.set_delay:
            time.sleep(self.set_delay)
        self.level = percent
        self.sets.append(percent)
        return True


def _wait_until(pred, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met in time")


# ── Parsers ─────────────────────────────────────────────────────────


def test_pactl_parse():
    out = "Volume: front-left: 42598 /  65% / -11.23 dB,   front-right: 42598 /  65% / -11.23 dB\n        balance 0.00\n"
    assert PactlBackend._parse(out) == 65


def test_wpctl_parse():
    assert WpctlBackend._parse("Volume: 0.65\n") == 65
    assert WpctlBackend._parse("Volume: 1.00 [MUTED]\n") == 100


def test_amixer_parse():
    out = (
        "Simple mixer control 'Master',0\n"
        "  Playback channels: Front Left - Front Right\n"
        "  Front Left: Playback 42598 [65%] [-11.23dB] [on]\n"
    )
    assert AmixerBackend._parse(out) == 65


def test_osascript_parse():
    assert OsascriptBackend._parse("65\n") == 65


def test_parse_garbage_returns_none():
    assert PactlBackend._parse("nope") is None
    assert WpctlBackend._parse("nope") is None
    assert AmixerBackend._parse("nope") is None


# ── Ducker ──────────────────────────────────────────────────────────


def test_duck_and_restore_roundtrip():
    b = FakeBackend(level=80)
    d = Ducker(50, b, ramp_ms=30, steps=3)

    d.duck()
    assert b.level == 40
    assert d.is_ducked

    d.restore()
    _wait_until(lambda: not d.is_ducked)
    assert b.level == 80
    # Ramp went up monotonically and ended exactly at the saved level.
    ramp = b.sets[1:]
    assert ramp == sorted(ramp)
    assert ramp[-1] == 80


def test_duck_is_idempotent():
    b = FakeBackend(level=80)
    d = Ducker(50, b, ramp_ms=30, steps=3)
    d.duck()
    d.duck()
    assert b.sets == [40]
    assert b.gets == 1


def test_reduck_during_ramp_uses_original_saved_level():
    # Slow set() so the ramp is definitely still in flight when we re-duck.
    b = FakeBackend(level=80, set_delay=0.02)
    d = Ducker(50, b, ramp_ms=200, steps=4)

    d.duck()
    assert b.level == 40
    d.restore()
    _wait_until(lambda: len(b.sets) >= 2)  # first ramp step landed

    d.duck()  # cancels ramp, re-ducks
    assert d.is_ducked
    assert b.level == 40
    assert b.gets == 1  # never re-read the mixer; saved level preserved

    d.restore()
    _wait_until(lambda: not d.is_ducked)
    assert b.level == 80


def test_restore_without_duck_is_noop():
    b = FakeBackend(level=80)
    d = Ducker(50, b)
    d.restore()
    assert b.sets == []
    assert not d.is_ducked


def test_double_restore_starts_one_ramp():
    b = FakeBackend(level=80, set_delay=0.02)
    d = Ducker(50, b, ramp_ms=100, steps=4)
    d.duck()
    d.restore()
    d.restore()
    _wait_until(lambda: not d.is_ducked)
    # 1 duck set + 4 ramp steps, not 8
    assert len(b.sets) == 5


def test_unreadable_volume_disables_ducking():
    b = FakeBackend(level=None)
    d = Ducker(50, b)
    d.duck()
    assert b.sets == []
    assert not d.is_ducked
    d.restore()
    assert b.sets == []


@pytest.mark.parametrize("amount,expected", [(150, 0), (-5, 80), (100, 0), (0, 80)])
def test_amount_clamped(amount, expected):
    b = FakeBackend(level=80)
    d = Ducker(amount, b)
    d.duck()
    assert b.level == expected


def test_close_snaps_back_immediately():
    b = FakeBackend(level=80)
    d = Ducker(50, b, ramp_ms=500, steps=10)
    d.duck()
    d.close()
    assert b.level == 80
    assert not d.is_ducked
    assert b.sets == [40, 80]


def test_close_during_ramp_cancels_and_snaps():
    b = FakeBackend(level=80, set_delay=0.02)
    d = Ducker(50, b, ramp_ms=400, steps=8)
    d.duck()
    d.restore()
    _wait_until(lambda: len(b.sets) >= 2)
    d.close()
    assert b.level == 80
    assert not d.is_ducked
    assert len(b.sets) < 9  # ramp did not run to completion


# ── StreamDucker ──

from vocal.volume import StreamDucker  # noqa: E402

_PACTL_LIST = """Sink Input #906
\tDriver: protocol-native.c
\tVolume: mono: 65536 / 100% / 0.00 dB
\t        balance 0.00
\tProperties:
\t\tapplication.name = "speech-dispatcher-dummy"
\t\tapplication.process.id = "193535"
Sink Input #3027
\tVolume: front-left: 39321 /  60% / -13.31 dB,   front-right: 39321 /  60% / -13.31 dB
\tProperties:
\t\tapplication.name = "Firefox"
\t\tapplication.process.id = "589263"
Sink Input #4612
\tVolume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB
\tProperties:
\t\tapplication.name = "python3"
\t\tapplication.process.id = "4242"
Sink Input #5000
\tVolume: front-left: 65536 / 100% / 0.00 dB
\tProperties:
\t\tapplication.name = "no-pid"
"""


class FakeRunner:
    def __init__(self, listing: str = _PACTL_LIST, fail_set: set[int] = frozenset()) -> None:
        self.listing = listing
        self.fail_set = fail_set
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> str | None:
        self.calls.append(cmd)
        if cmd[1] == "list":
            return self.listing
        if cmd[1] == "set-sink-input-volume":
            return None if int(cmd[2]) in self.fail_set else ""
        return None


def test_stream_ducker_parse():
    parsed = StreamDucker._parse(_PACTL_LIST)
    assert parsed == [(906, 193535, 100), (3027, 589263, 60), (4612, 4242, 100), (5000, None, 100)]


def test_stream_ducker_parse_empty():
    assert StreamDucker._parse("") == []


def test_stream_ducker_excludes_own_pid_and_restores():
    run = FakeRunner()
    d = StreamDucker(50, exclude_pid=4242, run=run)
    d.duck()
    sets = [c for c in run.calls if c[1] == "set-sink-input-volume"]
    assert sorted((c[2], c[3]) for c in sets) == [("3027", "30%"), ("5000", "50%"), ("906", "50%")]
    assert d.is_ducked
    run.calls.clear()
    d.restore()
    sets = [c for c in run.calls if c[1] == "set-sink-input-volume"]
    assert sorted((c[2], c[3]) for c in sets) == [("3027", "60%"), ("5000", "100%"), ("906", "100%")]
    assert not d.is_ducked


def test_stream_ducker_duck_idempotent_and_skips_failed_sets():
    run = FakeRunner(fail_set={906})
    d = StreamDucker(50, exclude_pid=4242, run=run)
    d.duck()
    d.duck()  # second duck must not re-duck already-ducked (would compound)
    sets = [c for c in run.calls if c[1] == "set-sink-input-volume"]
    assert len(sets) == 3 + 1  # 3 first pass (906 fails, not saved) + 906 retried once
    run.calls.clear()
    d.restore()
    restored = {c[2] for c in run.calls if c[1] == "set-sink-input-volume"}
    assert restored == {"3027", "5000"}  # 906 never succeeded, so never restored


def test_stream_ducker_handles_pactl_failure():
    run = FakeRunner()
    run.listing = None  # type: ignore[assignment]
    d = StreamDucker(50, exclude_pid=1, run=lambda cmd: None)
    d.duck()
    assert not d.is_ducked
    d.restore()
