"""VocalApp: config application routing, engine rebuild guards, speech coupling."""

from __future__ import annotations

import copy
import threading
import time
from pathlib import Path

import pytest

from vocal.app import VocalApp, changed_paths
from vocal.config import VocalConfig
from vocal.state import DictationState


class FakeEngine:
    instances: list["FakeEngine"] = []

    def __init__(self, mode: str, **kw) -> None:
        self.mode = mode
        self.kw = kw
        self.started = False
        self.shut = False
        self.suppressed = 0
        self.released = 0
        self.phrasebook_calls: list[tuple] = []
        self._state = DictationState.LISTENING
        FakeEngine.instances.append(self)

    @property
    def current_state(self) -> DictationState:
        return self._state

    def start(self) -> None:
        self.started = True

    def shutdown(self) -> None:
        self.shut = True
        # A real engine fires on_shutdown_requested when run() returns.
        self.kw["on_shutdown_requested"]()

    def suppress_input(self) -> None:
        self.suppressed += 1

    def release_input(self) -> None:
        self.released += 1

    def toggle_pause(self) -> None:
        self._state = DictationState.SLEEPING if self._state == DictationState.LISTENING else DictationState.LISTENING
        self.kw["on_state_change"](self._state)

    def set_phrasebook(self, pb, seed, replace) -> None:
        self.phrasebook_calls.append((pb, seed, replace))

    # test helpers
    def emit_transcript(self, text: str) -> None:
        self.kw["on_transcript"](text)


class FakeSpeech:
    def __init__(self, cfg, on_speech_start=None, on_speech_end=None) -> None:
        self.cfg = cfg
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.applied: list = []
        self.stopped = 0
        self.said: list = []
        self._speaking = False
        self.shut = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def apply_config(self, new) -> None:
        self.applied.append(copy.deepcopy(new))

    def say(self, text, interrupt=False, voice=None) -> None:
        self.said.append((text, interrupt, voice))

    def stop(self) -> None:
        self.stopped += 1

    def shutdown(self) -> None:
        self.shut = True

    # test helpers
    def begin(self) -> None:
        self._speaking = True
        self.on_speech_start()

    def end(self) -> None:
        self._speaking = False
        self.on_speech_end()


class FakeServer:
    instances: list["FakeServer"] = []

    def __init__(self, controller, host, port) -> None:
        self.host, self.port = host, port
        self.running = False
        FakeServer.instances.append(self)

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


class FakeDucker:
    def __init__(self, amount: int) -> None:
        self._amount = amount
        self.ducks = 0
        self.restores = 0

    def duck(self) -> None:
        self.ducks += 1

    def restore(self) -> None:
        self.restores += 1


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    FakeEngine.instances.clear()
    FakeServer.instances.clear()
    monkeypatch.setattr("vocal.app.CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr("vocal.config.CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr("vocal.app.load_phrasebook", lambda: "PB")
    yield


def _app(**cfg_edits) -> VocalApp:
    cfg = VocalConfig()
    for path, value in cfg_edits.items():
        obj = cfg
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)
    import vocal.app as app_mod

    app = VocalApp(
        cfg,
        config_path=app_mod.CONFIG_PATH,
        engine_factory=FakeEngine,
        speech_factory=FakeSpeech,
        server_factory=FakeServer,
        ducker_factory=FakeDucker,
        ducker_available=lambda: True,
    )
    quits: list[bool] = []
    app.start(quit_loop=lambda: quits.append(True))
    app._quits = quits  # type: ignore[attr-defined]
    return app


def _wait(pred, timeout: float = 1.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met")


# ── changed_paths ──


def test_changed_paths_nested() -> None:
    a = {"x": 1, "in": {"m": {"size": "a"}, "k": 2}}
    b = {"x": 1, "in": {"m": {"size": "b"}, "k": 2, "new": 3}}
    assert changed_paths(a, b) == {"in.m.size", "in.new"}


# ── startup ──


def test_start_builds_engine_then_server_then_starts_engine() -> None:
    app = _app()
    eng = FakeEngine.instances[0]
    assert eng.mode == "live" and eng.started
    assert FakeServer.instances[0].running
    assert app._ducker is not None  # duck=True by default
    assert "on_before_record" in eng.kw


def test_hotkey_mode_and_phrasebook_wiring() -> None:
    app = _app(**{"input.engine": "hotkey", "input.phrasebook.replace": True})
    eng = FakeEngine.instances[0]
    assert eng.mode == "hotkey"
    assert eng.kw["phrasebook"] == "PB" and eng.kw["phrasebook_replace"] is True
    assert app.phrasebook == "PB"


# ── apply_config routing ──


def test_apply_speech_only_change_does_not_rebuild(tmp_path: Path) -> None:
    app = _app()
    new = copy.deepcopy(app.config)
    new.output.speech.speed = 1.7
    notes = app.apply_config(new)
    assert len(FakeEngine.instances) == 1
    assert app.speech.applied and app.speech.applied[-1].speed == 1.7
    assert app.config.output.speech.speed == 1.7
    assert (tmp_path / "config.toml").exists()
    assert any("Speech" in n for n in notes)


def test_apply_input_change_rebuilds_engine_once() -> None:
    app = _app()
    old = FakeEngine.instances[0]
    new = copy.deepcopy(app.config)
    new.input.model.size = "tiny.en"
    new.input.hotkey.key = "F9"
    app.apply_config(new)
    assert old.shut
    assert len(FakeEngine.instances) == 2 and FakeEngine.instances[1].started
    assert app.config.input.model.size == "tiny.en"
    # The old engine's exit callback must NOT have shut the app down.
    assert app._quits == [] and not app._shutdown_started.is_set()


def test_apply_phrasebook_flags_only_hot_swaps() -> None:
    app = _app()
    eng = FakeEngine.instances[0]
    new = copy.deepcopy(app.config)
    new.input.phrasebook.replace = True
    app.apply_config(new)
    assert len(FakeEngine.instances) == 1
    assert eng.phrasebook_calls == [("PB", False, True)]


def test_apply_server_change_restarts_server() -> None:
    app = _app()
    first = FakeServer.instances[0]
    new = copy.deepcopy(app.config)
    new.output.server.port = 5555
    app.apply_config(new)
    assert not first.running
    assert FakeServer.instances[1].running and FakeServer.instances[1].port == 5555

    new = copy.deepcopy(app.config)
    new.output.server.enabled = False
    app.apply_config(new)
    assert not FakeServer.instances[1].running and app._server is None


def test_apply_duck_toggle_syncs_ducker() -> None:
    app = _app()
    d0 = app._ducker
    new = copy.deepcopy(app.config)
    new.output.speech.duck_amount = 80
    app.apply_config(new)
    assert app._ducker is not d0 and d0.restores == 1 and app._ducker._amount == 80
    new = copy.deepcopy(app.config)
    new.output.speech.duck = False
    app.apply_config(new)
    assert app._ducker is None


def test_set_voice_persists(tmp_path: Path) -> None:
    app = _app()
    app.set_voice("piper-en-amy-low")
    assert app.config.output.speech.voice == "piper-en-amy-low"
    assert 'voice = "piper-en-amy-low"' in (tmp_path / "config.toml").read_text()


# ── generation guard ──


def test_superseded_engine_events_are_dropped() -> None:
    app = _app()
    seen: list[str] = []
    app.on_transcript.connect(seen.append)
    old = FakeEngine.instances[0]
    new = copy.deepcopy(app.config)
    new.input.engine = "hotkey"
    app.apply_config(new)
    old.emit_transcript("stale")
    FakeEngine.instances[1].emit_transcript("fresh")
    assert seen == ["fresh"]


def test_engine_exit_outside_rebuild_requests_shutdown() -> None:
    app = _app()
    FakeEngine.instances[0].kw["on_shutdown_requested"]()
    assert app._quits == [True]


# ── speech ↔ input coupling ──


def test_speech_suppresses_input_and_releases_after_tail() -> None:
    app = _app(**{"output.speech.pause_input_tail_ms": 20})
    eng = FakeEngine.instances[0]
    speaking: list[bool] = []
    app.on_speaking.connect(speaking.append)

    app.speech.begin()
    assert eng.suppressed == 1 and speaking == [True]
    _wait(lambda: app._ducker.ducks == 1)
    app.speech.end()
    assert speaking == [True, False]
    _wait(lambda: eng.released == 1)
    _wait(lambda: app._ducker.restores == 1)


def test_rebuild_during_speech_suppresses_new_engine_and_cancels_timer() -> None:
    app = _app(**{"output.speech.pause_input_tail_ms": 10_000})
    app.speech.begin()
    app.speech.end()  # timer pending for 10 s
    assert app._release_timer is not None
    app.speech._speaking = True
    new = copy.deepcopy(app.config)
    new.input.model.size = "base.en"
    app.apply_config(new)
    assert app._release_timer is None
    assert FakeEngine.instances[1].suppressed == 1


def test_toggle_pause_emits_state() -> None:
    app = _app()
    states: list[DictationState] = []
    app.on_state.connect(states.append)
    app.toggle_pause()
    assert states == [DictationState.SLEEPING] and app.state == DictationState.SLEEPING


# ── shutdown ──


def test_shutdown_order_and_idempotence() -> None:
    app = _app()
    app.shutdown()
    assert not FakeServer.instances[0].running
    assert app.speech.shut and FakeEngine.instances[0].shut
    assert app._quits == []  # engine exit during shutdown must not re-enter quit_loop
    app.shutdown()  # second call is harmless


def test_request_shutdown_once() -> None:
    app = _app()
    t = threading.Thread(target=app.request_shutdown)
    t.start()
    t.join()
    app.request_shutdown()
    assert app._quits == [True]
