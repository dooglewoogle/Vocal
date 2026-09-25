"""Hotkey backend selection degrades to a no-op listener when nothing is installed."""

from __future__ import annotations

import sys
import threading

import pytest

from vocal.config import HotkeyConfig
from vocal.input import hotkey


def _block(monkeypatch: pytest.MonkeyPatch, *modules: str) -> None:
    for m in modules:
        monkeypatch.setitem(sys.modules, m, None)  # makes `import m` raise ImportError


def test_no_backends_gives_null_listener(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    _block(monkeypatch, "evdev", "pynput")
    assert hotkey.available_backends() == []
    lst = hotkey.create_listener(HotkeyConfig(), on_start=lambda: None, on_stop=lambda: None)
    assert isinstance(lst, hotkey.NullHotkeyListener)
    assert "No hotkey backend installed" in caplog.text
    t = threading.Thread(target=lst.run, daemon=True)
    t.start()
    lst.stop()
    t.join(1.0)
    assert not t.is_alive()


def test_requested_backend_missing_falls_back_with_warning(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    _block(monkeypatch, "evdev", "pynput")
    lst = hotkey.create_listener(HotkeyConfig(backend="evdev"), on_start=lambda: None, on_stop=lambda: None)
    assert isinstance(lst, hotkey.NullHotkeyListener)
    assert "requested but not installed" in caplog.text


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown hotkey backend"):
        hotkey.create_listener(HotkeyConfig(backend="telepathy"), on_start=lambda: None, on_stop=lambda: None)


def test_auto_prefers_evdev_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("evdev")
    monkeypatch.setattr(hotkey.sys, "platform", "linux")
    assert hotkey.available_backends()[0] == "evdev"


def _fake_pynput(monkeypatch: pytest.MonkeyPatch, listener_cls: type) -> None:
    import types

    keyboard = types.ModuleType("pynput.keyboard")
    keyboard.Key = types.SimpleNamespace(pause="pause")  # type: ignore[attr-defined]
    keyboard.Listener = listener_cls  # type: ignore[attr-defined]
    pynput = types.ModuleType("pynput")
    pynput.keyboard = keyboard  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynput", pynput)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)


class _RefusedListener:
    """Mimics pynput when the OS refuses the event tap: the thread ends at once."""

    def __init__(self, on_press, on_release) -> None:
        pass

    def start(self) -> None:
        pass

    def join(self) -> None:
        pass

    def stop(self) -> None:
        pass


def test_pynput_refused_tap_keeps_blocking_until_stopped(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    _fake_pynput(monkeypatch, _RefusedListener)
    lst = hotkey.PynputHotkeyListener(HotkeyConfig(key="pause"), on_start=lambda: None, on_stop=lambda: None)
    t = threading.Thread(target=lst.run, daemon=True)
    t.start()
    t.join(0.3)
    assert t.is_alive(), "run() returned, which would shut the dictation engine down"
    assert "hotkey is disabled" in caplog.text
    lst.stop()
    t.join(1.0)
    assert not t.is_alive()


def test_pynput_stop_returns_without_error(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    _fake_pynput(monkeypatch, _RefusedListener)
    lst = hotkey.PynputHotkeyListener(HotkeyConfig(key="pause"), on_start=lambda: None, on_stop=lambda: None)
    lst.stop()
    lst.run()
    assert "hotkey is disabled" not in caplog.text
