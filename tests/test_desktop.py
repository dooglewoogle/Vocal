"""Desktop-entry installer writes absolute Exec paths into XDG dirs."""

from __future__ import annotations

from pathlib import Path

import pytest

from vocal import desktop


@pytest.fixture
def xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(desktop, "vocal_command", lambda: "/opt/vocal/.venv/bin/vocal")
    monkeypatch.setattr(desktop, "supported", lambda: True)
    return tmp_path


def test_install_writes_menu_entry_icon_and_autostart(xdg: Path) -> None:
    written = desktop.install(autostart=True)
    app, auto, icon = desktop.applications_file(), desktop.autostart_file(), desktop.icon_file()
    assert set(written) == {app, auto, icon}
    assert app.read_text().splitlines()[0] == "[Desktop Entry]"
    assert "Exec=/opt/vocal/.venv/bin/vocal\n" in app.read_text()
    assert "Icon=vocal\n" in app.read_text()
    assert "X-GNOME-Autostart-enabled=true" in auto.read_text()
    assert "X-GNOME-Autostart-enabled" not in app.read_text()
    assert icon.read_bytes()[:4] == b"\x89PNG"
    assert desktop.is_autostart_enabled()


def test_autostart_toggle_keeps_menu_entry(xdg: Path) -> None:
    desktop.set_autostart(True)
    desktop.set_autostart(False)
    assert not desktop.is_autostart_enabled()
    assert desktop.applications_file().exists()
    desktop.install(autostart=None)  # leaves autostart alone
    assert not desktop.is_autostart_enabled()


def test_uninstall_removes_everything(xdg: Path) -> None:
    desktop.install(autostart=True)
    removed = desktop.uninstall()
    assert len(removed) == 3
    assert not any(p.exists() for p in (desktop.applications_file(), desktop.autostart_file(), desktop.icon_file()))
    assert desktop.uninstall() == []


def test_vocal_command_prefers_sibling_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "bin" / "vocal"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setattr(desktop.sys, "executable", str(tmp_path / "bin" / "python"))
    assert desktop.vocal_command() == str(exe)
    exe.unlink()
    monkeypatch.setattr(desktop.shutil, "which", lambda name: None)
    assert desktop.vocal_command().endswith(" -m vocal")
