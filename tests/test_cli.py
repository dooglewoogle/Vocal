"""Tests for CLI engine-mode resolution."""

from __future__ import annotations

import argparse

import pytest

from vocal.cli import _resolve_initial_mode


def _args(**kw) -> argparse.Namespace:
    base = dict(hotkey=False, live=False, duck=None, mode=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({}, None),
        ({"hotkey": True}, "hotkey"),
        ({"live": True}, "live"),
        ({"duck": True}, "hotkey"),
        ({"duck": True, "live": True}, "live"),
        ({"mode": "ptt"}, "hotkey"),  # deprecated flag still implies hotkey mode
        ({"hotkey": True, "duck": True}, "hotkey"),
    ],
)
def test_resolve_initial_mode(kw, expected):
    assert _resolve_initial_mode(_args(**kw)) == expected


# ── argument parsing: subcommands vs. legacy flat flags ──

from vocal.cli import parse_args  # noqa: E402


def test_no_subcommand_keeps_legacy_flags():
    a = parse_args(["--hotkey", "--duck", "--model", "tiny.en", "--no-server"])
    assert a.command is None
    assert a.hotkey is True and a.duck is True and a.model == "tiny.en" and a.no_server is True


def test_no_args_is_daemon():
    a = parse_args([])
    assert a.command is None and a.live is False and a.hotkey is False


def test_say_parsing():
    a = parse_args(["say", "-i", "--voice", "system", "hello", "there"])
    assert a.command == "say" and a.interrupt is True and a.say_voice == "system"
    assert a.text == ["hello", "there"]
    assert parse_args(["say"]).text == []
    assert parse_args(["say", "-"]).text == ["-"]


def test_root_config_flag_before_subcommand():
    a = parse_args(["--config", "/tmp/x.toml", "say", "hi"])
    assert a.config == "/tmp/x.toml" and a.command == "say"


def test_models_parsing():
    assert parse_args(["models"]).models_command == "list"
    assert parse_args(["models", "list"]).models_command == "list"
    a = parse_args(["models", "download", "piper-en_US-amy-low"])
    assert a.models_command == "download" and a.name == "piper-en_US-amy-low"
    a = parse_args(["models", "remove", "x"])
    assert a.models_command == "remove" and a.name == "x"


def test_stop_status_parsing():
    assert parse_args(["stop"]).command == "stop"
    assert parse_args(["status"]).command == "status"


def test_headless_flag() -> None:
    from vocal.cli import parse_args

    assert parse_args(["--headless"]).headless is True
    assert parse_args([]).headless is False


def test_cli_overrides_report_paths() -> None:
    from vocal.cli import _apply_cli_overrides, parse_args
    from vocal.config import VocalConfig

    cfg = VocalConfig()
    args = parse_args(["--duck", "--key", "END", "--phrasebook", "--model", "tiny.en"])
    touched = _apply_cli_overrides(cfg, args)
    assert cfg.input.engine == "hotkey" and cfg.input.hotkey.duck is True
    assert cfg.input.hotkey.key == "END" and cfg.input.phrasebook.seed is True
    assert cfg.input.model.size == "tiny.en"
    assert touched == {"input.engine", "input.hotkey.duck", "input.hotkey.key",
                       "input.phrasebook.seed", "input.model.size"}

    # deprecated --mode: accepted, implies hotkey, writes nothing else
    cfg3 = VocalConfig()
    assert _apply_cli_overrides(cfg3, parse_args(["--mode", "ptt"])) == {"input.engine"}

    cfg2 = VocalConfig()
    cfg2.input.engine = "hotkey"
    assert _apply_cli_overrides(cfg2, parse_args([])) == set()
    assert cfg2.input.engine == "hotkey"  # file value survives when no flag given


def test_install_desktop_parsing() -> None:
    from vocal.cli import parse_args

    a = parse_args(["install-desktop"])
    assert a.command == "install-desktop" and a.autostart is None and a.uninstall is False
    assert parse_args(["install-desktop", "--autostart"]).autostart is True
    assert parse_args(["install-desktop", "--no-autostart"]).autostart is False
    assert parse_args(["install-desktop", "--uninstall"]).uninstall is True


# ── say / models commands ──


def test_say_reports_daemon_fallback(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from vocal import cli
    from vocal.output import client

    sent = []

    def fake_say(text, interrupt=False, voice=None):
        sent.append(voice)
        return {"ok": True, "voice": "kokoro-bf_emma", "fallback": f"unknown voice {voice!r}"}

    monkeypatch.setattr(client, "say", fake_say)
    assert cli._cmd_say(parse_args(["say", "--voice", "nope", "hi"])) == 0
    assert sent == ["nope"]
    assert capsys.readouterr().err.strip() == "Unknown voice 'nope'; using kokoro-bf_emma"


def test_models_list_filters_and_marks_default(tmp_path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture) -> None:
    from vocal import cli

    monkeypatch.setenv("VOCAL_MODELS_DIR", str(tmp_path / "models"))
    cfg = tmp_path / "config.toml"
    cfg.write_text('[output.speech]\nvoice = "piper-en_GB-alan-medium"\n')
    assert cli._cmd_models(parse_args(["--config", str(cfg), "models", "list", "EN_GB-ALAN"])) == 0
    out = capsys.readouterr().out
    assert "piper · English (Great Britain)" in out
    assert "[ ]*piper-en_GB-alan-medium" in out and "[ ] piper-en_GB-alan-low" in out
    assert "kokoro-" not in out  # only the matches are listed
    assert cli._cmd_models(parse_args(["--config", str(cfg), "models", "list", "zzz-no-such-voice"])) == 1


def test_permissions_off_macos(monkeypatch, capsys):
    from vocal import cli

    monkeypatch.setattr(cli.sys, "platform", "linux")
    assert cli._cmd_permissions(parse_args(["permissions"])) == 0
    assert "only needed on macOS" in capsys.readouterr().out
