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
    assert a.command == "say" and a.interrupt is True and a.voice == "system"
    assert a.text == ["hello", "there"]
    assert parse_args(["say"]).text == []
    assert parse_args(["say", "-"]).text == ["-"]


def test_root_config_flag_before_subcommand():
    a = parse_args(["--config", "/tmp/x.toml", "say", "hi"])
    assert a.config == "/tmp/x.toml" and a.command == "say"


def test_models_parsing():
    assert parse_args(["models"]).models_command == "list"
    assert parse_args(["models", "list"]).models_command == "list"
    a = parse_args(["models", "download", "piper-en-amy-low"])
    assert a.models_command == "download" and a.name == "piper-en-amy-low"
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
