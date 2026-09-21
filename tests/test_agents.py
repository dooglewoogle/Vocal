"""install-agents merges hooks into agent config JSON and manages the instruction block."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vocal import agents


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(agents.shutil, "which", lambda _b: None)
    monkeypatch.setattr(agents.sys, "executable", "/opt/vocal/venv/bin/python")
    return tmp_path


def _claude_settings(home: Path, stale: bool = False) -> Path:
    p = home / ".claude" / "settings.json"
    p.parent.mkdir()
    data = {
        "permissions": {"allow": ["Bash(ls:*)"]},
        "hooks": {
            "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "rtk hook claude"}]}],
        },
    }
    if stale:
        data["hooks"]["MessageDisplay"] = [
            {"hooks": [{"type": "command", "command": "python3 ~/.claude/hooks/say-hook.py", "timeout": 5}]},
        ]
    p.write_text(json.dumps(data))
    return p


def test_detected_by_dir_or_binary(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (home / ".codex").mkdir()
    assert agents.detected() == ["codex"]
    monkeypatch.setattr(agents.shutil, "which", lambda b: "/usr/bin/gemini" if b == "gemini" else None)
    assert agents.detected() == ["codex", "gemini"]


def test_install_merges_and_replaces_stale_hook(home: Path) -> None:
    settings = _claude_settings(home, stale=True)
    written = agents.install(["claude"])
    assert set(written) == {settings, home / ".claude" / "CLAUDE.md"}

    data = json.loads(settings.read_text())
    assert data["permissions"] == {"allow": ["Bash(ls:*)"]}
    assert data["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "rtk hook claude"
    md = data["hooks"]["MessageDisplay"]
    assert len(md) == 1 and len(md[0]["hooks"]) == 1
    assert md[0]["hooks"][0] == {
        "type": "command", "command": "/opt/vocal/venv/bin/python -m vocal.hooks.say_hook", "timeout": 5,
    }
    assert agents.is_installed("claude")


def test_install_is_idempotent(home: Path) -> None:
    (home / ".codex").mkdir()
    (home / ".codex" / "AGENTS.md").write_text("# Mine\n\nKeep this.\n")
    first = agents.install(["codex"])
    assert len(first) == 2
    snapshot = {p: p.read_bytes() for p in first}
    assert agents.install(["codex"]) == []
    assert {p: p.read_bytes() for p in first} == snapshot

    text = (home / ".codex" / "AGENTS.md").read_text()
    assert text.startswith("# Mine\n\nKeep this.\n\n<!-- vocal:begin -->")
    assert text.count("<!-- vocal:begin -->") == 1
    assert text.endswith("<!-- vocal:end -->\n")


def test_uninstall_removes_only_ours(home: Path) -> None:
    settings = _claude_settings(home)
    agents.install(["claude"])
    (home / ".claude" / "CLAUDE.md").write_text(
        "Before.\n\n" + agents.INSTRUCTIONS + "\nAfter.\n")
    changed = agents.uninstall(["claude"])
    assert set(changed) == {settings, home / ".claude" / "CLAUDE.md"}

    data = json.loads(settings.read_text())
    assert "MessageDisplay" not in data["hooks"]
    assert data["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "rtk hook claude"
    assert (home / ".claude" / "CLAUDE.md").read_text() == "Before.\n\nAfter.\n"
    assert not agents.is_installed("claude")
    assert agents.uninstall(["claude"]) == []


def test_gemini_entry_uses_milliseconds_and_name(home: Path) -> None:
    agents.install(["gemini"])
    data = json.loads((home / ".gemini" / "settings.json").read_text())
    hook = data["hooks"]["AfterAgent"][0]["hooks"][0]
    assert hook["timeout"] == 5000 and hook["name"] == "vocal-say"
    assert (home / ".gemini" / "GEMINI.md").read_text() == agents.INSTRUCTIONS


def test_malformed_json_aborts_without_writing(home: Path) -> None:
    (home / ".claude").mkdir()
    bad = home / ".claude" / "settings.json"
    bad.write_text("{not json")
    (home / ".codex").mkdir()
    with pytest.raises(agents.AgentConfigError, match="settings.json"):
        agents.install(["codex", "claude"])
    assert bad.read_text() == "{not json"
    assert not (home / ".codex" / "hooks.json").exists()
    assert not (home / ".codex" / "AGENTS.md").exists()
