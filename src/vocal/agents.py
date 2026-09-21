"""Hook Vocal's speech into AI coding agents (Claude Code, Codex CLI, Gemini CLI).

``vocal install-agents`` does two things per agent, both in the user's global
agent config, never per project:

* registers ``python -m vocal.hooks.say_hook`` as a hook on the event that
  carries the assistant's text, merged into the agent's existing hooks JSON;
* appends a marked instruction block to the agent's global instructions file
  telling the model to wrap a one-sentence spoken summary in ``<say>`` tags.

Both are idempotent and reversible (``--uninstall``). The hook command points
at *this* interpreter by absolute path, like ``desktop.vocal_command()``.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

HOOK_MODULE = "vocal.hooks.say_hook"
# Commands recognised as ours (the second is the hand-rolled predecessor).
OURS = (HOOK_MODULE, "say-hook.py")
BEGIN, END = "<!-- vocal:begin -->", "<!-- vocal:end -->"
BLOCK_RE = re.compile(rf"{re.escape(BEGIN)}.*?{re.escape(END)}\n?", re.DOTALL)

INSTRUCTIONS = f"""{BEGIN}
## Vocal (text-to-speech)
A Vocal daemon on this machine reads aloud anything you put inside <say>...</say> tags
(a hook posts it to the daemon; tags inside code blocks are ignored).
- In your opening line and again in your final reply, give a one-sentence conversational
  answer with the single most relevant piece of information, wrapped in <say> tags, with
  as much snark or sarcasm as you can, then continue with the rest of your reply as usual.
- Keep it to plain prose: no code, file paths, or markdown inside the tags.
- Nothing is spoken when the daemon is not running, so never wait on it.
{END}
"""


@dataclass(frozen=True)
class AgentSpec:
    name: str
    title: str
    home: str               # directory under $HOME
    binary: str             # executable that marks the agent as installed
    hooks_file: str         # JSON file under ``home`` holding a top-level "hooks" table
    event: str              # hook event that carries the assistant text
    timeout: int            # in the agent's own unit (Claude/Codex: s, Gemini: ms)
    instructions_file: str  # global instructions markdown under ``home``


AGENTS: dict[str, AgentSpec] = {
    "claude": AgentSpec("claude", "Claude Code", ".claude", "claude", "settings.json",
                        "MessageDisplay", 5, "CLAUDE.md"),
    "codex": AgentSpec("codex", "Codex CLI", ".codex", "codex", "hooks.json",
                       "Stop", 5, "AGENTS.md"),
    "gemini": AgentSpec("gemini", "Gemini CLI", ".gemini", "gemini", "settings.json",
                        "AfterAgent", 5000, "GEMINI.md"),
}


class AgentConfigError(RuntimeError):
    """An agent's config file could not be parsed; nothing was written."""


def home_dir(spec: AgentSpec) -> Path:
    return Path.home() / spec.home


def hooks_path(spec: AgentSpec) -> Path:
    return home_dir(spec) / spec.hooks_file


def instructions_path(spec: AgentSpec) -> Path:
    return home_dir(spec) / spec.instructions_file


def detected() -> list[str]:
    """Agents whose config dir exists or whose binary is on PATH, in AGENTS order."""
    return [n for n, s in AGENTS.items() if home_dir(s).is_dir() or shutil.which(s.binary)]


def hook_command() -> str:
    return f"{sys.executable} -m {HOOK_MODULE}"


def _is_ours(hook: dict) -> bool:
    cmd = str(hook.get("command", ""))
    return any(marker in cmd for marker in OURS)


def _hook_entry(spec: AgentSpec) -> dict:
    hook = {"type": "command", "command": hook_command(), "timeout": spec.timeout}
    if spec.name == "gemini":
        hook["name"] = "vocal-say"
    return {"hooks": [hook]}


def _strip_ours(entries: list) -> list:
    kept = []
    for entry in entries:
        if not isinstance(entry, dict):
            kept.append(entry)
            continue
        inner = [h for h in entry.get("hooks", []) if not (isinstance(h, dict) and _is_ours(h))]
        if inner:
            kept.append({**entry, "hooks": inner})
    return kept


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text() or "{}")
    except ValueError as e:
        raise AgentConfigError(f"{path} is not valid JSON ({e}); fix or move it and re-run") from None
    if not isinstance(data, dict):
        raise AgentConfigError(f"{path} must contain a JSON object at the top level")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _merged_hooks(data: dict, spec: AgentSpec, add: bool) -> dict:
    data = json.loads(json.dumps(data))  # deep copy
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise AgentConfigError(f"'hooks' in {hooks_path(spec)} is not a JSON object")
    entries = _strip_ours(list(hooks.get(spec.event) or []))
    if add:
        entries.append(_hook_entry(spec))
    if entries:
        hooks[spec.event] = entries
    else:
        hooks.pop(spec.event, None)
    if not hooks:
        data.pop("hooks")
    return data


def _without_block(text: str) -> str:
    rest = re.sub(r"\n{3,}", "\n\n", BLOCK_RE.sub("", text)).strip("\n")
    return rest + "\n" if rest else ""


def _with_block(text: str) -> str:
    rest = _without_block(text)
    return (rest + "\n" if rest else "") + INSTRUCTIONS


def _apply(names: list[str], add: bool) -> list[Path]:
    specs = [AGENTS[n] for n in names]
    # Validate every file first so a bad one aborts before anything is written.
    plans = []
    for spec in specs:
        before = _load(hooks_path(spec))
        after = _merged_hooks(before, spec, add)
        ipath = instructions_path(spec)
        itext = ipath.read_text() if ipath.exists() else ""
        new_itext = _with_block(itext) if add else _without_block(itext)
        plans.append((spec, before, after, ipath, itext, new_itext))

    touched: list[Path] = []
    for spec, before, after, ipath, itext, new_itext in plans:
        hpath = hooks_path(spec)
        if after != before or (add and not hpath.exists()):
            _write_json(hpath, after)
            touched.append(hpath)
        if new_itext != itext or (add and not ipath.exists()):
            ipath.parent.mkdir(parents=True, exist_ok=True)
            ipath.write_text(new_itext)
            touched.append(ipath)
    return touched


def install(names: list[str]) -> list[Path]:
    """Register the hook and instruction block for each agent. Returns files written."""
    return _apply(names, add=True)


def uninstall(names: list[str]) -> list[Path]:
    """Remove our hook entries and instruction block. Returns files changed."""
    return _apply(names, add=False)


def is_installed(name: str) -> bool:
    spec = AGENTS[name]
    try:
        entries = _load(hooks_path(spec)).get("hooks", {}).get(spec.event, [])
    except AgentConfigError:
        return False
    return any(_is_ours(h) for e in entries if isinstance(e, dict) for h in e.get("hooks", []) if isinstance(h, dict))
