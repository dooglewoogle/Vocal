"""Black-box tests for install.sh's interpreter selection.

Each case runs ``bash install.sh --no-system`` with a PATH of stub interpreters
(two-line shell scripts that print ``Python X.Y.Z``), VOCAL_HOME in a tmp dir and
``n`` on stdin, so the script prints its plan and aborts before changing anything.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "install.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _supported_from_script() -> list[str]:
    m = re.search(r'^SUPPORTED_PYTHONS="([^"]+)"', INSTALL_SH.read_text(), re.M)
    assert m, "SUPPORTED_PYTHONS not found in install.sh"
    return m.group(1).split()


def test_supported_pythons_match_pyproject():
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^requires-python\s*=\s*">=3\.(\d+),<3\.(\d+)"', text, re.M)
    assert m, "requires-python must be of the form '>=3.X,<3.Y'"
    lo, hi = int(m.group(1)), int(m.group(2))
    expected = [f"3.{n}" for n in range(hi - 1, lo - 1, -1)]  # newest first
    assert _supported_from_script() == expected


class Sandbox:
    def __init__(self, tmp: Path):
        self.bin = tmp / "bin"
        self.bin.mkdir()
        self.home = tmp / "vocal-home"
        # Only the external tools install.sh needs before the Proceed? prompt, so no
        # real python3.X on the developer's machine can leak into the probe.
        tools = tmp / "tools"
        tools.mkdir()
        for name in ("uname", "id", "grep", "sed", "dirname", "sh"):
            real = shutil.which(name)
            assert real, f"{name} not found"
            (tools / name).symlink_to(real)
        self.env = {
            "PATH": f"{self.bin}:{tools}",
            "HOME": str(tmp),
            "USER": os.environ.get("USER", "tester"),
            "VOCAL_HOME": str(self.home),
            "VOCAL_BIN": str(tmp / "localbin"),
        }

    def stub(self, name: str, body: str, where: Path | None = None) -> Path:
        path = (where or self.bin) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def python(self, name: str, version: str, where: Path | None = None) -> Path:
        return self.stub(name, f'echo "Python {version}"', where)

    def run(self, *args: str, no_system: bool = True, **env: str) -> subprocess.CompletedProcess:
        flags = ["--no-system"] if no_system else []
        return subprocess.run(
            [shutil.which("bash"), str(INSTALL_SH), *flags, *args],
            input="n\n",
            capture_output=True,
            text=True,
            env={**self.env, **env},
            cwd=ROOT,
            timeout=30,
        )


@pytest.fixture
def sb(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def test_system_python_is_used_when_supported(sb: Sandbox):
    sb.python("python3", "3.12.4")
    r = sb.run()
    assert r.returncode == 0, r.stderr
    assert "using python3 (Python 3.12.4)" in r.stdout
    assert "Aborted. Nothing was changed." in r.stdout
    assert not sb.home.exists()


def test_falls_through_to_newest_supported(sb: Sandbox):
    sb.python("python3", "3.14.3")
    sb.python("python3.13", "3.13.7")
    r = sb.run()
    assert r.returncode == 0, r.stderr
    assert "using python3.13 (Python 3.13.7)" in r.stdout


def test_no_supported_python_fails_early_on_linux(sb: Sandbox):
    sb.python("python3", "3.14.3")
    sb.stub("uname", "echo Linux")
    r = sb.run()
    assert r.returncode == 1
    assert "3.10" in r.stderr and "3.13" in r.stderr
    assert "python3 = 3.14.3" in r.stderr
    assert "PYTHON=python3.13 ./install.sh" in r.stderr
    assert "Proceed?" not in r.stdout


def test_explicit_python_override_is_not_replaced(sb: Sandbox):
    sb.python("python3", "3.12.0")
    old = sb.python("mypy314", "3.14.3")
    r = sb.run(PYTHON=str(old))
    assert r.returncode == 1
    assert "3.14.3" in r.stderr and "3.10" in r.stderr
    assert "using python3" not in r.stdout


def test_stale_venv_is_replaced(sb: Sandbox):
    sb.python("python3", "3.12.0")
    sb.python("python", "3.14.3", where=sb.home / "venv" / "bin")
    r = sb.run()
    assert r.returncode == 0, r.stderr
    assert "Replace the virtual environment" in r.stdout
    assert "Python 3.14.3, unsupported" in r.stdout


def test_macos_offers_brew_python(sb: Sandbox, tmp_path: Path):
    brew_prefix = tmp_path / "homebrew"
    brew_prefix.mkdir()
    sb.python("python3", "3.14.3")
    sb.stub("uname", "echo Darwin")
    sb.stub("brew", f'case "$1" in --prefix) echo "{brew_prefix}" ;; list) exit 1 ;; esac')
    r = sb.run(no_system=False)
    assert r.returncode == 0, r.stderr
    assert "brew install portaudio python@3.13" in r.stdout
    assert "python3 here is 3.14.3" in r.stdout
    assert "using python3.13 (installed in step 1)" in r.stdout


def test_macos_without_brew_fails_early(sb: Sandbox):
    sb.python("python3", "3.14.3")
    sb.stub("uname", "echo Darwin")
    r = sb.run()
    assert r.returncode == 1
    assert "brew install python@3.13" in r.stderr
