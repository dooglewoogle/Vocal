"""Startup must not race first imports across threads (macOS: KeyError: 'faster_whisper')."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Mirrors run_gui: the window starts the whisper cache probe, then the main
# thread builds the engine, whose StreamingVAD imports faster_whisper.vad.
SCRIPT = """
import threading
from vocal.gui.dictation_tab import probe_whisper_cache

threads = []
def run_bg(fn, on_done=None, name=None):
    t = threading.Thread(target=lambda: on_done(fn()), name=name)
    threads.append(t)
    t.start()

results = []
probe_whisper_cache(["tiny.en", "small.en"], run_bg, results.append)
from faster_whisper.vad import get_vad_model  # noqa: F401  (what StreamingVAD does)
for t in threads:
    t.join()
assert results and set(results[0]) == {"tiny.en", "small.en"}, results
print("ok")
"""


def test_cache_probe_does_not_race_engine_import() -> None:
    # Fresh interpreters: the race only exists on a package's first import.
    # Before the fix this failed every time on Linux too.
    for _ in range(3):
        r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
