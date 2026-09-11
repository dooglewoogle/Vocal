# Lessons

Patterns from user corrections. Read at session start.

## 2026-09-08 — GUI settings review (vocal 0.4)

**Mistake:** exposed the same setting in two places (phrasebook seed/replace as checkboxes on the Settings tab *and* on the Phrasebook tab; Whisper model and voice as both a dropdown and a grid).
**Rule:** one control per setting. When a dedicated view (grid, tab) owns a choice, remove it from the generic form and keep a drift guard (`OWNED_ELSEWHERE`) so the omission is deliberate.
**Trigger:** adding any field to a settings form — first ask "is there already a widget that sets this?"

**Mistake:** kept a behaviour switch (hotkey toggle vs push-to-talk) as a config option because it existed, instead of asking whether both behaviours are needed.
**Rule:** when porting CLI options into a GUI, treat each one as a candidate for deletion. Fewer modes beats a well-labelled mode switch. Propose the removal in the plan rather than carrying the option forward.
**Trigger:** any option whose values are two behaviours a user picks once and never changes.

**Mistake:** explanatory help text under a dropdown, duplicating what short labels in the items could say.
**Rule:** put the qualifier in the choice label itself — `live (always listening)` — and drop the subtext. Reserve help text for things a label cannot carry.
**Trigger:** writing a `help=` on a choice field.

**Mistake:** grouped settings by config table (Hotkey, Speech, Server) rather than by the user's mental model (input side vs output side; ducking as its own concern).
**Rule:** tabs and sections follow what the user is doing (dictating vs. making Vocal speak), and cross-cutting knobs (ducking) get their own section on each side. Config-table names are an implementation detail.
**Trigger:** laying out a settings form from a dataclass — do not mirror the dataclass nesting.

## 2026-09-08 — second settings review

**Mistake:** kept `output.speech.backend` as a setting although the chosen voice already determines the backend; and used a "Show advanced" checkbox that sprinkled extra rows into every section.
**Rule:** a config key that is derivable from another key is not a setting — delete it. Advanced options live in one collapsed block, not interleaved with basics; the basic block should be short enough to read at a glance.
**Trigger:** any field whose value could be computed from the current config, or any UI toggle that changes the shape of several sections at once.

## 2026-09-10 — distribution review

**Mistake:** asserted "pynput is the fallback when evdev is missing on Linux" from reading our own code, without checking pynput's own dependencies. pynput requires evdev on Linux, so the fallback did not exist for a pip install.
**Rule:** before claiming an install-time fallback, do the install in a fresh venv and read `pip show <dep>` for the transitive requirements. Our code's fallback path is irrelevant if the package manager pulls the heavy dependency anyway.
**Trigger:** any statement of the form "X is optional because we fall back to Y" about a third-party package.

## 2026-09-11 — first macOS install (Python 3.14)

**Mistake:** declared `requires-python = ">=3.10"` with no ceiling while depending on kokoro-onnx, which caps at `<3.14`. The installer built a 3.14 venv and pip failed after the venv existed, leaving a broken install behind.
**Rule:** when adding a dependency, read its `requires_python` on PyPI (`curl -s https://pypi.org/pypi/<pkg>/json | jq .info.requires_python`) and mirror the tightest ceiling in pyproject. The installer probes interpreter versions before creating a venv and never builds one on an unsupported Python.
**Trigger:** any edit to the `dependencies` list, or a new default platform whose system Python may be newer than ours.
