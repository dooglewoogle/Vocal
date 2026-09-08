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
