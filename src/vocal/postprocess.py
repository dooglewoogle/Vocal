"""Post-processing pipeline for transcribed text."""

from __future__ import annotations

import re

from vocal.config import PostprocessConfig
from vocal.phrasebook import Phrasebook

HALLUCINATION_PATTERNS = [
    re.compile(r"^\s*\[.*\]\s*$"),
    re.compile(r"^\s*\(.*\)\s*$"),
    re.compile(r"^(\s*\.)+\s*$"),
    re.compile(r"(?i)^(thank you|thanks)\.?\s*$"),
    re.compile(r"(?i)^you$"),
    re.compile(r"(?i)^(bye|goodbye)\.?\s*$"),
]

FILLER_WORDS = {"um", "uh", "hmm", "mm", "mhm", "uh-huh", "like"}
_PUNCT_CHARS = frozenset(".,!?;:'\"")


def postprocess(text: str, config: PostprocessConfig, phrasebook: Phrasebook | None = None) -> str:
    """Clean up Whisper output for dictation."""
    if not text or not text.strip():
        return ""

    text = text.strip()

    if phrasebook:
        text = phrasebook.apply_replacements(text)

    if config.remove_hallucinations:
        for pattern in HALLUCINATION_PATTERNS:
            if pattern.match(text):
                return ""

    if config.remove_filler_words:
        words = text.split()
        words = [w for w in words if w.lower().strip("".join(_PUNCT_CHARS)) not in FILLER_WORDS]
        text = " ".join(words)
        if not text:
            return ""

    if config.strip_leading_space:
        text = text.strip()

    if config.capitalize_first and text:
        text = text[0].upper() + text[1:]

    return text
