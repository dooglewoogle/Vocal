"""Phrasebook — custom vocabulary hints and post-transcription replacements.

Layer 1: Deduplicated replacement values are fed to Whisper's initial_prompt to bias decoding.
Layer 2: Replacements fix mishearings that still slip through.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from vocal.config import CONFIG_DIR

logger = logging.getLogger(__name__)

PHRASEBOOK_PATH = CONFIG_DIR / "phrasebook.toml"


@dataclass
class Phrasebook:
    """Loaded phrasebook: replacement rules + auto-derived vocabulary hints."""

    # wrong → right replacements applied after transcription
    replacements: dict[str, str] = field(default_factory=dict)

    # Compiled replacement patterns (built once at load time)
    _patterns: list[tuple[re.Pattern, str]] = field(
        default_factory=list, repr=False,
    )

    def build_initial_prompt(self) -> str | None:
        """Build initial_prompt from deduplicated replacement values."""
        if not self.replacements:
            return None
        # Deduplicate values, preserve insertion order
        terms = list(dict.fromkeys(self.replacements.values()))
        return " ".join(terms)

    def apply_replacements(self, text: str) -> str:
        """Apply all replacement rules to text. Whole-word, case-insensitive."""
        for pattern, replacement in self._patterns:
            text = pattern.sub(replacement, text)
        return text


def _compile_replacements(replacements: dict[str, str]) -> list[tuple[re.Pattern, str]]:
    """Compile replacement rules into whole-word, case-insensitive regex patterns."""
    patterns = []
    for wrong, right in replacements.items():
        pattern = re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE)
        patterns.append((pattern, right))
    return patterns


def load_phrasebook(path: Path | None = None) -> Phrasebook:
    """Load phrasebook from TOML file, returning empty Phrasebook if missing."""
    path = path or PHRASEBOOK_PATH

    if not path.exists():
        logger.warning("No phrasebook found at %s", path)
        return Phrasebook()

    if tomllib is None:
        import warnings
        warnings.warn(
            f"Phrasebook {path} found but cannot be loaded: "
            "install 'tomli' on Python < 3.11 (`pip install tomli`)",
            stacklevel=2,
        )
        return Phrasebook()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    replacements = data.get("replacements", {})

    if not isinstance(replacements, dict):
        logger.warning("phrasebook.toml: 'replacements' should be a table, ignoring")
        replacements = {}

    patterns = _compile_replacements(replacements)

    pb = Phrasebook(replacements=replacements, _patterns=patterns)
    terms = list(dict.fromkeys(replacements.values()))
    logger.info(
        "Loaded phrasebook: %d replacements, %d unique terms for initial_prompt",
        len(replacements), len(terms),
    )
    return pb
