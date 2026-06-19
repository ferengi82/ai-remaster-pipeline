"""Artifact path naming shared between the GUI locator and the producer scripts.

A leaf module (depends only on config/paths/artifact_ids) so any sibling can import these without a
cycle. Names are derived through scripts/artifact_ids.py, the single source of truth, so the GUI and
the producer scripts can never disagree on a filename.
"""

from __future__ import annotations

import sys

from .config import DATA_ROOT, SCRIPTS
from .paths import rel, resolve

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import artifact_ids as aid  # noqa: E402


def manifest_for_outpainted(outpainted_text: str) -> str:
    """Path to the shot manifest CSV for a given outpainted video (keyed on that video only)."""
    if not outpainted_text:
        return ""
    outpainted = resolve(outpainted_text)
    ident = aid.shots_identity(outpainted.stem)
    return rel(DATA_ROOT / "manifests" / "references" / aid.artifact_name(aid.source_word(outpainted.name), "shots", ident, "csv"))
