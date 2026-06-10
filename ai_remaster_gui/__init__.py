"""Local web GUI for the AI remaster pipeline.

Shared artifact identity/naming/sizing lives in scripts/artifact_ids.py (stdlib-only) so the
producer scripts and this package can never drift apart. Put scripts/ on sys.path once, here,
so every module in the package can simply ``import artifact_ids``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
