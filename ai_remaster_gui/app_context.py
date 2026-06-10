"""Holds the PipelineApp singleton shared across the GUI modules.

server.py assigns ``APP`` here as soon as the instance exists. Every other module reads it
through this module (``from . import app_context`` then ``app_context.APP`` at call time)
instead of importing server, which would be circular — server imports them all.

``APP`` is ``None`` only while server.py itself is still importing; by the time any request
handler or helper runs, it is always set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .server import PipelineApp

APP: "PipelineApp | None" = None
