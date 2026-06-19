"""Shared application state.

Holds the single :class:`PipelineApp` instance. ``server.py`` creates it at import time and
assigns it to :data:`APP` here, so the sibling modules (cache, media, references, outpaint_guides,
project_io, http_handler, …) can reach shared app state via ``state.APP`` without importing
``server`` — which would be a circular import, since ``server`` imports all of them.

Sibling modules read ``state.APP`` *inside* functions (request-handling time), by which point the
instance has been registered, so the ``None`` placeholder is never observed in practice.

This module intentionally has no project imports, so it stays a safe leaf in the dependency graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import PipelineApp

APP: "PipelineApp | None" = None
