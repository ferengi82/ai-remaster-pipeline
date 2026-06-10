"""Locator helpers mapping GUI settings to artifact paths and sizes.

These are the GUI-side counterparts of the naming the producer scripts do via
scripts/artifact_ids.py — one call per artifact kind, so the GUI and the scripts can never
disagree about where a finished render lives.
"""

from __future__ import annotations

from pathlib import Path

import artifact_ids as aid  # scripts/ is on sys.path via the package __init__

from . import app_context
from .config import ROOT
from .media import video_metrics
from .paths import even_int, rel, resolve, resolve_video_source


def _outpaint_crop_black(values: dict[str, str]) -> tuple[list[int], bool]:
    crop = [int(float(values.get(key, "0") or 0)) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom")]
    black = values.get("outpaint_all_black_regions", "false") == "true"
    return crop, black


def manifest_for_outpainted(outpainted_text: str) -> str:
    if not outpainted_text:
        return ""
    outpainted = resolve(outpainted_text)
    ident = aid.shots_identity(outpainted.stem)
    return rel(ROOT / "manifests" / "references" / aid.artifact_name(aid.source_word(outpainted.name), "shots", ident, "csv"))


def outpaint_output_for(source_text: str, aspect: str, target_height_text: str = "720") -> str:
    if not source_text:
        return ""
    source = resolve_video_source(source_text)
    # Name via the shared identity (scripts/artifact_ids.py), the same call the producer
    # (outpaint_video.default_output) makes, so the GUI and the script can never drift apart.
    width, height = outpaint_work_size_for_source(source_text, aspect, target_height_text)
    app = app_context.APP
    values = app.settings.get("outpaint", {}) if app is not None else {}
    crop, black = _outpaint_crop_black(values)
    return rel(ROOT / "intermediate" / "outpainted" / aid.outpaint_name(source.name, aspect, width, height, crop, black, "outpaint", "mp4"))


def upscale_target_size(values: dict[str, str]) -> tuple[int, int]:
    try:
        width = even_int(int(float(values.get("target_width", "3840") or 3840)))
    except ValueError:
        width = 3840
    try:
        height = even_int(int(float(values.get("target_height", "2160") or 2160)))
    except ValueError:
        height = 2160
    return max(2, width), max(2, height)


def upscale_output_for(source_text: str, values: dict[str, str]) -> str:
    if not source_text:
        return ""
    source = resolve(source_text)
    width, height = upscale_target_size(values)
    ident = aid.upscale_identity(source.stem, width, height, "flashvsr")
    return rel(ROOT / "output" / "upscaled" / aid.artifact_name(aid.source_word(source.name), "upscale", ident, "mp4"))


def soundtrack_output_for(source_text: str, values: dict[str, str]) -> str:
    if not source_text:
        return ""
    source = resolve(source_text)
    music = values.get("create_music", "true") == "true"
    sfx = values.get("create_sfx", "true") == "true"
    ident = aid.soundtrack_identity(source.stem, music, sfx)
    return rel(ROOT / "output" / "with_soundtrack" / aid.artifact_name(aid.source_word(source.name), "audio", ident, "mp4"))


def upscale_preview_output_for(source_text: str, values: dict[str, str]) -> str:
    if not source_text:
        return ""
    source = resolve(source_text)
    width, height = upscale_target_size(values)
    seconds = str(values.get("preview_seconds", "6") or "6")
    ident = aid.upscale_preview_identity(source.stem, width, height, "flashvsr", seconds)
    return rel(ROOT / "output" / "upscaled" / "previews" / aid.artifact_name(aid.source_word(source.name), "upscalepreview", ident, "mp4"))


def source_duration_text(source: Path) -> str:
    try:
        duration = float(video_metrics(source).get("duration") or 0)
    except Exception:
        return ""
    return f"{duration:.3f}" if duration > 0 else ""


def source_video_height(source_text: str) -> int:
    try:
        source = resolve_video_source(source_text)
        metrics = video_metrics(source)
        return even_int(int(metrics.get("height") or 720))
    except Exception:
        return 720


# Size math is centralised in scripts/artifact_ids.py so the GUI and the producer scripts agree.
def resolved_outpaint_height(source_text: str, target_height_text: str = "720") -> int:
    return aid.resolved_height(source_video_height(source_text), target_height_text)


def outpaint_size_for_source(source_text: str, aspect: str, target_height_text: str = "720") -> tuple[int, int]:
    return aid.delivery_size(source_video_height(source_text), aspect, target_height_text)


def outpaint_work_size_for_source(source_text: str, aspect: str, target_height_text: str = "720") -> tuple[int, int]:
    return aid.work_size(source_video_height(source_text), aspect, target_height_text)


def outpaint_chunk_dir_for(source_text: str, values: dict[str, str]) -> Path:
    source = resolve_video_source(source_text)
    aspect = values.get("target_aspect", "16:9")
    width, height = outpaint_work_size_for_source(source_text, aspect, values.get("target_height", "720"))
    crop, black = _outpaint_crop_black(values)
    return ROOT / ".cache" / "outpaint_chunks" / aid.outpaint_basename(source.name, aspect, width, height, crop, black, "chunks")


def outpaint_chunk_manifest_for(source_text: str, values: dict[str, str]) -> str:
    if not source_text:
        return ""
    source = resolve_video_source(source_text)
    aspect = values.get("target_aspect", "16:9")
    width, height = outpaint_work_size_for_source(source_text, aspect, values.get("target_height", "720"))
    crop, black = _outpaint_crop_black(values)
    return rel(ROOT / "manifests" / "outpaint_chunks" / aid.outpaint_name(source.name, aspect, width, height, crop, black, "chunks", "csv"))


def outpaint_chunk_offset_slug(row: dict[str, str]) -> str:
    try:
        offset_x = int(float(row.get("offset_x", "0") or 0))
        offset_y = int(float(row.get("offset_y", "0") or 0))
    except ValueError:
        offset_x = offset_y = 0
    return "" if not (offset_x or offset_y) else f"_ox{offset_x:+d}_oy{offset_y:+d}"


def outpaint_prepared_for(source_text: str, values: dict[str, str]) -> Path:
    source = resolve_video_source(source_text)
    aspect = values.get("target_aspect", "16:9")
    height_text = values.get("target_height", "720")
    work_w, work_h = outpaint_work_size_for_source(source_text, aspect, height_text)
    crop, black = _outpaint_crop_black(values)
    return ROOT / "intermediate" / "outpaint_prepared" / aid.outpaint_name(source.name, aspect, work_w, work_h, crop, black, "prepared", "mp4")
