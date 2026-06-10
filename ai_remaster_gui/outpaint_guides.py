from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import app_context
from .artifacts import (
    outpaint_chunk_dir_for,
    outpaint_chunk_manifest_for,
    outpaint_chunk_offset_slug,
    outpaint_prepared_for,
    outpaint_size_for_source,
    outpaint_work_size_for_source,
)
from .config import FILE_PREVIEW_DIR, IMAGE_EXTS, OUTPAINT_PROMPT, QWEN_IMAGE_EDIT_MODEL, ROOT, SCRIPTS, current_config
from .file_dialogs import browse_path
from .manifests import read_outpaint_chunk_rows, write_outpaint_chunk_rows
from .media import aspect_preview_at, ensure_source_section_clip, extract_video_frame_at, pipeline_source_text, video_metrics
from .paths import rel, resolve, resolve_video_source
from .process_utils import format_timecode
from .runtime_settings import qwen_masked_workflow_for
from .sam_masks import sam2_mask_for_image

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import artifact_ids as aid  # noqa: E402
from guide_frame_utils import guide_output_size_for_prepared, save_edge_mask_for_image  # noqa: E402



def chunk_frame_preview(source: Path, seconds: float, suffix: str) -> str:
    if not source.exists():
        return ""
    return extract_video_frame_at(source, FILE_PREVIEW_DIR / "chunks", f"{suffix}_{int(seconds * 1000):010d}", seconds)

def _parse_guide_frames(row: dict[str, str]) -> list[dict]:
    """Return the guide_frames list for a manifest row, migrating from old fields if needed."""
    raw = row.get("guide_frames", "").strip()
    if raw:
        try:
            frames = json.loads(raw)
            if isinstance(frames, list):
                return frames
        except json.JSONDecodeError:
            pass
    # Migrate from legacy guide_image / guide_end_image fields.
    frames: list[dict] = []
    if row.get("guide_image"):
        try:
            strength = float(row.get("guide_strength", "0.7") or "0.7")
        except ValueError:
            strength = 0.7
        frames.append({"frame_idx": 0, "strength": round(strength, 3), "image": row["guide_image"]})
    if row.get("guide_end_image"):
        try:
            strength = float(row.get("guide_end_strength", "1.0") or "1.0")
        except ValueError:
            strength = 1.0
        frames.append({"frame_idx": -1, "strength": round(strength, 3), "image": row["guide_end_image"]})
    return frames

def _save_guide_frames(manifest: Path, chunk_index: int, frames: list[dict]) -> None:
    """Persist guide_frames JSON back to the manifest row."""
    rows = read_outpaint_chunk_rows(manifest)
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    rows[chunk_index]["guide_frames"] = json.dumps(frames)
    write_outpaint_chunk_rows(manifest, [rows[k] for k in sorted(rows)])

def _guide_source_seconds(row: dict, frame_idx: int, fps: float) -> float:
    """Convert a frame_idx (possibly negative) to absolute seconds in the prepared canvas."""
    start = float(row.get("start", 0.0))
    end = float(row.get("end", 0.0))
    length_frames = int(row.get("end_frame", 0)) - int(row.get("start_frame", 0))
    if frame_idx < 0:
        actual = max(0, length_frames + frame_idx)
    else:
        actual = frame_idx
    return max(start, min(end - (1.0 / max(1.0, fps)), start + actual / max(1.0, fps)))

def _build_guide_frames_view(
    row: dict,
    source_text: str,
    aspect: str,
    start_seconds: float,
    end_seconds: float,
    fps: float,
    length_frames: int,
) -> list[dict]:
    """Build the view list for guide frames, including thumbnail previews."""
    frames = _parse_guide_frames(row)
    view = []
    for i, gf in enumerate(frames):
        frame_idx = int(gf.get("frame_idx", 0))
        strength = float(gf.get("strength", 0.7))
        image_rel = gf.get("image", "")
        image_path = resolve(image_rel) if image_rel else None
        image_exists = bool(image_path and image_path.exists())
        view.append({
            "guide_index": i,
            "frame_idx": frame_idx,
            "strength": strength,
            "image": image_rel,
            "image_exists": image_exists,
            "image_mtime": int(image_path.stat().st_mtime_ns) if image_exists and image_path else 0,
            "source_preview": "",
        })
    return view

def guide_frame_generation_command(chunk_index: int, guide_index: int, frame_idx: int, prompt: str) -> tuple[list[str], str, Path, float]:
    """Build the Qwen generation command for any guide frame position."""
    state = outpaint_chunks_state(app_context.APP.settings, sync=True)
    rows = state.get("rows", [])
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    if chunk_index < 0 or chunk_index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")

    row = rows[chunk_index]
    fps = float(row.get("fps", 24) or 24)
    source_seconds = _guide_source_seconds(row, frame_idx, fps)

    source_text = pipeline_source_text(app_context.APP.settings)
    if not source_text:
        raise RuntimeError("No source material is selected.")
    range_source = ensure_outpaint_prepared_canvas(source_text, app_context.APP.settings.get("outpaint", {}))
    cache_key = f"gf_qwen_{int(source_seconds * 1000):010d}"
    preview_rel = chunk_frame_preview(range_source, source_seconds, cache_key)
    source_img = resolve(preview_rel) if preview_rel else Path("")
    if not source_img.is_file():
        raise FileNotFoundError(f"Could not extract source frame for Qwen guide at {source_seconds:.3f}s from {range_source}.")

    manifest = resolve(str(manifest_text))
    output_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    output = output_dir / f"chunk_{chunk_index:04d}_guide_{guide_index:02d}_qwen.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_cached_file(output)
    source_img = save_qwen_input_copy(source_img, output.with_name(f"chunk_{chunk_index:04d}_guide_{guide_index:02d}_qwen_input{source_img.suffix.lower() or '.jpg'}"))

    # Pre-write the output path into guide_frames so the thumbnail updates immediately.
    stored = read_outpaint_chunk_rows(manifest)
    if chunk_index in stored:
        frames = _parse_guide_frames(stored[chunk_index])
        if 0 <= guide_index < len(frames):
            frames[guide_index]["image"] = rel(output)
            frames[guide_index].pop("seed", None)
        else:
            frames.append({"frame_idx": frame_idx, "strength": 0.7, "image": rel(output)})
        stored[chunk_index]["guide_frames"] = json.dumps(frames)
        write_outpaint_chunk_rows(manifest, [stored[k] for k in sorted(stored)])

    guide_prompt = prompt.strip() or DEFAULT_ANCHOR_PROMPT
    mask = save_edge_mask_for_image(source_img, output.with_name(f"chunk_{chunk_index:04d}_guide_{guide_index:02d}_qwen_edge_mask.png"))
    cmd = auto_masked_guide_command(source_img, output, guide_prompt, mask)
    return cmd, rel(output), resolve(range_source), source_seconds

def _composite_guide_in_place(output: Path, prepared_canvas: Path, source_seconds: float | None = None) -> None:
    """Composite a Qwen guide PNG with actual source content, then inpaint black corners, in-place.

    Steps:
      1. Scale the Qwen guide to the LTX work canvas size.
      2. Overlay actual source pixels from the prepared canvas wherever they are non-black
         (i.e. the source content area â€” e.g. 960Ã—704 centred in 1280Ã—704).  This ensures
         pixel-accurate alignment between the guide and the prepared canvas regardless of any
         sub-pixel shifts introduced by Qwen's internal patch processing.
      3. Inpaint any remaining near-black pixels (corners where both guide and source are black).
    Saves the result back over *output*.

    source_seconds: timestamp in the prepared canvas to use as the source frame.
    Defaults to t=0 (actual first frame).
    """
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    # Read prepared canvas dimensions and extract the source frame.
    cap = cv2.VideoCapture(str(prepared_canvas))
    canvas_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    canvas_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    seek_ms = (source_seconds * 1000.0) if source_seconds is not None else 0.0
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
    ok, src_frame = cap.read()
    if not ok or src_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, src_frame = cap.read()
    cap.release()
    if not ok or src_frame is None:
        raise RuntimeError(f"Could not read frame from prepared canvas: {prepared_canvas}")
    if src_frame.shape[1] != canvas_w or src_frame.shape[0] != canvas_h:
        src_frame = cv2.resize(src_frame, (canvas_w, canvas_h), interpolation=cv2.INTER_LANCZOS4)

    with PILImage.open(output) as img:
        img_w, img_h = img.size
        guide_rgb = img.convert("RGB")

    # Preserve the raw Qwen output alongside the composited result for inspection.
    raw_copy = output.with_name(output.stem + "_raw" + output.suffix)
    if not raw_copy.exists():
        import shutil as _shutil
        _shutil.copy2(output, raw_copy)

    resampling = getattr(PILImage, "Resampling", PILImage).LANCZOS

    guide_w, guide_h = guide_output_size_for_prepared(prepared_canvas, canvas_w, canvas_h)
    if src_frame.shape[1] != guide_w or src_frame.shape[0] != guide_h:
        src_frame = cv2.resize(src_frame, (guide_w, guide_h), interpolation=cv2.INTER_LANCZOS4)

    # Step 1: fill-resize Qwen output to the model-safe guide canvas. Do not preserve
    # Qwen's AR here; the prepared video geometry is the authority.
    canvas_pil = guide_rgb.resize((guide_w, guide_h), resampling)

    canvas_bgr = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)

    # Step 2: blend the source frame's content pixels over the centre with a soft edge.
    # black_lift raises all source pixels above 0; the padding margins are exact black (0,0,0).
    # A ~10px Gaussian feather at the content boundary avoids a hard seam where Qwen's
    # outpainting meets the composited source pixels.
    src_is_content = np.any(src_frame > 4, axis=2)
    if src_is_content.any():
        feather_px = 10
        alpha = cv2.GaussianBlur(
            src_is_content.astype(np.float32),
            (feather_px * 2 + 1, feather_px * 2 + 1),
            feather_px / 2,
        )
        # Mask the alpha back to zero outside the content area so the blur never
        # bleeds black pillar pixels into Qwen's outpainting â€” feather is inward only.
        alpha = (alpha * src_is_content.astype(np.float32))[:, :, np.newaxis]
        canvas_bgr = (
            src_frame.astype(np.float32) * alpha
            + canvas_bgr.astype(np.float32) * (1.0 - alpha)
        ).clip(0, 255).astype(np.uint8)

    # Step 3: inpaint the small corner triangles that remain black
    # (top/bottom strips outside both the Qwen letterbox and the source content area).
    still_black = np.all(canvas_bgr <= 4, axis=2).astype(np.uint8) * 255
    if still_black.any():
        canvas_bgr = cv2.inpaint(canvas_bgr, still_black, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    PILImage.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)).save(output, format="PNG")

def save_qwen_input_copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.copy2(source, target)
    return target

def auto_masked_guide_command(source: Path, output: Path, prompt: str, mask: Path) -> list[str]:
    values = app_context.APP.settings.get("references", {})
    config = current_config()
    workflow = qwen_masked_workflow_for(values, config)
    if not workflow:
        raise RuntimeError("Automatic guide generation needs a Qwen masked edit workflow.")
    if not resolve(workflow).is_file():
        raise FileNotFoundError(f"Masked edit workflow not found: {workflow}")
    cmd = [
        sys.executable, "-u",
        str(SCRIPTS / "edit_reference_image.py"),
        "--source-image", str(source),
        "--mask", rel(mask),
        "--output", rel(output),
        "--workflow", workflow,
        "--comfy-url", values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188"),
        "--comfy-dir", config.get("comfy_dir", str(ROOT / "tools" / "comfyui")),
        "--comfy-output-root", values.get("comfy_output_root") or str(Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))) / "output"),
        "--model-backend", values.get("model_backend", "gguf"),
        "--gguf-model", values.get("gguf_model", QWEN_IMAGE_EDIT_MODEL),
        "--instruction", prompt,
        "--load-image-node-id", values.get("load_image_node_id", "auto"),
        "--save-node-id", values.get("save_node_id", "auto"),
        "--no-normalize-to-source-size",
        "--force",
    ]
    if values.get("prompt_node_id"):
        cmd.extend(["--prompt-node-id", values["prompt_node_id"]])
    return cmd

def outpaint_guide_generation_command(index: int, prompt: str) -> tuple[list[str], str, Path]:
    state = outpaint_chunks_state(app_context.APP.settings, sync=True)
    rows = state.get("rows", [])
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    row = rows[index]
    start_seconds = float(row.get("start", 0.0))
    guide_source_seconds = start_seconds
    source_text = pipeline_source_text(app_context.APP.settings)
    if not source_text:
        raise RuntimeError("No source material is selected.")
    range_source = ensure_outpaint_prepared_canvas(source_text, app_context.APP.settings.get("outpaint", {}))
    preview_rel = chunk_frame_preview(range_source, guide_source_seconds, "source_guide_qwen")
    source = resolve(preview_rel) if preview_rel else Path("")
    if not source.is_file():
        raise FileNotFoundError(f"Could not extract source frame for Qwen guide at {guide_source_seconds:.3f}s from {range_source}.")

    manifest = resolve(str(manifest_text))
    output_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    output = output_dir / f"chunk_{index:04d}_guide_qwen.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_cached_file(output)
    source = save_qwen_input_copy(source, output.with_name(f"chunk_{index:04d}_guide_qwen_input{source.suffix.lower() or '.jpg'}"))

    stored = read_outpaint_chunk_rows(manifest)
    if index not in stored:
        raise IndexError(f"Outpaint chunk not found in manifest: {index + 1}")
    stored[index]["guide_image"] = rel(output)
    write_outpaint_chunk_rows(manifest, [stored[key] for key in sorted(stored)])

    guide_prompt = prompt.strip() or DEFAULT_ANCHOR_PROMPT
    mask = save_edge_mask_for_image(source, output.with_name(f"chunk_{index:04d}_guide_qwen_edge_mask.png"))
    cmd = auto_masked_guide_command(source, output, guide_prompt, mask)
    return cmd, rel(output), resolve(range_source), guide_source_seconds

def outpaint_end_guide_generation_command(index: int, prompt: str) -> tuple[list[str], str, Path, float]:
    state = outpaint_chunks_state(app_context.APP.settings, sync=True)
    rows = state.get("rows", [])
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    row = rows[index]
    fps = float(row.get("fps", 24) or 24)
    end_seconds = float(row.get("end", 0.0))
    # Use the last meaningful frame (end - 1/fps) as the Qwen source for the end guide.
    guide_source_seconds = max(float(row.get("start", 0.0)), end_seconds - (1.0 / max(1.0, fps)))
    source_text = pipeline_source_text(app_context.APP.settings)
    if not source_text:
        raise RuntimeError("No source material is selected.")
    range_source = ensure_outpaint_prepared_canvas(source_text, app_context.APP.settings.get("outpaint", {}))
    preview_rel = chunk_frame_preview(range_source, guide_source_seconds, "source_guide_end_qwen")
    source = resolve(preview_rel) if preview_rel else Path("")
    if not source.is_file():
        raise FileNotFoundError(f"Could not extract source frame for Qwen end guide at {guide_source_seconds:.3f}s from {range_source}.")

    manifest = resolve(str(manifest_text))
    output_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    output = output_dir / f"chunk_{index:04d}_guide_end_qwen.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_cached_file(output)
    source = save_qwen_input_copy(source, output.with_name(f"chunk_{index:04d}_guide_end_qwen_input{source.suffix.lower() or '.jpg'}"))

    stored = read_outpaint_chunk_rows(manifest)
    if index not in stored:
        raise IndexError(f"Outpaint chunk not found in manifest: {index + 1}")
    stored[index]["guide_end_image"] = rel(output)
    write_outpaint_chunk_rows(manifest, [stored[key] for key in sorted(stored)])

    guide_prompt = prompt.strip() or DEFAULT_ANCHOR_PROMPT
    mask = save_edge_mask_for_image(source, output.with_name(f"chunk_{index:04d}_guide_end_qwen_edge_mask.png"))
    cmd = auto_masked_guide_command(source, output, guide_prompt, mask)
    return cmd, rel(output), resolve(range_source), guide_source_seconds


def _get_guide_manifest() -> tuple[Path, dict[int, dict[str, str]], str]:
    state = outpaint_chunks_state(app_context.APP.settings, sync=True)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    return manifest, rows, manifest_text

def add_guide_frame(chunk_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    frames.append({"frame_idx": 0, "strength": 0.7, "image": ""})
    _save_guide_frames(manifest, chunk_index, frames)
    app_context.APP.log.append(f"Added guide frame to chunk {chunk_index + 1} (total: {len(frames)})")
    return {"guide_index": len(frames) - 1}

def remove_guide_frame(chunk_index: int, guide_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    removed = frames.pop(guide_index)
    if removed.get("image"):
        remove_cached_file(resolve(removed["image"]))
    _save_guide_frames(manifest, chunk_index, frames)
    app_context.APP.log.append(f"Removed guide frame {guide_index} from chunk {chunk_index + 1}")
    return {"removed": guide_index}

def save_guide_frame(chunk_index: int, guide_index: int, frame_idx: int, strength: float) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    frames[guide_index]["frame_idx"] = int(frame_idx)
    frames[guide_index]["strength"] = round(max(0.0, min(1.0, float(strength))), 3)
    _save_guide_frames(manifest, chunk_index, frames)
    app_context.APP.log.append(f"Saved guide frame {guide_index} for chunk {chunk_index + 1}: frame_idx={frame_idx}, strength={strength:.2f}")
    return {"frame_idx": frame_idx, "strength": frames[guide_index]["strength"]}

def upload_guide_frame_image(chunk_index: int, guide_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    current = frames[guide_index].get("image", "")
    selected = browse_path("image", current)
    if not selected:
        return {"selected": "", "image": current}
    source = resolve(selected)
    if source.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError("Choose a PNG or JPEG image for the guide frame.")
    target_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    target = target_dir / f"chunk_{chunk_index:04d}_guide_{guide_index:02d}{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    frames[guide_index]["image"] = rel(target)
    frames[guide_index].pop("seed", None)
    _save_guide_frames(manifest, chunk_index, frames)
    app_context.APP.log.append(f"Uploaded guide frame {guide_index} for chunk {chunk_index + 1}: {rel(target)}")
    return {"selected": selected, "image": rel(target)}

def clear_guide_frame_image(chunk_index: int, guide_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    current = frames[guide_index].get("image", "")
    if current:
        remove_cached_file(resolve(current))
    frames[guide_index]["image"] = ""
    _save_guide_frames(manifest, chunk_index, frames)
    app_context.APP.log.append(f"Cleared guide frame {guide_index} image for chunk {chunk_index + 1}")
    return {"image": ""}

def _guide_edit_dir(manifest: Path, chunk_index: int, guide_index: int) -> Path:
    return ROOT / "intermediate" / "outpaint_guides" / manifest.stem / "edits" / f"chunk_{chunk_index:04d}_guide_{guide_index:02d}"

def _next_guide_edit_output(manifest: Path, chunk_index: int, guide_index: int) -> Path:
    folder = _guide_edit_dir(manifest, chunk_index, guide_index)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = folder / f"edit_{stamp}.png"
    if not base.exists() and not base.with_suffix(base.suffix + ".json").exists():
        return base
    suffix = 1
    while True:
        candidate = folder / f"edit_{stamp}_{suffix:02d}.png"
        if not candidate.exists() and not candidate.with_suffix(candidate.suffix + ".json").exists():
            return candidate
        suffix += 1

def _save_guide_edit_mask(manifest: Path, chunk_index: int, guide_index: int, mask_data: str) -> str:
    if not mask_data:
        return ""
    import base64

    payload = mask_data.split(",", 1)[1] if "," in mask_data else mask_data
    raw = base64.b64decode(payload)
    folder = _guide_edit_dir(manifest, chunk_index, guide_index)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"mask_{time.strftime('%Y%m%d_%H%M%S')}.png"
    path.write_bytes(raw)
    return rel(path)

def _guide_edit_prompt(instruction: str, sampled_color: str = "") -> str:
    parts = [instruction.strip()]
    if sampled_color.strip():
        parts.append(f"Use the sampled colour/value exactly where relevant: {sampled_color.strip()}.")
    return " ".join(part for part in parts if part).strip() or DEFAULT_ANCHOR_PROMPT

def normalize_guide_preview_to_source(output: Path, source: Path) -> None:
    """Fill-resize a Qwen guide-edit preview to the editor source image size."""
    if not output.is_file() or not source.is_file():
        return
    from PIL import Image as PILImage

    raw_copy = output.with_name(output.stem + "_raw" + output.suffix)
    if not raw_copy.exists():
        shutil.copy2(output, raw_copy)
    with PILImage.open(source) as src_img:
        target_size = src_img.size
    with PILImage.open(output) as out_img:
        if out_img.size == target_size:
            return
        resampling = getattr(PILImage, "Resampling", PILImage).LANCZOS
        out_img.convert("RGB").resize(target_size, resampling).save(output, format="PNG")

def _guide_editor_source(chunk_index: int, guide_index: int, frames: list[dict]) -> tuple[str, Path | None, float | None]:
    current = frames[guide_index].get("image", "")
    if current and resolve(current).is_file():
        return current, None, None
    state = outpaint_chunks_state(app_context.APP.settings)
    rows = state.get("rows", [])
    if chunk_index < 0 or chunk_index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    row = rows[chunk_index]
    fps = float(row.get("fps", 24) or 24)
    source_seconds = _guide_source_seconds(row, int(frames[guide_index].get("frame_idx", 0)), fps)
    source_text = pipeline_source_text(app_context.APP.settings)
    if not source_text:
        raise RuntimeError("No source material is selected.")
    prepared = ensure_outpaint_prepared_canvas(source_text, app_context.APP.settings.get("outpaint", {}))
    preview_rel = chunk_frame_preview(prepared, source_seconds, f"guide_edit_{chunk_index}_{guide_index}")
    if not preview_rel or not resolve(preview_rel).is_file():
        raise FileNotFoundError("Could not prepare a guide image for editing.")
    return preview_rel, prepared, source_seconds

def guide_edit_preview_command(chunk_index: int, guide_index: int, instruction: str, mask_data: str = "", sampled_color: str = "") -> tuple[list[str], str]:
    manifest, rows, _manifest_text = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    source_rel, _prepared, _source_seconds = _guide_editor_source(chunk_index, guide_index, frames)
    source = resolve(source_rel)
    output = _next_guide_edit_output(manifest, chunk_index, guide_index)
    mask = _save_guide_edit_mask(manifest, chunk_index, guide_index, mask_data)
    if not mask:
        mask_path = _guide_edit_dir(manifest, chunk_index, guide_index) / f"mask_edge_{time.strftime('%Y%m%d_%H%M%S')}.png"
        mask = rel(save_edge_mask_for_image(source, mask_path))
    prompt = _guide_edit_prompt(instruction, sampled_color)
    values = app_context.APP.settings.get("references", {})
    config = current_config()
    comfy_dir = config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))
    comfy_url = values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188")
    comfy_output = values.get("comfy_output_root") or str(Path(comfy_dir) / "output")
    workflow = qwen_masked_workflow_for(values, config)
    if not workflow:
        raise RuntimeError("Guide editing needs a Qwen masked edit workflow. ARP's bundled masked workflow was not found, and no custom workflow is set.")
    if not resolve(workflow).is_file():
        raise FileNotFoundError(f"Masked edit workflow not found: {workflow}")
    cmd = [
        sys.executable, "-u", str(SCRIPTS / "edit_reference_image.py"),
        "--source-image", str(source),
        "--mask", mask,
        "--output", rel(output),
        "--workflow", workflow,
        "--comfy-url", comfy_url,
        "--comfy-dir", comfy_dir,
        "--comfy-output-root", comfy_output,
        "--model-backend", values.get("model_backend", "gguf"),
        "--gguf-model", values.get("gguf_model", QWEN_IMAGE_EDIT_MODEL),
        "--instruction", prompt,
        "--no-normalize-to-source-size",
        "--force",
    ]
    if values.get("prompt_node_id"):
        cmd.extend(["--prompt-node-id", values["prompt_node_id"]])
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "chunk_index": chunk_index,
                "guide_index": guide_index,
                "source_image": rel(source),
                "mask": mask,
                "instruction": instruction,
                "sampled_color": sampled_color,
                "prompt": prompt,
                "output": rel(output),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return cmd, rel(output)

def sam_guide_mask(chunk_index: int, guide_index: int, points: list[dict], width: int, height: int, fallback_path: str = "") -> dict[str, str]:
    manifest, rows, _manifest_text = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    source_rel = frames[guide_index].get("image", "") or fallback_path
    if not source_rel:
        source_rel, _prepared, _source_seconds = _guide_editor_source(chunk_index, guide_index, frames)
    source = resolve(source_rel)
    if not source.is_file():
        raise FileNotFoundError(f"Guide image not found: {source_rel}")
    return sam2_mask_for_image(source, points, width, height)

def accept_guide_edit(chunk_index: int, guide_index: int, preview_path: str) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    preview = resolve(preview_path)
    if not preview.is_file():
        raise FileNotFoundError(f"Edited guide not found: {preview}")
    previous = frames[guide_index].get("image", "")
    frames[guide_index]["image_previous"] = previous
    frames[guide_index]["image"] = rel(preview)
    frames[guide_index].pop("seed", None)
    _save_guide_frames(manifest, chunk_index, frames)
    app_context.APP.log.append(f"Accepted edited guide frame {guide_index + 1} for chunk {chunk_index + 1}: {rel(preview)}")
    return {"image": rel(preview), "previous": previous}

def revert_guide_edit(chunk_index: int, guide_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    current = frames[guide_index].get("image", "")
    previous = frames[guide_index].get("image_previous", "")
    if not previous:
        raise RuntimeError("No previous guide image is recorded for this guide frame.")
    if not resolve(previous).is_file():
        raise FileNotFoundError(f"Previous guide image not found: {previous}")
    frames[guide_index]["image_previous"] = current
    frames[guide_index]["image"] = previous
    _save_guide_frames(manifest, chunk_index, frames)
    app_context.APP.log.append(f"Reverted guide frame {guide_index + 1} for chunk {chunk_index + 1}: {previous}")
    return {"image": previous, "previous": current}


DEFAULT_ANCHOR_PROMPT = "Replace the black bars."


def ensure_outpaint_prepared_canvas(source_text: str, values: dict[str, str]) -> Path:
    source = resolve_video_source(source_text)
    prepared = outpaint_prepared_for(source_text, values)
    if prepared.exists():
        return prepared

    cmd = [
        sys.executable,
        str(SCRIPTS / "prepare_outpaint_input.py"),
        "--source",
        str(source),
        "--target-aspect",
        values.get("target_aspect", "16:9"),
        "--black-lift",
        str(values.get("black_lift", "0.018") or "0.018"),
        "--gamma",
        str(values.get("gamma", "1.06") or "1.06"),
        "--output",
        str(prepared),
        "--crop-left",
        str(values.get("crop_left", "0") or "0"),
        "--crop-right",
        str(values.get("crop_right", "0") or "0"),
        "--crop-top",
        str(values.get("crop_top", "0") or "0"),
        "--crop-bottom",
        str(values.get("crop_bottom", "0") or "0"),
        "--target-width",
        str(outpaint_work_size_for_source(source_text, values.get("target_aspect", "16:9"), values.get("target_height", "720"))[0]),
        "--target-height",
        str(outpaint_work_size_for_source(source_text, values.get("target_aspect", "16:9"), values.get("target_height", "720"))[1]),
        "--delivery-width",
        str(outpaint_size_for_source(source_text, values.get("target_aspect", "16:9"), values.get("target_height", "720"))[0]),
        "--delivery-height",
        str(outpaint_size_for_source(source_text, values.get("target_aspect", "16:9"), values.get("target_height", "720"))[1]),
    ]
    if values.get("outpaint_all_black_regions", "false") == "true":
        cmd.append("--outpaint-all-black-regions")
    app_context.APP.log.append(f"Preparing expanded canvas for guide frame: {rel(prepared)}")
    app_context.APP.log.append("> " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
    for line in (result.stdout or "").splitlines():
        app_context.APP.log.append(line)
    for line in (result.stderr or "").splitlines():
        app_context.APP.log.append(line)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Could not prepare expanded outpaint canvas.")
    if not prepared.exists():
        raise RuntimeError(f"Prepared expanded canvas was not created: {prepared}")
    return prepared


def outpaint_chunks_state(settings: dict, sync: bool = False) -> dict:
    """Build the chunk-table view for the Outpainting tab.

    Read-only by default so /api/state polls cannot rewrite the chunk manifest; pass
    ``sync=True`` from mutating endpoints that need the manifest written to disk before
    they edit individual rows.
    """
    try:
        ensure_source_section_clip(settings)
    except Exception as exc:
        return {"manifest": "", "rows": [], "error": f"Could not prepare selected source section: {exc}"}

    source_text = pipeline_source_text(settings)
    if not source_text:
        return {"manifest": "", "rows": []}
    source = resolve_video_source(source_text)
    if not source.exists():
        return {"manifest": "", "rows": [], "error": f"Source material is not a readable file: {source}"}
    values = settings.get("outpaint", {})
    metrics = video_metrics(source)
    fps = metrics.get("fps") or 24.0
    total_frames = int(metrics.get("frames") or 0)
    if total_frames <= 0:
        message = f"Outpaint chunk preview skipped; could not count frames in: {source}"
        app_context.APP.log.append(message)
        return {"manifest": "", "rows": [], "error": message}
    try:
        chunk_seconds = float(values.get("chunk_seconds", "20") or 20)
    except ValueError:
        chunk_seconds = 20.0
    try:
        overlap_frames = int(float(values.get("overlap_frames", "8") or 8))
    except ValueError:
        overlap_frames = 8
    chunk_dir = outpaint_chunk_dir_for(source_text, values)
    manifest = resolve(outpaint_chunk_manifest_for(source_text, values))
    existing = read_outpaint_chunk_rows(manifest)
    ranges = outpaint_chunk_ranges(total_frames, fps, chunk_seconds, overlap_frames, existing)
    global_prompt = values.get("prompt") or OUTPAINT_PROMPT
    global_negative = values.get("negative_prompt", "")
    rows = []
    for index, start_frame, end_frame in ranges:
        row = dict(existing.get(index, {}))
        row.setdefault("offset_x", "0")
        row.setdefault("offset_y", "0")
        offset_slug = outpaint_chunk_offset_slug(row)
        prepared = chunk_dir / f"prepared_{index:04d}_{start_frame:06d}_{end_frame:06d}{offset_slug}.mp4"
        raw = chunk_dir / f"raw_{index:04d}_{start_frame:06d}_{end_frame:06d}{offset_slug}.mp4"
        row.update({
            "chunk_index": str(index),
            "start_frame": str(start_frame),
            "end_frame": str(end_frame),
            "start_seconds": f"{start_frame / fps:.6f}",
            "end_seconds": f"{end_frame / fps:.6f}",
            "prepared_path": rel(prepared),
            "raw_path": rel(raw),
        })
        row.setdefault("custom_seconds", "")
        if not row.get("seed"):
            row["seed"] = str(42 + index)
        row.setdefault("prompt_suffix", "")
        row.setdefault("negative_suffix", "")
        row.setdefault("guide_image", "")
        row.setdefault("guide_strength", "0.7")
        row.setdefault("guide_end_image", "")
        row.setdefault("guide_end_strength", "1.0")
        row.setdefault("guide_frames", "")
        rows.append(row)
    if sync:
        write_outpaint_chunk_rows(manifest, rows)
    view_rows = []
    for row in rows:
        raw = resolve(row["raw_path"])
        prepared = resolve(row["prepared_path"])
        start_seconds = float(row["start_seconds"])
        end_seconds = float(row["end_seconds"])
        length_frames = int(row["end_frame"]) - int(row["start_frame"])
        aspect = values.get("target_aspect", "16:9")
        guides = _build_guide_frames_view(row, source_text, aspect, start_seconds, end_seconds, fps, length_frames)
        view_rows.append(row | {
            "index": int(row["chunk_index"]),
            "start": float(row["start_seconds"]),
            "end": float(row["end_seconds"]),
            "fps": fps,
            "total_frames": total_frames,
            "length_frames": length_frames,
            "max_length_frames": max(1, total_frames - int(row["start_frame"])),
            "start_label": format_timecode(float(row["start_seconds"])),
            "end_label": format_timecode(float(row["end_seconds"])),
            "raw_exists": raw.exists(),
            "raw_mtime": int(raw.stat().st_mtime_ns) if raw.exists() else 0,
            "prepared_exists": prepared.exists(),
            "guides": guides,
            "source_start_preview": "",
            "source_middle_preview": "",
            "source_end_preview": "",
            "raw_start_preview": "",
            "raw_middle_preview": "",
            "raw_end_preview": "",
            "effective_prompt": aid.combine_prompt(global_prompt, row.get("prompt_suffix", "")),
            "effective_negative_prompt": aid.combine_prompt(global_negative, row.get("negative_suffix", "")),
        })
    return {"manifest": rel(manifest), "rows": view_rows}


def outpaint_chunk_preview(settings: dict, chunk_index: int, kind: str, position: str) -> str:
    chunks = outpaint_chunks_state(settings)
    row = next((r for r in chunks.get("rows", []) if int(r.get("index", -1)) == chunk_index), None)
    if row is None:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")

    position = position if position in {"start", "middle", "end"} else "middle"
    fps = max(1.0, float(row.get("fps", 24) or 24))
    start_seconds = float(row.get("start", 0.0) or 0.0)
    end_seconds = float(row.get("end", start_seconds) or start_seconds)
    duration = max(0.0, end_seconds - start_seconds)

    if position == "start":
        offset = 0.0
    elif position == "end":
        offset = max(0.0, duration - (1.0 / fps))
    else:
        offset = duration / 2

    if kind == "raw":
        raw = resolve(str(row.get("raw_path", "")))
        if not raw.exists():
            return ""
        return chunk_frame_preview(raw, offset, f"raw_{chunk_index}_{position}")

    source_text = pipeline_source_text(settings)
    if not source_text:
        return ""
    aspect = settings.get("outpaint", {}).get("target_aspect", "16:9")
    try:
        offset_x = int(float(row.get("offset_x", "0") or 0))
        offset_y = int(float(row.get("offset_y", "0") or 0))
    except ValueError:
        offset_x = offset_y = 0
    return aspect_preview_at(source_text, aspect, start_seconds + offset, offset_x, offset_y)


def outpaint_chunk_ranges(total_frames: int, fps: float, default_seconds: float, overlap_frames: int, existing: dict[int, dict[str, str]]) -> list[tuple[int, int, int]]:
    ranges = []
    start = 0
    index = 0
    while start < total_frames:
        seconds = default_seconds
        custom = existing.get(index, {}).get("custom_seconds", "")
        if custom:
            try:
                seconds = float(custom)
            except ValueError:
                seconds = default_seconds
        chunk_frames = total_frames if seconds <= 0 else max(1, int(round(seconds * fps)))
        end = min(total_frames, start + chunk_frames)
        ranges.append((index, start, end))
        if end >= total_frames:
            break
        overlap = max(0, min(overlap_frames, chunk_frames - 1))
        start += max(1, chunk_frames - overlap)
        index += 1
    return ranges


def _truthy_payload_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def update_outpaint_chunk(index: int, seed: str, prompt_suffix: str, custom_seconds: str = "", negative_suffix: str = "", guide_strength: str = "", guide_end_strength: str = "", custom_length=None, offset_x: str = "0", offset_y: str = "0") -> None:
    state = outpaint_chunks_state(app_context.APP.settings, sync=True)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    rows = read_outpaint_chunk_rows(resolve(str(manifest_text)))
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")
    row = rows[index]
    row["seed"] = str(int(float(seed or row.get("seed") or 42 + index)))
    row["prompt_suffix"] = prompt_suffix
    row["negative_suffix"] = negative_suffix
    row["offset_x"] = str(int(float(offset_x or 0)))
    row["offset_y"] = str(int(float(offset_y or 0)))
    use_custom_length = _truthy_payload_value(custom_length) if custom_length is not None else bool(custom_seconds)
    if use_custom_length and custom_seconds:
        row["custom_seconds"] = f"{max(0.1, float(custom_seconds)):.3f}"
    else:
        row["custom_seconds"] = ""
    if guide_strength:
        try:
            row["guide_strength"] = f"{max(0.0, min(1.0, float(guide_strength))):.3f}"
        except ValueError:
            pass
    if guide_end_strength:
        try:
            row["guide_end_strength"] = f"{max(0.0, min(1.0, float(guide_end_strength))):.3f}"
        except ValueError:
            pass
    ordered = [rows[key] for key in sorted(rows)]
    write_outpaint_chunk_rows(resolve(str(manifest_text)), ordered)
    app_context.APP.log.append(f"Saved outpaint chunk {index + 1}: seed {row['seed']}")


def remove_cached_file(path: Path) -> bool:
    removed = False
    for candidate in (path, path.with_suffix(path.suffix + ".sig.json"), path.with_suffix(path.suffix + ".partial")):
        try:
            if candidate.exists() and candidate.is_file():
                candidate.unlink()
                removed = True
        except PermissionError:
            app_context.APP.log.append(f"Could not delete cached file because it is open in another process: {rel(candidate)}")
        except OSError as exc:
            app_context.APP.log.append(f"Could not delete cached file {rel(candidate)}: {exc}")
    return removed


def clear_cached_guide_frames(manifest: Path, index: int) -> int:
    guide_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    if not guide_dir.exists():
        # Also check legacy path name used before the anchorâ†’guide rename.
        guide_dir = ROOT / "intermediate" / "outpaint_anchors" / manifest.stem
        if not guide_dir.exists():
            return 0
    removed = 0
    for path in guide_dir.glob(f"chunk_{index:04d}_*"):
        if path.is_file() and remove_cached_file(path):
            removed += 1
    return removed


def install_outpaint_guide(index: int) -> dict[str, str]:
    state = outpaint_chunks_state(app_context.APP.settings, sync=True)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    current = rows[index].get("guide_image", "")
    selected = browse_path("image", current)
    if not selected:
        return {"selected": "", "guide_image": current}

    source = resolve(selected)
    if source.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError("Choose a PNG or JPEG image for the outpaint guide frame.")
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)

    target_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    target = target_dir / f"chunk_{index:04d}_guide{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    rows[index]["guide_image"] = rel(target)
    write_outpaint_chunk_rows(manifest, [rows[key] for key in sorted(rows)])
    app_context.APP.log.append(f"Installed outpaint guide frame for chunk {index + 1}: {rel(target)}")
    return {"selected": selected, "guide_image": rel(target)}


def clear_outpaint_guide(index: int) -> dict[str, str]:
    state = outpaint_chunks_state(app_context.APP.settings, sync=True)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")
    removed = clear_cached_guide_frames(manifest, index)
    rows[index]["guide_image"] = ""
    if "anchor_image" in rows[index]:
        rows[index]["anchor_image"] = ""
    write_outpaint_chunk_rows(manifest, [rows[key] for key in sorted(rows)])
    suffix = f" and deleted {removed} cached file(s)" if removed else ""
    app_context.APP.log.append(f"Cleared outpaint guide frame for chunk {index + 1}{suffix}")
    return {"guide_image": ""}


def clear_outpaint_anchor(index: int) -> dict[str, str]:
    return clear_outpaint_guide(index)


def install_outpaint_end_guide(index: int) -> dict[str, str]:
    state = outpaint_chunks_state(app_context.APP.settings, sync=True)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    current = rows[index].get("guide_end_image", "")
    selected = browse_path("image", current)
    if not selected:
        return {"selected": "", "guide_end_image": current}

    source = resolve(selected)
    if source.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError("Choose a PNG or JPEG image for the outpaint end guide frame.")
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)

    target_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    target = target_dir / f"chunk_{index:04d}_guide_end{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    rows[index]["guide_end_image"] = rel(target)
    write_outpaint_chunk_rows(manifest, [rows[key] for key in sorted(rows)])
    app_context.APP.log.append(f"Installed outpaint end guide frame for chunk {index + 1}: {rel(target)}")
    return {"selected": selected, "guide_end_image": rel(target)}


def clear_outpaint_end_guide(index: int) -> dict[str, str]:
    state = outpaint_chunks_state(app_context.APP.settings, sync=True)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")
    # Remove the end guide file if it's in our managed directory.
    current = rows[index].get("guide_end_image", "")
    if current:
        path = resolve(current)
        remove_cached_file(path)
    rows[index]["guide_end_image"] = ""
    write_outpaint_chunk_rows(manifest, [rows[key] for key in sorted(rows)])
    app_context.APP.log.append(f"Cleared outpaint end guide frame for chunk {index + 1}")
    return {"guide_end_image": ""}
