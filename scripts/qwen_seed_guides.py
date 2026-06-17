"""Automatically seed LTX outpaint chunks with Qwen-generated guide frames at shot changes.

When LTX refuses to outpaint a clip (it hands back the black pillarbox bars), the reliable
fix is to give it a start frame whose bars are already filled. This module does that
automatically: it detects shot changes in the prepared (pillarboxed) canvas, and for the
first frame of each shot runs Qwen Image Edit ("Replace the black bars.") to outpaint that
single frame, composites the real source content back over the centre, and registers it as a
guide frame for the chunk that contains it. LTX then extends from a filled anchor instead of
copying the bars.

Shot-change guides are derived purely from the source, so they are precomputed before the LTX
chunk loop. Continuation chunk starts (no shot change at the boundary) keep using the previous
chunk's output as their auto-guide, so this only adds anchors where they are actually needed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import generate_references as gr
from common import ROOT, root_relative
from guide_frame_utils import resize_frame_for_qwen, save_edge_mask_for_frame
from common import DATA_ROOT

DEFAULT_SEED_PROMPT = "Replace the black bars."


def detect_shot_start_frames(
    video: Path,
    sample_seconds: float = 0.0,
    shot_threshold: float | None = None,
    min_shot_seconds: float | None = None,
) -> list[int]:
    """Return the start frame of every detected shot in *video* (always includes frame 0).

    Reuses the Shot Detection analysis so seeded boundaries match what the colorize stage
    would find. Detection runs on the prepared canvas, whose frame indices map 1:1 onto the
    outpaint chunk frames, so a boundary is directly a chunk frame index.
    """
    argv = ["--source-video", str(video), "--sample-seconds", str(sample_seconds)]
    if shot_threshold is not None:
        argv += ["--shot-threshold", str(shot_threshold)]
    if min_shot_seconds is not None:
        argv += ["--min-shot-seconds", str(min_shot_seconds)]
    args = gr.build_parser().parse_args(argv)
    info = gr.probe_video(video)
    samples = gr.sample_video(video, info, args)
    shots = gr.detect_shots(samples, info, args)
    return sorted({int(shot.start_frame) for shot in shots})


def _composite_seed_guide(qwen_png: Path, src_frame: "np.ndarray", out_png: Path) -> None:
    """Composite the Qwen outpaint with the real source centre, then fill any black corners.

    Mirrors the GUI's guide compositing: scale Qwen's output to the canvas, overlay the
    actual (pixel-aligned) source content with a soft inward feather so only the outpainted
    margins come from Qwen, and inpaint any residual black corners.
    """
    from PIL import Image as PILImage

    height, width = src_frame.shape[:2]
    with PILImage.open(qwen_png) as img:
        guide_rgb = img.convert("RGB")
        img_w, img_h = img.size
    resampling = getattr(PILImage, "Resampling", PILImage).LANCZOS
    # Warp to the exact model-safe guide canvas instead of preserving Qwen's nearby
    # patch-friendly AR. This keeps guide frames pixel-compatible with the prepared video.
    canvas = guide_rgb.resize((width, height), resampling)

    canvas_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    content = np.any(src_frame > 4, axis=2)
    if content.any():
        feather = 10
        alpha = cv2.GaussianBlur(content.astype(np.float32), (feather * 2 + 1, feather * 2 + 1), feather / 2)
        alpha = (alpha * content.astype(np.float32))[:, :, np.newaxis]
        canvas_bgr = (src_frame.astype(np.float32) * alpha + canvas_bgr.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)

    black = np.all(canvas_bgr <= 4, axis=2).astype(np.uint8) * 255
    if black.any():
        canvas_bgr = cv2.inpaint(canvas_bgr, black, 3, cv2.INPAINT_TELEA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas_bgr)


def generate_seed_guide(prepared_frame: "np.ndarray", qwen_args: dict, out_guide: Path, force: bool = False) -> Path | None:
    """Run Qwen Image Edit on a single pillarboxed frame and composite a usable guide.

    Caches the expensive Qwen pass by raw output path. If the raw PNG already exists and
    *force* is False, rebuild the final guide from it so compositor fixes can refresh old
    guides without another Qwen render.
    """
    out_guide.parent.mkdir(parents=True, exist_ok=True)
    qwen_in = out_guide.with_name(out_guide.stem + "_qwen_input.png")
    qwen_mask = out_guide.with_name(out_guide.stem + "_qwen_edge_mask.png")
    qwen_raw = out_guide.with_name(out_guide.stem + "_qwen_raw.png")
    if out_guide.exists() and qwen_raw.exists() and not force:
        _composite_seed_guide(qwen_raw, prepared_frame, out_guide)
        return out_guide
    if out_guide.exists() and not force:
        return out_guide
    cv2.imwrite(str(qwen_in), prepared_frame)
    save_edge_mask_for_frame(prepared_frame, qwen_mask)
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "edit_reference_image.py"),
        "--source-image", str(qwen_in),
        "--mask", str(qwen_mask),
        "--output", str(qwen_raw),
        "--workflow", qwen_args.get("masked_workflow") or qwen_args["workflow"],
        "--comfy-url", qwen_args["comfy_url"],
        "--comfy-dir", qwen_args["comfy_dir"],
        "--comfy-output-root", qwen_args["comfy_output_root"],
        "--model-backend", qwen_args.get("model_backend", "gguf"),
        "--gguf-model", qwen_args["gguf_model"],
        "--instruction", qwen_args.get("prompt", DEFAULT_SEED_PROMPT),
        "--load-image-node-id", qwen_args.get("load_image_node_id", "auto"),
        "--save-node-id", qwen_args.get("save_node_id", "auto"),
        "--no-normalize-to-source-size",
        "--force",
    ]
    print(f"  Qwen seed guide -> {out_guide.name}", flush=True)
    subprocess.run(cmd, check=True)
    if not qwen_raw.exists():
        print(f"  Warning: Qwen produced no output for {out_guide.name}; skipping this seed.", flush=True)
        return None
    _composite_seed_guide(qwen_raw, prepared_frame, out_guide)
    return out_guide


def seed_guides(
    prepared: Path,
    ranges: list[tuple[int, int, int]],
    manifest_stem: str,
    qwen_args: dict,
    sample_seconds: float = 0.0,
    shot_threshold: float | None = None,
    min_shot_seconds: float | None = None,
    start_strength: float = 0.7,
    mid_strength: float = 1.0,
    force: bool = False,
    occupied_frame_idxs: dict[int, set[int]] | None = None,
) -> dict[int, list[dict]]:
    """Generate Qwen seed guides at every shot change and map them to chunks.

    Returns {chunk_index: [{frame_idx, strength, image, seed:True}, ...]} for chunks that
    contain a shot boundary. A boundary at a chunk's first frame becomes a frame_idx 0 (i2v
    start) guide; interior boundaries become mid-chunk LTXVAddGuideAdvanced anchors.
    """
    boundaries = detect_shot_start_frames(prepared, sample_seconds, shot_threshold, min_shot_seconds)
    print(f"Seed guides: detected {len(boundaries)} shot start(s) in the prepared canvas.", flush=True)
    out_dir = DATA_ROOT / "intermediate" / "outpaint_seed_guides" / manifest_stem
    cap = cv2.VideoCapture(str(prepared))
    cache: dict[int, Path | None] = {}
    result: dict[int, list[dict]] = {}
    try:
        for chunk_index, start, end in ranges:
            for boundary in boundaries:
                if not (start <= boundary < end):
                    continue
                frame_idx = boundary - start
                if occupied_frame_idxs and int(frame_idx) in occupied_frame_idxs.get(chunk_index, set()):
                    continue
                if boundary not in cache:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(boundary))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        cache[boundary] = None
                        continue
                    guide_frame = resize_frame_for_qwen(frame, prepared)
                    cache[boundary] = generate_seed_guide(guide_frame, qwen_args, out_dir / f"shot_{boundary:06d}.png", force)
                guide_path = cache[boundary]
                if guide_path is None:
                    continue
                result.setdefault(chunk_index, []).append({
                    "frame_idx": int(frame_idx),
                    "strength": float(start_strength if frame_idx == 0 else mid_strength),
                    "image": root_relative(guide_path),
                    "seed": True,
                })
    finally:
        cap.release()
    seeded_chunks = ", ".join(str(c) for c in sorted(result))
    print(f"Seed guides: added anchors to chunk(s) [{seeded_chunks or 'none'}].", flush=True)
    return result
