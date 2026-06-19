from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from comfy_api import ensure_node_types, extract_output_files, object_info, queue_prompt, wait_for_comfy, wait_for_prompt
from common import ROOT, copy_to_comfy_input, file_fingerprint, find_ffmpeg, load_local_config, newest_output as newest_comfy_output, replace_unless_identical, replace_with_retry, resolve_path, root_relative, safe_stem, resumable_output, split_matches_source, video_info, write_signature, write_split_sidecar
from common import DATA_ROOT
from common import CACHE_ROOT


config = load_local_config()


_LOG_LOCK = threading.Lock()
# Suppress ffmpeg's banner + per-frame stats by default (they drowned out the useful log lines);
# set ARP_LOG_FFMPEG=1 to restore the full ffmpeg output for debugging.
_FFMPEG_VERBOSE = os.environ.get("ARP_LOG_FFMPEG", "").strip().lower() in ("1", "true", "yes")


def log(message: str) -> None:
    """Thread-safe stdout line — parallel chunk workers log concurrently."""
    with _LOG_LOCK:
        print(message, flush=True)


def run_ffmpeg(command: list[str]) -> None:
    """Run an ffmpeg command quietly (banner/stats hidden) unless ARP_LOG_FFMPEG=1.
    Errors are still printed (loglevel error) and a non-zero exit still raises (check=True)."""
    if command and not _FFMPEG_VERBOSE and "-loglevel" not in command:
        command = [command[0], "-hide_banner", "-loglevel", "error", "-nostats", *command[1:]]
    subprocess.run(command, check=True)


def gpu_label(comfy_url: str) -> str:
    """Human label for a ComfyUI instance, e.g. 'GPU2 :8190' (instance 0 = port 8188)."""
    try:
        port = urlparse(comfy_url).port or 0
    except ValueError:
        port = 0
    if port >= 8188:
        return f"GPU{port - 8188} :{port}"
    return comfy_url


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _ws_progress_listener(comfy_url: str, client_id: str, prompt_id: str, label: str,
                          node_titles: dict[str, str], stop_event: threading.Event) -> None:
    """Best-effort: stream ComfyUI execution progress for our prompt into the log via the
    /ws websocket. Any failure (no aiohttp, connection error) silently disables it, so the
    upscale run is never affected."""
    try:
        import asyncio

        import aiohttp
    except Exception:
        return
    ws_url = comfy_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + f"/ws?clientId={client_id}"

    async def _run() -> None:
        last_node = ""
        last_pct = -1
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, heartbeat=20) as ws:
                while not stop_event.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    try:
                        event = json.loads(msg.data)
                    except Exception:
                        continue
                    data = event.get("data") or {}
                    if data.get("prompt_id") not in (None, prompt_id):
                        continue
                    etype = event.get("type")
                    if etype == "executing":
                        node = data.get("node")
                        if node is None:
                            break  # this client's prompt finished executing
                        name = node_titles.get(str(node), str(node))
                        if name != last_node:
                            last_node, last_pct = name, -1
                            log(f"  {label}  {name} …")
                    elif etype == "progress":
                        value = int(data.get("value") or 0)
                        maximum = int(data.get("max") or 0)
                        if maximum:
                            pct = value * 100 // maximum
                            if pct >= last_pct + 10 or value >= maximum:
                                last_pct = pct
                                name = node_titles.get(str(data.get("node")), last_node or "")
                                log(f"  {label}  {name} step {value}/{maximum}")
                    elif etype in ("execution_success", "execution_error", "execution_interrupted"):
                        break

    try:
        asyncio.run(_run())
    except Exception:
        pass


def default_output(source: Path, width: int, height: int) -> Path:
    suffix = f"flashvsr_{width}x{height}" if width and height else "flashvsr"
    return DATA_ROOT / "output" / "upscaled" / f"{safe_stem(source.name)}_{suffix}.mp4"


# The legacy single-node workflow hardcoded these values, so leaving a knob at its
# default produces the same pixels as before the advanced node was adopted.
ADVANCED_DEFAULTS = {
    "flashvsr_color_fix": True,
    "flashvsr_tile_size": 256,
    "flashvsr_tile_overlap": 24,
    "flashvsr_sparse_ratio": 2.0,
    "flashvsr_kv_ratio": 3.0,
    "flashvsr_local_range": 11,
}


def signature(args: argparse.Namespace, source: Path, output_width: int, output_height: int) -> dict[str, Any]:
    sig = {
        "version": 5,
        "tool": "upscale_video.py",
        "method": "flashvsr_ultra_fast",
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "target_width": output_width,
        "target_height": output_height,
        "comfy_dir": root_relative(resolve_path(args.comfy_dir)),
        "comfy_url": args.comfy_url,
        "flashvsr_model": args.flashvsr_model,
        "flashvsr_mode": args.flashvsr_mode,
        "flashvsr_scale": args.flashvsr_scale,
        "flashvsr_tiled_vae": args.flashvsr_tiled_vae,
        "flashvsr_tiled_dit": args.flashvsr_tiled_dit,
        "flashvsr_unload_dit": args.flashvsr_unload_dit,
        "flashvsr_seed": args.flashvsr_seed,
        "chunk_seconds": args.chunk_seconds,
        "overlap_frames": args.overlap_frames,
        "fps": args.fps,
    }
    # Only record advanced knobs that differ from the legacy behaviour so outputs and
    # cached chunks rendered before these flags existed remain reusable.
    for name, default in ADVANCED_DEFAULTS.items():
        value = getattr(args, name)
        if value != default:
            sig[name] = value
    return sig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upscale an ARP video.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--target-width", type=int, default=3840)
    parser.add_argument("--target-height", type=int, default=2160)
    parser.add_argument("--comfy-dir", default=config.get("comfy_dir", str(ROOT / "tools" / "comfyui")))
    parser.add_argument("--comfy-url", default=config.get("comfy_url", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-urls", default=os.environ.get("ARP_COMFY_URLS", ""), help="Comma-separated ComfyUI URLs for multi-GPU parallel chunk upscaling. Defaults to the ARP_COMFY_URLS env var, else --comfy-url.")
    parser.add_argument("--comfy-output-root", default="")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--flashvsr-model", choices=["FlashVSR", "FlashVSR-v1.1"], default="FlashVSR-v1.1")
    parser.add_argument("--flashvsr-mode", choices=["tiny", "tiny-long", "full"], default="tiny")
    parser.add_argument("--flashvsr-scale", type=int, default=2)
    parser.add_argument("--flashvsr-tiled-vae", dest="flashvsr_tiled_vae", action="store_true", default=True)
    parser.add_argument("--no-flashvsr-tiled-vae", dest="flashvsr_tiled_vae", action="store_false")
    parser.add_argument("--flashvsr-tiled-dit", dest="flashvsr_tiled_dit", action="store_true", default=True)
    parser.add_argument("--no-flashvsr-tiled-dit", dest="flashvsr_tiled_dit", action="store_false")
    parser.add_argument("--flashvsr-unload-dit", dest="flashvsr_unload_dit", action="store_true", default=False)
    parser.add_argument("--no-flashvsr-unload-dit", dest="flashvsr_unload_dit", action="store_false")
    parser.add_argument("--flashvsr-color-fix", dest="flashvsr_color_fix", action="store_true", default=True, help="Wavelet color correction that matches output colors back to the source.")
    parser.add_argument("--no-flashvsr-color-fix", dest="flashvsr_color_fix", action="store_false")
    parser.add_argument("--flashvsr-tile-size", type=int, default=256, help="DiT tile edge in pixels when tiled DiT is on (multiple of 32, max 1024). Larger tiles give faces more surrounding context at the cost of VRAM.")
    parser.add_argument("--flashvsr-tile-overlap", type=int, default=24, help="Feathered overlap between DiT tiles in pixels; raise it if tile seams are visible.")
    parser.add_argument("--flashvsr-sparse-ratio", type=float, default=2.0, help="Sparse attention density, 1.5-2.0: 2.0 is most stable, 1.5 is faster.")
    parser.add_argument("--flashvsr-kv-ratio", type=float, default=3.0, help="Attention memory budget, 1.0-3.0: 3.0 is highest quality, lower saves VRAM.")
    parser.add_argument("--flashvsr-local-range", type=int, choices=[9, 11], default=11, help="Temporal attention window: 11 is more stable, 9 keeps more fine motion (lips) and detail.")
    parser.add_argument("--flashvsr-seed", type=int, default=0)
    parser.add_argument("--chunk-seconds", type=float, default=6.0, help="Upscale in chunks of roughly this many seconds. Use 0 to send the whole clip.")
    parser.add_argument("--overlap-frames", type=int, default=8, help="Frames repeated before each chunk, then trimmed before stitching.")
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def default_from_spec(spec: Any) -> Any:
    if isinstance(spec, (list, tuple)):
        if len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]:
            return spec[1]["default"]
        if spec and isinstance(spec[0], list) and spec[0]:
            return spec[0][0]
    return None


def flashvsr_node_inputs(class_type: str, info: dict[str, Any], values: dict[str, Any], connections: dict[str, list[Any]]) -> dict[str, Any]:
    inputs: dict[str, Any] = dict(connections)
    input_info = info.get(class_type, {}).get("input", {})
    accepted: set[str] = set()
    for group in ("required", "optional"):
        for name, spec in (input_info.get(group) or {}).items():
            accepted.add(name)
            if name in connections:
                continue
            value = default_from_spec(spec)
            if value is not None:
                inputs[name] = value

    # Use the Ultra Fast node's real input names so ARP logs and saved settings map
    # directly to the Comfy node.
    for name, value in values.items():
        if not accepted or name in accepted:
            inputs[name] = value
    return inputs


def flashvsr_prompt(video_name: str, fps: float, args: argparse.Namespace, prefix: str, info: dict[str, Any]) -> dict[str, Any]:
    init_values = {
        "model": args.flashvsr_model,
        "mode": args.flashvsr_mode,
        # The basic FlashVSRNode ran fp16; keep it so existing renders stay reproducible.
        "precision": "fp16",
    }
    adv_values = {
        "scale": args.flashvsr_scale,
        "color_fix": bool(args.flashvsr_color_fix),
        "tiled_vae": bool(args.flashvsr_tiled_vae),
        "tiled_dit": bool(args.flashvsr_tiled_dit),
        "tile_size": args.flashvsr_tile_size,
        "tile_overlap": args.flashvsr_tile_overlap,
        "unload_dit": bool(args.flashvsr_unload_dit),
        "sparse_ratio": args.flashvsr_sparse_ratio,
        "kv_ratio": args.flashvsr_kv_ratio,
        "local_range": args.flashvsr_local_range,
        "seed": args.flashvsr_seed,
    }
    return {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": 0.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "2": {"class_type": "FlashVSRInitPipe", "inputs": flashvsr_node_inputs("FlashVSRInitPipe", info, init_values, {})},
        "3": {"class_type": "FlashVSRNodeAdv", "inputs": flashvsr_node_inputs("FlashVSRNodeAdv", info, adv_values, {"pipe": ["2", 0], "frames": ["1", 0]})},
        "4": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["3", 0],
                "audio": ["1", 2],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 16,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def flashvsr_run(args: argparse.Namespace, source: Path, partial: Path, output_width: int, output_height: int, comfy_url: str | None = None) -> Path:
    comfy_url = comfy_url or args.comfy_url
    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    if not (comfy_dir / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI main.py not found: {comfy_dir / 'main.py'}")
    wait_for_comfy(comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    required_nodes = {
        "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
        "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
        "FlashVSRInitPipe": "ComfyUI-FlashVSR_Ultra_Fast",
        "FlashVSRNodeAdv": "ComfyUI-FlashVSR_Ultra_Fast",
    }
    ensure_node_types(comfy_url, required_nodes, "FlashVSR upscaling")
    info = object_info(comfy_url)
    video_name = copy_to_comfy_input(source, comfy_dir, "arp_upscale")
    fps = args.fps or float(video_info(source)["fps"])
    prefix = f"arp_upscale/{safe_stem(source.name)}_flashvsr_{output_width}x{output_height}"
    prompt = flashvsr_prompt(video_name, fps, args, prefix, info)
    client_id = str(uuid.uuid4())
    node_titles = {node_id: node.get("class_type", node_id) for node_id, node in prompt.items()}
    label = gpu_label(comfy_url)
    prompt_id = queue_prompt(comfy_url, prompt, client_id=client_id)
    log(f"  {label} queued prompt {prompt_id}")
    stop_event = threading.Event()
    listener = threading.Thread(
        target=_ws_progress_listener,
        args=(comfy_url, client_id, prompt_id, label, node_titles, stop_event),
        daemon=True,
    )
    listener.start()
    try:
        history = wait_for_prompt(comfy_url, prompt_id, args.poll_seconds)
    finally:
        stop_event.set()
        listener.join(timeout=3)
    produced = newest_comfy_output(extract_output_files(history, comfy_output_root), {".mp4", ".mov", ".mkv", ".webm"}, "FlashVSR video")
    partial.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, partial)
    return partial


def chunk_ranges(total_frames: int, fps: float, chunk_seconds: float, overlap_frames: int) -> list[tuple[int, int, int]]:
    if total_frames <= 0 or fps <= 0 or chunk_seconds <= 0:
        return [(0, 0, total_frames)]
    chunk_frames = max(1, int(round(chunk_seconds * fps)))
    if chunk_frames >= total_frames:
        return [(0, 0, total_frames)]
    overlap = max(0, min(int(overlap_frames), chunk_frames - 1))
    ranges: list[tuple[int, int, int]] = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + chunk_frames)
        source_start = max(0, start - overlap)
        trim_start = start - source_start
        ranges.append((source_start, end, trim_start))
        if end >= total_frames:
            break
        start = end
    return ranges


def split_video_chunk(ffmpeg: str, source: Path, target: Path, start_frame: int, end_frame: int, fps: float, force: bool, source_fingerprint: dict[str, Any]) -> None:
    if target.exists() and not force and split_matches_source(target, source_fingerprint):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial" + target.suffix)
    vf = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1"
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        vf,
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    run_ffmpeg(command)
    replace_unless_identical(partial, target, f"Upscale prepared chunk {target.name}")
    write_split_sidecar(target, source, source_fingerprint)


def normalize_chunk(ffmpeg: str, source: Path, target: Path, width: int, height: int, trim_start: int, fps: float, force: bool) -> None:
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial" + target.suffix)
    filters = []
    if trim_start > 0:
        filters.append(f"trim=start_frame={trim_start}")
    filters.append(f"setpts=N/({fps:.8f}*TB)")
    filters.append(f"fps={fps:.8f}")
    filters.append(f"scale={width}:{height}:flags=lanczos,setsar=1")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        ",".join(filters),
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    run_ffmpeg(command)
    replace_with_retry(partial, target, f"Upscale normalized chunk {target.name}")


def stitch_chunks(ffmpeg: str, chunks: list[Path], source: Path, output: Path) -> None:
    if not chunks:
        raise RuntimeError("No upscale chunks were produced.")
    output.parent.mkdir(parents=True, exist_ok=True)
    video_partial = output.with_suffix(output.suffix + ".video.partial" + output.suffix)
    final_partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    with tempfile.TemporaryDirectory(prefix="arp_upscale_concat_") as tmp_text:
        list_file = Path(tmp_text) / "chunks.txt"
        list_file.write_text("".join(f"file '{chunk.as_posix()}'\n" for chunk in chunks), encoding="utf-8")
        log(f"[upscale] Stitching {len(chunks)} chunk(s) + muxing original audio")
        run_ffmpeg([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(video_partial)])
    mux_command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_partial),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        str(final_partial),
    ]
    run_ffmpeg(mux_command)
    video_partial.unlink(missing_ok=True)
    replace_with_retry(final_partial, output, "Upscaled output")


def ensure_flashvsr_model(args: argparse.Namespace) -> None:
    """Download the FlashVSR weights once, serially, before any parallel ComfyUI init.

    The vendored FlashVSR node's own downloader (``model_downlod``) guards on the *model
    directory* existing, not on the weight files. When several ComfyUI instances run
    ``FlashVSRInitPipe`` in parallel, the first creates the directory and starts a large
    download while the others see the directory already present, skip the download, and
    then fail with ``"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist``.
    A single idempotent ``snapshot_download`` here guarantees the weights are complete
    before chunks fan out across instances (and recovers a partial leftover dir).
    HF_HUB_DISABLE_XET=1 (set by the entrypoint) keeps this on plain HTTPS for the volume.
    """
    model = getattr(args, "flashvsr_model", "FlashVSR-v1.1")
    model_dir = resolve_path(args.comfy_dir) / "models" / model
    if (model_dir / "diffusion_pytorch_model_streaming_dmd.safetensors").exists():
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return  # Fall back to the node's own downloader (fine on the single-instance path).
    log(f"[upscale] Ensuring FlashVSR model '{model}' is present (one-time download)…")
    snapshot_download(repo_id=f"JunhaoZhuang/{model}", local_dir=str(model_dir),
                      local_dir_use_symlinks=False, resume_download=True)


def chunked_flashvsr_run(args: argparse.Namespace, source: Path, output: Path, output_width: int, output_height: int, info: dict[str, Any], source_fingerprint: dict[str, Any]) -> None:
    ffmpeg = find_ffmpeg(args.ffmpeg)
    fps = args.fps or float(info["fps"])
    ranges = chunk_ranges(int(info["frames"]), fps, args.chunk_seconds, args.overlap_frames)
    if len(ranges) <= 1:
        raw_partial = output.with_suffix(output.suffix + ".flashvsr.partial" + output.suffix)
        final_partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
        for path in (raw_partial, final_partial):
            if path.exists():
                path.unlink()
        log(f"[upscale] Single clip (no chunking) → {gpu_label(args.comfy_url)}")
        flashvsr_run(args, source, raw_partial, output_width, output_height)
        if not raw_partial.exists():
            raise RuntimeError(f"FlashVSR finished but did not create expected output: {raw_partial}")
        raw_info = video_info(raw_partial)
        if raw_info["width"] == output_width and raw_info["height"] == output_height:
            replace_with_retry(raw_partial, final_partial, "Upscaled preview")
        else:
            scale_video(ffmpeg, raw_partial, final_partial, output_width, output_height)
            raw_partial.unlink(missing_ok=True)
        if output.exists():
            output.unlink()
        replace_with_retry(final_partial, output, "Upscaled output")
        return

    chunk_dir = CACHE_ROOT / "upscale_chunks" / f"{safe_stem(source.name)}_flashvsr_{output_width}x{output_height}_{int(args.chunk_seconds * 1000)}ms_ov{max(0, args.overlap_frames)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # One ComfyUI URL per GPU (multi-GPU pods). Chunks are independent, so they are dispatched
    # concurrently across the instances; a URL queue guarantees at most one in-flight job per
    # instance. A single URL behaves exactly like the old sequential loop.
    urls = [u.strip() for u in (getattr(args, "comfy_urls", "") or "").split(",") if u.strip()] or [args.comfy_url]
    workers = max(1, len(urls))
    total = len(ranges)
    log(f"[upscale] {total} chunk(s) · {args.chunk_seconds:g}s each · {max(0, args.overlap_frames)} overlap frame(s) · {workers} GPU worker(s) → {output_width}×{output_height}")
    # Pre-download the FlashVSR weights once before fanning chunks out: the node's own
    # downloader races across parallel instances (it guards on the dir, not the files).
    if workers > 1:
        ensure_flashvsr_model(args)
    digits = max(4, int(math.log10(total)) + 1)

    url_pool: "queue.Queue[str]" = queue.Queue()
    for url in urls:
        url_pool.put(url)

    run_start = time.monotonic()
    progress_lock = threading.Lock()
    progress = {"done": 0}
    durations: list[float] = []

    def note_done(index: int, elapsed: float | None) -> None:
        """Log a chunk completion with a running counter and ETA from the rolling average."""
        with progress_lock:
            progress["done"] += 1
            done = progress["done"]
            if elapsed is not None:
                durations.append(elapsed)
            avg = (sum(durations) / len(durations)) if durations else 0.0
            eta = avg * (total - done) / max(1, workers)
            took = "reuse" if elapsed is None else fmt_duration(elapsed)
            eta_text = f" · ETA ~{fmt_duration(eta)}" if (avg and done < total) else ""
            log(f"[upscale] Chunk {index + 1:>{len(str(total))}}/{total} ✓ {took}  (Ø {fmt_duration(avg)} · {done}/{total} fertig{eta_text})")

    def process_chunk(index: int, start_frame: int, end_frame: int, trim_start: int) -> Path:
        chunk_input = chunk_dir / f"input_{index:0{digits}d}_{start_frame:06d}_{end_frame:06d}.mp4"
        chunk_raw = chunk_dir / f"raw_{index:0{digits}d}_{start_frame:06d}_{end_frame:06d}.mp4"
        chunk_final = chunk_dir / f"final_{index:0{digits}d}_{start_frame:06d}_{end_frame:06d}.mp4"
        split_video_chunk(ffmpeg, source, chunk_input, start_frame, end_frame, fps, args.force, source_fingerprint)
        chunk_sig = signature(args, chunk_input, output_width, output_height)
        if not args.force and resumable_output(chunk_final, chunk_sig, width=output_width, height=output_height):
            note_done(index, None)
            return chunk_final
        url = url_pool.get()
        chunk_start = time.monotonic()
        try:
            log(f"[upscale] Chunk {index + 1:>{len(str(total))}}/{total} → {gpu_label(url)}  Frames {start_frame}–{end_frame}")
            flashvsr_run(args, chunk_input, chunk_raw, output_width, output_height, comfy_url=url)
        finally:
            url_pool.put(url)
        normalize_chunk(ffmpeg, chunk_raw, chunk_final, output_width, output_height, trim_start, fps, True)
        write_signature(chunk_final, chunk_sig)
        chunk_raw.unlink(missing_ok=True)
        note_done(index, time.monotonic() - chunk_start)
        return chunk_final

    results: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_chunk, index, start_frame, end_frame, trim_start): index
            for index, (start_frame, end_frame, trim_start) in enumerate(ranges)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    normalized_chunks = [results[index] for index in range(len(ranges))]
    if output.exists():
        output.unlink()
    stitch_chunks(ffmpeg, normalized_chunks, source, output)

    total_elapsed = time.monotonic() - run_start
    avg = (sum(durations) / len(durations)) if durations else 0.0
    try:
        size_mb = output.stat().st_size / (1024 * 1024)
    except OSError:
        size_mb = 0.0
    log(f"[upscale] Fertig: {total} Chunks in {fmt_duration(total_elapsed)} (Ø {fmt_duration(avg)}/Chunk) · "
        f"{workers} GPU(s) · {output_width}×{output_height} · {size_mb:.0f} MB → {output}")


def fit_dimensions(source_width: int, source_height: int, target_width: int, target_height: int) -> tuple[int, int]:
    if target_width <= 0 and target_height <= 0:
        return source_width * 4, source_height * 4
    if target_width <= 0:
        target_width = round(target_height * source_width / source_height)
    if target_height <= 0:
        target_height = round(target_width * source_height / source_width)
    return max(2, target_width // 2 * 2), max(2, target_height // 2 * 2)


def scale_video(ffmpeg: str, source: Path, output: Path, width: int, height: int) -> None:
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:{height}:flags=lanczos,setsar=1",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    run_ffmpeg(command)


def run(args: argparse.Namespace) -> int:
    source = resolve_path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"Input video not found for upscaling: {source}")

    info = video_info(source)
    output_width, output_height = fit_dimensions(int(info["width"]), int(info["height"]), args.target_width, args.target_height)
    output = resolve_path(args.output) if args.output else default_output(source, output_width, output_height)
    sig = signature(args, source, output_width, output_height)

    if not args.force and resumable_output(output, sig, video_like=source, width=output_width, height=output_height):
        print(f"Reuse upscaled video: {output}", flush=True)
        return 0
    if args.dry_run:
        print(f"Would upscale {source} -> {output} using FlashVSR in ComfyUI at {args.comfy_url} ({args.comfy_dir})", flush=True)
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    chunked_flashvsr_run(args, source, output, output_width, output_height, info, sig["source_fingerprint"])
    write_signature(output, sig)
    print(f"Wrote upscaled video: {output}", flush=True)
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
