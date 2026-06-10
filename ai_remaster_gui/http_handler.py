from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen


def bind_context(context: dict) -> None:
    globals().update(context)


ALLOWED_HOST_NAMES = {"127.0.0.1", "localhost", "[::1]"}


class Handler(BaseHTTPRequestHandler):
    server_version = "AIRemasterGUI/1.0"

    def host_allowed(self) -> bool:
        """Only answer requests addressed to localhost.

        The server binds to 127.0.0.1, but a malicious web page can still reach it via DNS
        rebinding (a hostname that resolves to 127.0.0.1 bypasses the browser's same-origin
        protections). Rejecting foreign Host headers closes that off.
        """
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):
            name = host.split("]", 1)[0] + "]"
        else:
            name = host.rsplit(":", 1)[0] if ":" in host else host
        return name in ALLOWED_HOST_NAMES

    def logfile_allowed(self, path: Path) -> bool:
        """The log viewer may only read log-like files from ARP itself or the ComfyUI install."""
        if path.suffix.lower() not in {".log", ".txt"}:
            return False
        config = current_config()
        allowed_roots = [ROOT, Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui")))]
        resolved = path.resolve()
        for root in allowed_roots:
            try:
                resolved.relative_to(root.resolve())
                return True
            except (ValueError, OSError):
                continue
        return False

    def do_GET(self) -> None:  # noqa: N802
        if not self.host_allowed():
            self.send_error(403)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_static(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path.startswith("/static/"):
            static_path = STATIC_DIR / unquote(parsed.path.removeprefix("/static/"))
            try:
                static_path.resolve().relative_to(STATIC_DIR.resolve())
            except ValueError:
                self.send_error(404)
                return
            self.send_static(static_path)
        elif parsed.path == "/site.webmanifest":
            self.send_json(
                {
                    "name": "ARP - AI Remaster Pipeline",
                    "short_name": "ARP",
                    "start_url": "/",
                    "display": "standalone",
                    "background_color": "#101316",
                    "theme_color": "#2d8f7d",
                    "icons": [
                        {"src": "/media?path=assets/branding/arp-app-icon-192.png", "sizes": "192x192", "type": "image/png"},
                        {"src": "/media?path=assets/branding/arp-app-icon-512.png", "sizes": "512x512", "type": "image/png"},
                    ],
                }
            )
        elif parsed.path == "/api/state":
            view = parse_qs(parsed.query).get("active", [""])[0]
            self.send_json(APP.state(view))
        elif parsed.path == "/api/command":
            stage = parse_qs(parsed.query).get("stage", [""])[0]
            self.send_json({"command": APP.command_for(stage) if stage else []})
        elif parsed.path == "/api/existing-outputs":
            stage = parse_qs(parsed.query).get("stage", [""])[0]
            self.send_json({"paths": APP.existing_outputs(stage) if stage else []})
        elif parsed.path == "/api/media-status":
            query = parse_qs(parsed.query)
            path_text = query.get("path", [""])[0]
            path = resolve(path_text)
            exists = path.is_file()
            self.send_json(
                {
                    "ok": True,
                    "path": path_text,
                    "exists": exists,
                    "mtime": path.stat().st_mtime if exists else 0,
                    "running": APP.process is not None and APP.process.poll() is None,
                    "running_stage": APP.running_stage,
                    "log_count": len(APP.log),
                }
            )
        elif parsed.path == "/api/comfy":
            url = parse_qs(parsed.query).get("url", ["http://127.0.0.1:8188"])[0].rstrip("/")
            try:
                with urlopen(url + "/queue", timeout=3) as response:
                    self.send_json({"ok": True, "queue": json.loads(response.read().decode("utf-8"))})
            except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/logfile":
            path = resolve(parse_qs(parsed.query).get("path", [""])[0])
            if not self.logfile_allowed(path):
                self.send_json({"text": "", "error": "Log files can only be read from the ARP folder or the configured ComfyUI folder (.log/.txt)."})
                return
            text = path.read_text(encoding="utf-8", errors="replace")[-12000:] if path.exists() else ""
            self.send_json({"text": text})
        elif parsed.path == "/api/openai-models":
            token = APP.settings.get("references", {}).get("openai_api_key", "").strip()
            if not token:
                self.send_json({"ok": False, "error": "Add your OpenAI API key first."})
                return
            try:
                request = Request("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {token}"})
                with urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                models = sorted(
                    item.get("id", "")
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and model_looks_image_capable(str(item.get("id", "")))
                )
                self.send_json({"ok": True, "models": models})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-preview":
            query = parse_qs(parsed.query)
            try:
                path = preview_reference_frame(query.get("manifest", [""])[0], int(query.get("index", ["0"])[0]), float(query.get("time", ["0"])[0]))
                self.send_json({"ok": True, "path": path})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/aspect-preview":
            query = parse_qs(parsed.query)
            try:
                path = aspect_preview_at_for_settings(APP.settings, float(query.get("time", ["0"])[0]))
                self.send_json({"ok": True, "path": path})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-auto-crop":
            query = parse_qs(parsed.query)
            try:
                result = auto_crop_for_settings(APP.settings, float(query.get("time", ["0"])[0]))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Auto Crop failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-chunk-preview":
            query = parse_qs(parsed.query)
            try:
                preview = outpaint_chunk_preview(
                    APP.settings,
                    int(query.get("chunk_index", ["0"])[0]),
                    query.get("kind", ["source"])[0],
                    query.get("position", ["middle"])[0],
                )
                self.send_json({"ok": True, "preview": preview})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-guide-preview":
            query = parse_qs(parsed.query)
            try:
                chunk_index = int(query.get("chunk_index", ["0"])[0])
                frame_idx = int(query.get("frame_idx", ["0"])[0])
                source_text = pipeline_source_text(APP.settings)
                if not source_text:
                    self.send_json({"ok": False, "error": "No source material"})
                else:
                    chunks_state = outpaint_chunks_state(APP.settings)
                    rows = chunks_state.get("rows", [])
                    row = next((r for r in rows if r.get("index") == chunk_index), None)
                    if row is None:
                        self.send_json({"ok": False, "error": "Chunk not found"})
                    else:
                        range_source = ensure_outpaint_prepared_canvas(source_text, APP.settings.get("outpaint", {}))
                        fps = float(row.get("fps", 24) or 24)
                        secs = _guide_source_seconds(row, frame_idx, fps)
                        cache_key = f"gfprev_{chunk_index}_{frame_idx}"
                        preview = chunk_frame_preview(range_source, secs, cache_key)
                        self.send_json({"ok": True, "preview": preview})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/media":
            query = parse_qs(parsed.query)
            path = resolve(unquote(query.get("path", [""])[0]))
            if "clip_start" in query or "clip_end" in query:
                try:
                    start = float(query.get("clip_start", ["0"])[0])
                    end = float(query.get("clip_end", [str(start + 0.041)])[0])
                    path = media_clip_path(path, start, end, query.get("clip_key", [""])[0])
                except FileNotFoundError:
                    self.send_error(404)
                    return
                except Exception as exc:
                    APP.log.append(f"Shot video preview failed: {exc}")
                    self.send_error(404)
                    return
            self.send_media(path)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if not self.host_allowed():
            self.send_error(403)
            return
        parsed = urlparse(self.path)
        data = self.read_json()
        if parsed.path == "/api/settings":
            APP.update_settings(str(data.get("stage", "")), data.get("values", {}))
            self.send_json({"ok": True})
        elif parsed.path == "/api/run":
            if data.get("all"):
                ok, message = APP.run_all()
            else:
                ok, message = APP.run_stage(str(data.get("stage", "")))
            self.send_json({"ok": ok, "message": message})
        elif parsed.path == "/api/upscale-preview":
            ok, message = APP.run_upscale_preview()
            self.send_json({"ok": ok, "message": message})
        elif parsed.path == "/api/stop":
            APP.stop()
            self.send_json({"ok": True})
        elif parsed.path == "/api/quit":
            # Kill any running stage, answer, then stop the server so the launching shell
            # returns to its prompt with nothing left running. Answer before shutdown so
            # this response still reaches the browser.
            APP.stop_for_quit()
            self.send_json({"ok": True})
            request_quit(self.server)
        elif parsed.path == "/api/shot-scrub":
            try:
                result = extract_reference_frame(str(data.get("manifest", "")), int(data.get("index", 0)), float(data.get("time", 0)))
                self.send_json({"ok": True, **result})
            except Exception as exc:
                APP.log.append(f"Shot scrub failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-prompt":
            try:
                update_manifest_row(resolve(str(data.get("manifest", ""))), int(data.get("index", 0)), {"prompt": str(data.get("prompt", ""))})
                self.send_json({"ok": True})
            except Exception as exc:
                APP.log.append(f"Shot prompt save failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-enabled":
            try:
                enabled = "true" if data.get("enabled") else "false"
                update_manifest_row(resolve(str(data.get("manifest", ""))), int(data.get("index", 0)), {"enabled": enabled})
                self.send_json({"ok": True})
            except Exception as exc:
                APP.log.append(f"Shot enabled save failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-merge":
            try:
                result = merge_manifest_shots(str(data.get("manifest", "")), int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("shots")})
            except Exception as exc:
                APP.log.append(f"Shot merge failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-split":
            try:
                result = split_manifest_shot(str(data.get("manifest", "")), int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("shots")})
            except Exception as exc:
                APP.log.append(f"Shot split failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-boundary":
            try:
                result = update_shot_boundary(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("edge", "")), float(data.get("time", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("shots")})
            except Exception as exc:
                APP.log.append(f"Shot boundary update failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-fade":
            try:
                result = update_shot_fade(str(data.get("manifest", "")), int(data.get("index", 0)), bool(data.get("enabled")), str(data.get("crossfade_seconds", "")))
                self.send_json({"ok": True, **result, "state": APP.state("shots")})
            except Exception as exc:
                APP.log.append(f"Shot fade update failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/reference-regenerate":
            try:
                ok, message = APP.run_reference_regeneration(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("provider", "qwen")))
                self.send_json({"ok": ok, "message": message, "state": APP.state() if ok else None, "error": "" if ok else message})
            except Exception as exc:
                APP.log.append(f"Reference regeneration failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/reference-delete":
            try:
                result = delete_color_reference(str(data.get("manifest", "")), int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Reference delete failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/reference-custom":
            try:
                result = install_custom_color_reference(str(data.get("manifest", "")), int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Custom reference install failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/reference-mask-sam":
            try:
                result = sam_reference_mask(
                    str(data.get("manifest", "")),
                    int(data.get("index", 0)),
                    data.get("points", []) if isinstance(data.get("points", []), list) else [],
                    int(data.get("width", 1)),
                    int(data.get("height", 1)),
                    int(data.get("tolerance", 10)),
                )
                self.send_json({"ok": True, **result})
            except Exception as exc:
                APP.log.append(f"Reference smart mask failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/guide-frame-mask-sam":
            try:
                result = sam_guide_mask(
                    int(data.get("chunk_index", 0)),
                    int(data.get("guide_index", 0)),
                    data.get("points", []) if isinstance(data.get("points", []), list) else [],
                    int(data.get("width", 1)),
                    int(data.get("height", 1)),
                    str(data.get("fallback_path", "")),
                )
                self.send_json({"ok": True, **result})
            except Exception as exc:
                APP.log.append(f"Guide smart mask failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/reference-edit-preview":
            try:
                ok, message, output = APP.run_reference_edit_preview(
                    str(data.get("manifest", "")),
                    int(data.get("index", 0)),
                    str(data.get("instruction", "")),
                    str(data.get("mask", "")),
                    str(data.get("sampled_color", "")),
                )
                self.send_json({"ok": ok, "message": message, "preview": output, "state": APP.state("references") if ok else None, "error": "" if ok else message})
            except Exception as exc:
                APP.log.append(f"Reference edit preview failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/reference-edit-accept":
            try:
                result = accept_reference_edit(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("preview", "")))
                self.send_json({"ok": True, **result, "state": APP.state("references")})
            except Exception as exc:
                APP.log.append(f"Reference edit accept failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/reference-edit-revert":
            try:
                result = revert_reference_edit(str(data.get("manifest", "")), int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("references")})
            except Exception as exc:
                APP.log.append(f"Reference edit revert failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/export-media":
            try:
                result = export_media_file(str(data.get("path", "")))
                self.send_json({"ok": True, **result})
            except Exception as exc:
                APP.log.append(f"Media export failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-chunk":
            try:
                update_outpaint_chunk(int(data.get("index", 0)), str(data.get("seed", "")), str(data.get("prompt_suffix", "")), str(data.get("custom_seconds", "")), str(data.get("negative_suffix", "")), str(data.get("guide_strength", "")), str(data.get("guide_end_strength", "")), data.get("custom_length", None), str(data.get("offset_x", "0")), str(data.get("offset_y", "0")))
                self.send_json({"ok": True, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Outpaint chunk save failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-chunk-regenerate":
            try:
                update_outpaint_chunk(int(data.get("index", 0)), str(data.get("seed", "")), str(data.get("prompt_suffix", "")), str(data.get("custom_seconds", "")), str(data.get("negative_suffix", "")), str(data.get("guide_strength", "")), str(data.get("guide_end_strength", "")), data.get("custom_length", None), str(data.get("offset_x", "0")), str(data.get("offset_y", "0")))
                ok, message = APP.run_outpaint_chunk(int(data.get("index", 0)))
                self.send_json({"ok": ok, "message": message, "state": APP.state("outpaint") if ok else None, "error": "" if ok else message})
            except Exception as exc:
                APP.log.append(f"Outpaint chunk regeneration failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-anchor":
            try:
                result = install_outpaint_guide(int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Outpaint guide install failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-anchor-clear":
            try:
                result = clear_outpaint_guide(int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Outpaint guide clear failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-anchor-generate":
            try:
                ok, message = APP.run_outpaint_guide_generation(
                    int(data.get("index", 0)),
                    str(data.get("prompt", "")),
                )
                self.send_json({"ok": ok, "message": message, "state": APP.state("outpaint") if ok else None, "error": "" if ok else message})
            except Exception as exc:
                APP.log.append(f"Outpaint guide generation failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-end-anchor":
            try:
                result = install_outpaint_end_guide(int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Outpaint end guide install failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-end-anchor-clear":
            try:
                result = clear_outpaint_end_guide(int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Outpaint end guide clear failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-end-anchor-generate":
            try:
                ok, message = APP.run_outpaint_end_guide_generation(
                    int(data.get("index", 0)),
                    str(data.get("prompt", "")),
                )
                self.send_json({"ok": ok, "message": message, "state": APP.state("outpaint") if ok else None, "error": "" if ok else message})
            except Exception as exc:
                APP.log.append(f"Outpaint end guide generation failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/guide-frame-add":
            try:
                result = add_guide_frame(int(data.get("chunk_index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Guide frame add failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/guide-frame-remove":
            try:
                result = remove_guide_frame(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Guide frame remove failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/guide-frame-save":
            try:
                result = save_guide_frame(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)), int(data.get("frame_idx", 0)), float(data.get("strength", 0.7)))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Guide frame save failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/guide-frame-upload":
            try:
                result = upload_guide_frame_image(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Guide frame upload failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/guide-frame-clear":
            try:
                result = clear_guide_frame_image(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Guide frame clear failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/guide-frame-generate":
            try:
                ok, message = APP.run_guide_frame_generation(
                    int(data.get("chunk_index", 0)),
                    int(data.get("guide_index", 0)),
                    int(data.get("frame_idx", 0)),
                    str(data.get("prompt", "")),
                )
                self.send_json({"ok": ok, "message": message, "state": APP.state("outpaint") if ok else None, "error": "" if ok else message})
            except Exception as exc:
                APP.log.append(f"Guide frame generation failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/guide-frame-edit-preview":
            try:
                ok, message, preview = APP.run_guide_edit_preview(
                    int(data.get("chunk_index", 0)),
                    int(data.get("guide_index", 0)),
                    str(data.get("instruction", "")),
                    str(data.get("mask", "")),
                    str(data.get("sampled_color", "")),
                )
                self.send_json({"ok": ok, "message": message, "preview": preview, "state": APP.state("outpaint") if ok else None, "error": "" if ok else message})
            except Exception as exc:
                APP.log.append(f"Guide frame edit preview failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/guide-frame-edit-accept":
            try:
                result = accept_guide_edit(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)), str(data.get("preview", "")))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Guide frame edit accept failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/guide-frame-edit-revert":
            try:
                result = revert_guide_edit(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state("outpaint")})
            except Exception as exc:
                APP.log.append(f"Guide frame edit revert failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/browse":
            try:
                selected = browse_path(str(data.get("kind", "file")), str(data.get("current", "")))
                self.send_json({"ok": True, "path": selected})
            except Exception as exc:
                APP.log.append(f"Browse failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/browse-global-source":
            try:
                selected = browse_path("file", str(data.get("current", "")))
                if selected:
                    APP.update_settings("global", {"source": selected})
                self.send_json({"ok": True, "path": selected, "state": APP.state("global")})
            except Exception as exc:
                APP.log.append(f"Browse failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/overview-clear":
            APP.clear_overview()
            self.send_json({"ok": True, "state": APP.state("global")})
        elif parsed.path == "/api/project-save":
            try:
                result = APP.save_project(bool(data.get("save_as")))
                self.send_json({"ok": True, **result, "state": APP.state("global")})
            except Exception as exc:
                APP.log.append(f"Project save failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/project-load":
            try:
                result = APP.load_project()
                self.send_json({"ok": True, **result, "state": APP.state("global")})
            except Exception as exc:
                APP.log.append(f"Project load failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/cache-delete":
            try:
                if data.get("all"):
                    result = delete_cache_category("all")
                elif data.get("category"):
                    result = delete_cache_category(str(data.get("category", "")))
                else:
                    result = delete_cache_file(str(data.get("path", "")))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Cache delete failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        else:
            self.send_error(404)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_media(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            file_size = path.stat().st_size
        except FileNotFoundError:
            self.send_error(404)
            return
        range_header = self.headers.get("Range", "")
        start = 0
        end = file_size - 1
        status = 200
        if range_header.startswith("bytes="):
            raw_range = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
            left, _, right = raw_range.partition("-")
            try:
                if left:
                    start = int(left)
                    if right:
                        end = int(right)
                elif right:
                    suffix = int(right)
                    start = max(0, file_size - suffix)
                if start < 0 or end < start or start >= file_size:
                    raise ValueError
                end = min(end, file_size - 1)
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except FileNotFoundError:
            return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def model_looks_image_capable(model_id: str) -> bool:
    text = model_id.lower()
    return text.startswith("gpt-image") or text.startswith("dall-e")

