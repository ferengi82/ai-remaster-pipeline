from __future__ import annotations

import json
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

from .config import IMAGE_EXTS, ROOT, VIDEO_EXTS
from . import state
from .manifests import read_manifest, read_outpaint_chunk_rows
from .paths import resolve, resolve_video_source, safe_stem
from .runtime_settings import load_settings

PROJECT_SCHEMA_VERSION = 2
PROJECT_JSON_NAME = "project.json"

# Extensions allowed inside a project bundle. .json is included so outpaint guide sidecars
# (resume signatures, edit metadata) travel with the guides and aren't treated as stale on load.
BUNDLE_EXTS = IMAGE_EXTS | {".csv", ".txt", ".json"}


def source_signature(source_text: str) -> tuple[str, int, int] | None:
    if not source_text:
        return None
    source = resolve_video_source(source_text)
    if not source.exists() or source.suffix.lower() not in VIDEO_EXTS:
        return None
    stat = source.stat()
    return str(source), stat.st_size, stat.st_mtime_ns

def source_analysis_key(signature: tuple[str, int, int]) -> str:
    return "\0".join(str(part) for part in signature)

def project_payload(settings: dict[str, dict[str, str]]) -> dict:
    stored_settings = {
        stage: {key: value for key, value in values.items() if key != "openai_api_key"}
        for stage, values in settings.items()
    }
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "app": "AI Remaster Pipeline",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "settings": stored_settings,
    }

def write_project_file(path: Path, settings: dict[str, dict[str, str]]) -> None:
    payload = project_payload(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PROJECT_JSON_NAME, json.dumps(payload, indent=2) + "\n")
        for asset in project_asset_paths(settings):
            archive.write(asset, asset.relative_to(ROOT).as_posix())

def read_project_file(path: Path) -> dict[str, dict[str, str]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            try:
                data = json.loads(archive.read(PROJECT_JSON_NAME).decode("utf-8-sig"))
            except KeyError as exc:
                raise RuntimeError("Project bundle does not contain project.json.") from exc
            extract_project_assets(archive)
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Project file is not valid JSON: {exc}") from exc
    return load_project_payload(data)

def load_project_payload(data: dict) -> dict[str, dict[str, str]]:
    try:
        version = int(data.get("schema_version", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Project file has an invalid schema_version.") from exc
    if version < 1:
        raise RuntimeError("Project file does not include a supported schema_version.")
    if version > PROJECT_SCHEMA_VERSION:
        raise RuntimeError(f"Project schema {version} is newer than this ARP build supports.")
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise RuntimeError("Project file does not contain settings.")
    loaded = load_settings()
    for stage, values in settings.items():
        if stage in loaded and isinstance(values, dict):
            loaded[stage].update({str(key): str(value) for key, value in values.items()})
    if not loaded.get("global", {}).get("source"):
        loaded.setdefault("global", {})["expand_outpaint"] = "true"
    loaded.setdefault("global", {}).setdefault("upscale", "false")
    return loaded

def project_asset_paths(settings: dict[str, dict[str, str]]) -> list[Path]:
    candidates: list[str] = []
    for key in ("shots", "references", "colour", "recomp"):
        manifest = settings.get(key, {}).get("manifest", "")
        if manifest:
            candidates.append(manifest)
    assets: list[Path] = []
    seen: set[Path] = set()
    for text in candidates:
        manifest = resolve(text)
        if not project_asset_is_bundleable(manifest) or manifest in seen:
            continue
        seen.add(manifest)
        assets.append(manifest)
        for row in read_manifest(manifest):
            for field in ("source_reference", "color_reference"):
                image = resolve(row.get(field, ""))
                if project_asset_is_bundleable(image) and image not in seen:
                    seen.add(image)
                    assets.append(image)
    # Outpaint chunk manifests and the hand-edited guide frames they reference, so a project
    # keeps the user's manual guide work for safekeeping.
    for asset in outpaint_guide_asset_paths(settings):
        if asset not in seen:
            seen.add(asset)
            assets.append(asset)
    return assets

def chunk_guide_image_texts(manifest: Path) -> list[str]:
    """Guide image paths referenced by a chunk manifest's rows (the active guide per frame plus
    one undo step), covering both hand-edited guides (outpaint_guides/) and auto seed guides
    (outpaint_seed_guides/). Mirrors outpaint_guides._parse_guide_frames, including the legacy
    guide_image / guide_end_image fields, without importing that module."""
    texts: list[str] = []
    try:
        rows = read_outpaint_chunk_rows(manifest)
    except Exception:
        return texts
    for row in rows.values():
        frames: list = []
        raw = (row.get("guide_frames") or "").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    frames = parsed
            except (json.JSONDecodeError, TypeError):
                frames = []
        for legacy in ("guide_image", "guide_end_image"):
            if row.get(legacy):
                frames.append({"image": row[legacy]})
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            for key in ("image", "image_previous"):
                value = frame.get(key)
                if value:
                    texts.append(str(value))
    return texts

def outpaint_guide_asset_paths(settings: dict[str, dict[str, str]]) -> list[Path]:
    # The outpaint chunk manifest is identity-keyed now (e.g. Metropolis_chunks_<key>.csv), so
    # locate it precisely via the same GUI helper that names it, rather than by stem prefix
    # (which also avoids bundling other sections of the same film). These helpers are injected
    # from server.py via bind_context.
    chunk_manifest_for = globals().get("outpaint_chunk_manifest_for")
    pipeline_source = globals().get("pipeline_source_text")
    if not chunk_manifest_for or not pipeline_source:
        return []
    try:
        manifest_text = chunk_manifest_for(pipeline_source(settings), settings.get("outpaint", {}))
    except Exception:
        return []
    manifest = resolve(manifest_text) if manifest_text else None
    if not manifest or not project_asset_is_bundleable(manifest):
        return []
    assets: list[Path] = [manifest]
    seen: set[Path] = {manifest}
    for image_text in chunk_guide_image_texts(manifest):
        image = resolve(image_text)
        # Bundle the guide image and its resume/edit sidecars so reloaded guides are used
        # as-is and not treated as stale.
        for candidate in (image, Path(str(image) + ".sig.json"), Path(str(image) + ".json")):
            if candidate not in seen and project_asset_is_bundleable(candidate):
                seen.add(candidate)
                assets.append(candidate)
        seen.add(image)
    return assets

def project_asset_is_bundleable(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.suffix.lower() not in BUNDLE_EXTS:
        return False
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True

def asset_already_on_disk(info: zipfile.ZipInfo, target: Path) -> bool:
    """True when the target file already holds the archived bytes. Restoring it anyway would
    only refresh its mtime, which used to invalidate resume signatures of dependent outputs
    (e.g. outpaint chunks re-rendering after every project load)."""
    try:
        if not target.is_file() or target.stat().st_size != info.file_size:
            return False
        return zlib.crc32(target.read_bytes()) == info.CRC
    except OSError:
        return False

def extract_project_assets(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if name == PROJECT_JSON_NAME or name.startswith("/") or ".." in Path(name).parts:
            continue
        if Path(name).suffix.lower() not in BUNDLE_EXTS:
            continue
        target = ROOT / name
        try:
            target.resolve().relative_to(ROOT.resolve())
        except ValueError:
            continue
        # Restore each asset independently so one failure (e.g. a Windows MAX_PATH limit on a
        # deeply nested guide-edit file) can't abort restoring the rest of the project.
        try:
            if asset_already_on_disk(info, target):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as dest:
                dest.write(source.read())
        except OSError as exc:
            print(f"Warning: could not restore project asset {name}: {exc}", flush=True)

def project_default_path(settings: dict[str, dict[str, str]]) -> Path:
    source = resolve_video_source(settings.get("global", {}).get("source", ""))
    stem = safe_stem(source.name if source.name else "arp_project")
    return ROOT / "projects" / f"{stem}.arpp"

def project_save_suggestion(settings: dict[str, dict[str, str]], project_path: Path | None = None) -> Path:
    if project_path:
        return project_path
    default_path = project_default_path(settings)
    last_dir = last_browse_dir(settings)
    return (last_dir / default_path.name) if last_dir else default_path

def last_browse_dir(settings: dict[str, dict[str, str]] | None = None) -> Path | None:
    values = settings or (state.APP.settings if state.APP is not None else {})
    text = values.get("global", {}).get("last_browse_dir", "") if isinstance(values, dict) else ""
    if not text:
        return None
    path = resolve(str(text))
    return path if path.exists() and path.is_dir() else None
