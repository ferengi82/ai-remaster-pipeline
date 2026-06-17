from __future__ import annotations

from pathlib import Path

from .config import ASPECT_PREVIEW_DIR, FILE_PREVIEW_DIR, MEDIA_CLIP_DIR, PREVIEW_DIR, ROOT
from .paths import rel, resolve
from .config import DATA_ROOT
from .config import CACHE_ROOT


def bind_context(context: dict) -> None:
    globals().update(context)


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"

def cache_categories() -> tuple[dict, ...]:
    return (
        {
            "key": "global",
            "title": "Overview",
            "description": "Source thumbnails, target-aspect previews, selected source sections, and small browser preview clips.",
            "folders": (
                PREVIEW_DIR,
                FILE_PREVIEW_DIR,
                ASPECT_PREVIEW_DIR,
                MEDIA_CLIP_DIR,
                DATA_ROOT / "intermediate" / "source_sections",
            ),
        },
        {
            "key": "outpaint",
            "title": "Outpainting",
            "description": "Prepared inputs, guide frames, per-chunk LTX renders, chunk manifests, and stitched outpainted videos.",
            "folders": (
                CACHE_ROOT / "outpaint_chunks",
                DATA_ROOT / "intermediate" / "outpaint_guides",
                DATA_ROOT / "intermediate" / "outpaint_anchors",  # legacy name
                DATA_ROOT / "intermediate" / "outpaint_prepared",
                DATA_ROOT / "intermediate" / "outpainted",
                DATA_ROOT / "manifests" / "outpaint_chunks",
            ),
        },
        {
            "key": "shots",
            "title": "Shot Detection",
            "description": "Shot manifests created by cut detection.",
            "folders": (DATA_ROOT / "manifests" / "references",),
        },
        {
            "key": "references",
            "title": "Reference Generation",
            "description": "Black-and-white shot screenshots and Qwen color reference stills.",
            "folders": (
                DATA_ROOT / "intermediate" / "outpainted_references",
                DATA_ROOT / "intermediate" / "outpainted_references_color",
            ),
        },
        {
            "key": "colour",
            "title": "Colorization",
            "description": "Per-shot colorized chunks and stitched Deep Exemplar colorized videos.",
            "folders": (
                CACHE_ROOT / "colorized_chunks",
                DATA_ROOT / "intermediate" / "outpainted_colorized",
            ),
        },
        {
            "key": "recomp",
            "title": "Recomposition",
            "description": "Final recomposited movies created by the Recomposition tab.",
            "folders": (),
        },
        {
            "key": "output",
            "title": "Output",
            "description": "Finished output movies shown on the Output tab.",
            "folders": (DATA_ROOT / "output" / "reassembled",),
        },
    )

def cache_state() -> dict:
    categories = []
    grand_total = 0
    grand_count = 0

    for category in cache_categories():
        files = cache_category_files(category)
        total = sum(int(file["size"]) for file in files)
        grand_total += total
        grand_count += len(files)
        categories.append(
            {
                "key": category["key"],
                "title": category["title"],
                "description": category["description"],
                "count": len(files),
                "total": total,
                "total_label": human_size(total),
                "files": files,
            }
        )

    return {
        "count": grand_count,
        "total": grand_total,
        "total_label": human_size(grand_total),
        "categories": categories,
    }

def cache_category_files(category: dict) -> list[dict]:
    files = []
    for folder in category["folders"]:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not cache_file_is_listable(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append(
                {
                    "path": rel(path),
                    "size": stat.st_size,
                    "size_label": human_size(stat.st_size),
                    "mtime": int(stat.st_mtime),
                }
            )
    return sorted(files, key=lambda item: str(item["path"]).lower())

def cache_file_is_listable(path: Path) -> bool:
    return path.is_file() and path.name != ".gitkeep" and not path.name.endswith((".partial", ".tmp"))

def delete_cache_file(path_text: str) -> dict:
    path = resolve(path_text)
    category = cache_category_for_path(path)
    if category is None:
        raise ValueError("That file is not in an ARP cache/intermediate category.")
    try:
        if not path.is_file():
            return {"deleted": 0, "bytes": 0}
        size = path.stat().st_size
        path.unlink()
    except FileNotFoundError:
        return {"deleted": 0, "bytes": 0}
    clean_empty_cache_dirs(category)
    APP.log.append(f"Deleted cached file: {rel(path)}")
    return {"deleted": 1, "bytes": size}

def delete_cache_category(category_key: str) -> dict:
    if category_key == "all":
        total = {"deleted": 0, "bytes": 0}
        for category in cache_categories():
            result = delete_cache_category(category["key"])
            total["deleted"] += result["deleted"]
            total["bytes"] += result["bytes"]
        APP.log.append(f"Cleared all ARP cache categories: {total['deleted']} files, {human_size(total['bytes'])}.")
        return total

    category = next((item for item in cache_categories() if item["key"] == category_key), None)
    if category is None:
        raise ValueError("Unknown cache category.")

    deleted = 0
    bytes_deleted = 0
    for file in cache_category_files(category):
        path = resolve(str(file["path"]))
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            APP.log.append(f"Could not delete cached file {rel(path)}: {exc}")
            continue
        deleted += 1
        bytes_deleted += size

    clean_empty_cache_dirs(category)
    APP.log.append(f"Cleared {category['title']}: {deleted} files, {human_size(bytes_deleted)}.")
    return {"deleted": deleted, "bytes": bytes_deleted}

def cache_category_for_path(path: Path) -> dict | None:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    for category in cache_categories():
        for folder in category["folders"]:
            try:
                resolved.relative_to(folder.resolve())
                return category
            except ValueError:
                continue
    return None

def clean_empty_cache_dirs(category: dict) -> None:
    for folder in category["folders"]:
        if not folder.exists():
            continue
        for path in sorted((item for item in folder.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass
