#!/usr/bin/env bash
# ARP first-boot setup + launcher for RunPod containers.
#
# The Docker image bakes in OS deps, Python 3.13, the ARP repo, ComfyUI core,
# PyTorch CUDA and all custom-node requirements. The large AI *models* are NOT in
# the image: they live on a persistent RunPod network volume mounted at
# ${ARP_WORKSPACE} (default /workspace) and are downloaded on first use (or eagerly
# when ARP_PREFETCH_MODELS=1). A sentinel file makes the one-time volume setup
# idempotent across pod restarts.
set -euo pipefail

ARP_ROOT="${ARP_ROOT:-/opt/arp}"
COMFY_DIR="${COMFY_DIR:-/opt/comfyui}"
WORKSPACE="${ARP_WORKSPACE:-/workspace}"
SENTINEL="$WORKSPACE/.arp_initialized"

# --- Runtime env for the GUI / ComfyUI -------------------------------------
# ARP data + caches live directly on the volume as REAL paths (not symlinks): some
# network filesystems fail ffmpeg's +faststart reopen when the output path traverses a
# symlink. ARP_DATA_DIR / ARP_CACHE_DIR make the code use these real paths.
export ARP_DATA_DIR="${ARP_DATA_DIR:-$WORKSPACE/arp-data}"
export ARP_CACHE_DIR="${ARP_CACHE_DIR:-$WORKSPACE/arp-cache}"
export HF_HOME="${HF_HOME:-$ARP_CACHE_DIR/huggingface}"
export AI_REMASTER_GUI_HOST="${AI_REMASTER_GUI_HOST:-0.0.0.0}"
export AI_REMASTER_GUI_PORT="${AI_REMASTER_GUI_PORT:-8765}"
export AI_REMASTER_NO_BROWSER=1
# The entrypoint owns ComfyUI (one instance per GPU), so the GUI must not start its own.
export AI_REMASTER_NO_COMFY_AUTOSTART=1
# HF_TOKEN (if set in the pod env) is inherited automatically and used for gated
# repos such as stabilityai/stable-audio-open-1.0.

log() { printf '[arp-entrypoint] %s\n' "$*"; }

# link <target_dir_or_path> <linkpath>
# Replace <linkpath> with a symlink to <target>, migrating any pre-existing real
# content (shipped placeholders, ComfyUI defaults) onto the persistent volume once.
link() {
  local target="$1" linkpath="$2"
  if [ -L "$linkpath" ]; then
    ln -sfn "$target" "$linkpath"
    return
  fi
  if [ -e "$linkpath" ]; then
    if [ -d "$linkpath" ]; then
      mkdir -p "$target"
      shopt -s dotglob nullglob
      local f
      for f in "$linkpath"/*; do
        mv -n "$f" "$target"/ 2>/dev/null || true
      done
      shopt -u dotglob nullglob
      rm -rf "$linkpath"
    else
      rm -f "$linkpath"
    fi
  fi
  ln -sfn "$target" "$linkpath"
}

init_volume() {
  log "Initializing persistent volume at $WORKSPACE"
  mkdir -p \
    "$WORKSPACE"/models/checkpoints \
    "$WORKSPACE"/models/diffusion_models \
    "$WORKSPACE"/models/loras \
    "$WORKSPACE"/models/text_encoders \
    "$WORKSPACE"/models/unet \
    "$WORKSPACE"/models/vae \
    "$WORKSPACE"/models/latent_upscale_models \
    "$WORKSPACE"/models/mmaudio \
    "$ARP_DATA_DIR"/input \
    "$ARP_DATA_DIR"/intermediate \
    "$ARP_DATA_DIR"/manifests \
    "$ARP_DATA_DIR"/output \
    "$ARP_CACHE_DIR"/huggingface

  # Models: download target of dependency_manager AND ComfyUI's load path both
  # resolve to <comfy>/models -> persistent volume. (A symlink is fine here: model
  # files are written by plain copy/download, not by ffmpeg's +faststart reopen.)
  link "$WORKSPACE/models" "$COMFY_DIR/models"

  # ARP data + caches are used directly via ARP_DATA_DIR / ARP_CACHE_DIR (real volume
  # paths) — no symlinks into the code tree, so ffmpeg +faststart works on the volume.

  write_config

  if [ "${ARP_PREFETCH_MODELS:-0}" = "1" ]; then
    prefetch_models
  else
    log "ARP_PREFETCH_MODELS!=1: models will download on demand when a stage first runs."
  fi
}

write_config() {
  log "Writing $ARP_ROOT/.ai_remaster_config.json"
  cat > "$ARP_ROOT/.ai_remaster_config.json" <<EOF
{
  "comfy_dir": "$COMFY_DIR",
  "comfy_url": "http://127.0.0.1:8188",
  "comfy_host": "0.0.0.0",
  "comfy_port": "8188",
  "comfy_managed_by_arp": "true"
}
EOF
}

prefetch_models() {
  log "ARP_PREFETCH_MODELS=1: prefetching LTX + Qwen model set (this can take a while)..."
  ( cd "$ARP_ROOT/scripts" && python - <<'PY'
from pathlib import Path
import dependency_manager as dm

comfy = Path("/opt/comfyui")
dm.ensure_outpaint_models(comfy)
dm.ensure_qwen_image_edit_models(comfy)
print("Prefetch complete.")
PY
  ) || log "Prefetch failed (continuing; models will retry on demand)."
}

# Detect the GPU count and launch one ComfyUI instance per GPU (each pinned to its own GPU).
# Instance 0 (port 8188) is the primary used by the GUI and all non-upscale stages; the upscale
# stage spreads chunks across every instance via the exported ARP_COMFY_URLS. Override the count
# with ARP_COMFY_GPUS (e.g. 1 to force single-GPU).
start_comfyui_instances() {
  local n="${ARP_COMFY_GPUS:-auto}"
  if [ "$n" = "auto" ] || [ "$n" = "0" ] || [ -z "$n" ]; then
    n=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
    fi
    if [ -z "$n" ] || [ "$n" = "0" ]; then
      n="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
    fi
  fi
  case "$n" in ''|*[!0-9]*) n=1 ;; esac
  [ "$n" -ge 1 ] || n=1
  log "Detected/using $n GPU(s); starting $n ComfyUI instance(s)."

  local urls="" i port bind
  for i in $(seq 0 $((n - 1))); do
    port=$((8188 + i))
    [ "$i" -eq 0 ] && bind="0.0.0.0" || bind="127.0.0.1"
    log "  ComfyUI #$i -> GPU $i, http://${bind}:${port}"
    ( cd "$COMFY_DIR" && CUDA_VISIBLE_DEVICES="$i" \
        python "$COMFY_DIR/main.py" --listen "$bind" --port "$port" \
        >> "$WORKSPACE/comfyui-$i.log" 2>&1 ) &
    urls="${urls:+$urls,}http://127.0.0.1:${port}"
  done
  export ARP_COMFY_URLS="$urls"
  log "ARP_COMFY_URLS=$ARP_COMFY_URLS"

  # Best-effort wait for each instance's HTTP server (model load stays lazy / per-prompt).
  local timeout="${ARP_COMFY_START_TIMEOUT:-300}" waited
  for i in $(seq 0 $((n - 1))); do
    port=$((8188 + i)); waited=0
    until curl -fsS -o /dev/null "http://127.0.0.1:${port}/" 2>/dev/null; do
      sleep 2; waited=$((waited + 2))
      if [ "$waited" -ge "$timeout" ]; then
        log "  ComfyUI on port $port not ready after ${waited}s (continuing; see comfyui-$i.log)."
        break
      fi
    done
    [ "$waited" -lt "$timeout" ] && log "  ComfyUI on port $port ready."
  done
}

# --- main -------------------------------------------------------------------
if [ ! -e "$SENTINEL" ]; then
  init_volume
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SENTINEL"
  log "First-boot setup complete."
else
  log "Volume already initialized ($SENTINEL); refreshing config + model symlink."
  # Re-assert the models symlink + config in case the image was rebuilt with changes.
  mkdir -p "$ARP_DATA_DIR"/input "$ARP_DATA_DIR"/intermediate \
           "$ARP_DATA_DIR"/manifests "$ARP_DATA_DIR"/output "$ARP_CACHE_DIR"/huggingface
  link "$WORKSPACE/models" "$COMFY_DIR/models"
  write_config
fi

# filebrowser-based web file manager. Supports chunked uploads (no proxy
# timeouts on multi-GB videos), delete, rename/move and mkdir. All DB mutations
# happen here BEFORE the server starts, because filebrowser's bolt DB is locked
# while the server runs.
start_filemanager() {
  if [ "${ARP_ENABLE_FILES:-1}" = "0" ]; then
    log "File manager disabled (ARP_ENABLE_FILES=0)."
    return
  fi
  if ! command -v filebrowser >/dev/null 2>&1; then
    log "filebrowser not found; skipping file manager."
    return
  fi

  local port="${ARP_FILES_PORT:-8888}"
  local user="${ARP_FILES_USER:-admin}"
  local fb_dir="$WORKSPACE/.filebrowser"
  local fb_db="$fb_dir/filebrowser.db"
  local fb_pwfile="$fb_dir/admin_password"
  mkdir -p "$fb_dir"

  # filebrowser requires passwords >= 12 chars. Use ARP_FILES_PASSWORD if valid,
  # otherwise reuse a previously generated one or mint a new one (logged once).
  local pass="${ARP_FILES_PASSWORD:-}"
  if [ -n "$pass" ] && [ "${#pass}" -lt 12 ]; then
    log "ARP_FILES_PASSWORD is shorter than 12 chars (filebrowser minimum); generating one instead."
    pass=""
  fi
  if [ -z "$pass" ]; then
    if [ -s "$fb_pwfile" ]; then
      pass="$(cat "$fb_pwfile")"
    else
      pass="$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 16)"
      printf '%s' "$pass" > "$fb_pwfile"
      chmod 600 "$fb_pwfile"
      log "Generated file-manager login (saved to $fb_pwfile):"
      log "    user: ${user}   password: ${pass}"
    fi
  fi

  if [ ! -f "$fb_db" ]; then
    log "Initializing filebrowser database at $fb_db"
    filebrowser config init -d "$fb_db" >/dev/null
  fi
  # (Re)apply runtime config every boot so env changes take effect.
  filebrowser config set -d "$fb_db" --auth.method=json \
    -a 0.0.0.0 -p "$port" -r "$WORKSPACE" >/dev/null
  # Ensure the admin user exists with the current password (server not running yet).
  if ! filebrowser users add "$user" "$pass" --perm.admin -d "$fb_db" >/dev/null 2>&1; then
    filebrowser users update "$user" --password "$pass" --perm.admin -d "$fb_db" >/dev/null 2>&1 || true
  fi

  log "Starting file manager (filebrowser) on 0.0.0.0:${port} -> $WORKSPACE (login user: ${user})"
  filebrowser -d "$fb_db" >> "$WORKSPACE/filemanager.log" 2>&1 &
  log "File manager log: $WORKSPACE/filemanager.log"
}

start_comfyui_instances
start_filemanager

log "Starting ARP GUI on ${AI_REMASTER_GUI_HOST}:${AI_REMASTER_GUI_PORT} (ComfyUI instances managed by entrypoint)"
cd "$ARP_ROOT"
exec python -m ai_remaster_gui "$@"
