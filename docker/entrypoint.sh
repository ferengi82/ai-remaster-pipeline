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
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export AI_REMASTER_GUI_HOST="${AI_REMASTER_GUI_HOST:-0.0.0.0}"
export AI_REMASTER_GUI_PORT="${AI_REMASTER_GUI_PORT:-8765}"
export AI_REMASTER_NO_BROWSER=1
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
    "$WORKSPACE"/arp-data/input \
    "$WORKSPACE"/arp-data/intermediate \
    "$WORKSPACE"/arp-data/manifests \
    "$WORKSPACE"/arp-data/output \
    "$WORKSPACE"/hf-cache \
    "$WORKSPACE"/arp-cache

  # Models: download target of dependency_manager AND ComfyUI's load path both
  # resolve to <comfy>/models -> persistent volume.
  link "$WORKSPACE/models" "$COMFY_DIR/models"

  # ARP user data + caches persist across restarts.
  link "$WORKSPACE/arp-data/input"        "$ARP_ROOT/input"
  link "$WORKSPACE/arp-data/intermediate" "$ARP_ROOT/intermediate"
  link "$WORKSPACE/arp-data/manifests"    "$ARP_ROOT/manifests"
  link "$WORKSPACE/arp-data/output"       "$ARP_ROOT/output"
  link "$WORKSPACE/arp-cache"             "$ARP_ROOT/.cache"

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

# --- main -------------------------------------------------------------------
if [ ! -e "$SENTINEL" ]; then
  init_volume
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SENTINEL"
  log "First-boot setup complete."
else
  log "Volume already initialized ($SENTINEL); refreshing config + symlinks."
  # Re-assert symlinks/config in case the image was rebuilt with changes.
  link "$WORKSPACE/models" "$COMFY_DIR/models"
  link "$WORKSPACE/arp-data/input"        "$ARP_ROOT/input"
  link "$WORKSPACE/arp-data/intermediate" "$ARP_ROOT/intermediate"
  link "$WORKSPACE/arp-data/manifests"    "$ARP_ROOT/manifests"
  link "$WORKSPACE/arp-data/output"       "$ARP_ROOT/output"
  link "$WORKSPACE/arp-cache"             "$ARP_ROOT/.cache"
  write_config
fi

log "Starting ARP GUI on ${AI_REMASTER_GUI_HOST}:${AI_REMASTER_GUI_PORT} (ComfyUI autostarts on :8188)"
cd "$ARP_ROOT"
exec python -m ai_remaster_gui "$@"
