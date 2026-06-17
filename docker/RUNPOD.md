# Running ARP on RunPod

The GitHub Action publishes a GPU-ready image to GHCR:

```
ghcr.io/ferengi82/ai-remaster-pipeline:latest
```

The image bakes in everything except the large AI models. Models download to a
**persistent network volume** the first time a pipeline stage needs them, so they
survive pod restarts and never bloat the image.

## 1. Make the GHCR image pullable

In GitHub → your fork → **Packages** → `ai-remaster-pipeline` → **Package settings**:

- Set visibility to **Public** (simplest — RunPod can pull without credentials), or
- Keep it **Private** and add registry credentials in the RunPod pod (username =
  your GitHub user, password = a PAT/`read:packages` token).

## 2. Create a Network Volume

RunPod → **Storage** → create a Network Volume in the region you'll deploy in.
Size it for the model set (LTX 2.3 + Qwen Image Edit + Deep Exemplar + optional
MMAudio/Stable Audio are tens of GB — **150 GB+ recommended**).

## 3. Deploy a Pod (or Template)

- **Image:** `ghcr.io/ferengi82/ai-remaster-pipeline:latest`
- **GPU:** an NVIDIA card with enough VRAM for LTX/Qwen (24 GB+ recommended).
- **Volume:** attach the Network Volume, **mount path `/workspace`**.
- **Expose HTTP Ports:** `8765` (ARP GUI), `8188` (ComfyUI) and `8888` (file manager).
- **Container start command / entrypoint:** leave default (the image's entrypoint
  handles first-boot setup and launch).

### Environment variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | – | Hugging Face token. Needed for the **gated** `stabilityai/stable-audio-open-1.0` music model; other models download without it. |
| `ARP_PREFETCH_MODELS` | `0` | Set `1` to eagerly download the LTX + Qwen model set during first boot instead of lazily per stage. |
| `AI_REMASTER_GUI_PORT` | `8765` | ARP GUI port. |
| `ARP_WORKSPACE` | `/workspace` | Volume mount path (match the RunPod mount). |
| `ARP_ENABLE_FILES` | `1` | Set `0` to disable the web file manager. |
| `ARP_FILES_PORT` | `8888` | File manager port. |
| `ARP_FILES_USER` | `admin` | File manager login user. |
| `ARP_FILES_PASSWORD` | _(generated)_ | File manager password. **Min. 12 chars.** If unset, a random one is generated on first boot, logged once, and saved to `/workspace/.filebrowser/admin_password`. |

## File manager (port 8888)

A full web file manager ([filebrowser](https://github.com/filebrowser/filebrowser))
serves the whole volume at the **port-8888 proxy URL**. Use it to upload source
videos (e.g. into `/workspace/arp-data/input`), browse intermediate/output, and
**create folders, rename, move and delete** files — no SSH/SCP needed.

Uploads are **chunked**, so large multi-GB videos no longer hit the RunPod proxy
timeout that a single POST would.

**Login is required** (filebrowser `json` auth). Set `ARP_FILES_USER` /
`ARP_FILES_PASSWORD` (password ≥ 12 chars), or let it generate a password on first
boot — printed to the container log and stored at
`/workspace/.filebrowser/admin_password`. The filebrowser database and settings live
at `/workspace/.filebrowser/` and persist with the volume; its runtime log is
`/workspace/filemanager.log`.

## 4. Use it

- Open the **port-8765 proxy URL** from the pod → the ARP web GUI.
- The GUI auto-starts ComfyUI on `:8188` (also reachable via its proxy URL).
- Pick a source video and run a stage; models download to
  `/workspace/models/...` on first use. Outputs land under
  `/workspace/arp-data/{intermediate,output,manifests}`.
- After a pod restart the volume is detected (sentinel `/workspace/.arp_initialized`),
  so no re-download happens.

## What lives where

| Path | Persisted? | Contents |
|---|---|---|
| `/opt/arp` | image | ARP code (GUI, scripts, workflows, vendored nodes) |
| `/opt/comfyui` | image | ComfyUI core + custom nodes (`models/` symlinked to volume) |
| `/opt/venv` | image | Python 3.13 env: PyTorch CUDA + all deps |
| `/workspace/models` | **volume** | Downloaded model weights / LoRAs |
| `/workspace/arp-data` | **volume** | input / intermediate / output / manifests |
| `/workspace/hf-cache`, `/workspace/arp-cache` | **volume** | HF + ARP caches |

## Security note

ARP and ComfyUI have **no built-in authentication**, and RunPod proxy URLs are
publicly reachable by anyone who has the URL. Don't expose sensitive material, and
add a reverse proxy with auth in front if you need access control.
