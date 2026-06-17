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
- **Expose HTTP Ports:** `8765` (ARP GUI) and `8188` (ComfyUI).
- **Container start command / entrypoint:** leave default (the image's entrypoint
  handles first-boot setup and launch).

### Environment variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | – | Hugging Face token. Needed for the **gated** `stabilityai/stable-audio-open-1.0` music model; other models download without it. |
| `ARP_PREFETCH_MODELS` | `0` | Set `1` to eagerly download the LTX + Qwen model set during first boot instead of lazily per stage. |
| `AI_REMASTER_GUI_PORT` | `8765` | ARP GUI port. |
| `ARP_WORKSPACE` | `/workspace` | Volume mount path (match the RunPod mount). |

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
