# ARP (AI Remaster Pipeline) — RunPod-ready image.
#
# Baked into the image: OS deps, ffmpeg, Python 3.13, PyTorch CUDA, ComfyUI core,
# the bundled ComfyUI custom nodes + their requirements, and the ARP repo itself.
# NOT baked in: the large AI models — they download to a persistent volume on first
# use (see docker/entrypoint.sh and docker/RUNPOD.md).
#
# Build:
#   docker build -t arp:test .
# Override CUDA / torch channel if needed:
#   docker build --build-arg CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 \
#                --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 -t arp:test .

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# --- OS dependencies + Python 3.13 (deadsnakes) ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git ffmpeg \
        build-essential libgl1 libglib2.0-0 \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3.13-dev \
 && rm -rf /var/lib/apt/lists/*

# --- Python venv ------------------------------------------------------------
RUN python3.13 -m venv /opt/venv \
 && pip install --upgrade pip setuptools wheel

# --- PyTorch CUDA (large, cache-stable layer; installed before ComfyUI so its
#     unpinned torch requirement is already satisfied and not clobbered) ------
RUN pip install torch torchvision torchaudio --index-url ${TORCH_INDEX_URL}

# --- ComfyUI core -----------------------------------------------------------
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/comfyui \
 && pip install -r /opt/comfyui/requirements.txt

# --- Extra runtime deps the Windows installer also pulls in -----------------
RUN pip install "huggingface_hub[cli]" opencv-contrib-python imageio-ffmpeg pillow numpy numba

# --- ARP repo ---------------------------------------------------------------
COPY . /opt/arp
RUN pip install -r /opt/arp/requirements.txt

# --- Bundled custom nodes -> ComfyUI + their requirements -------------------
RUN set -eux; \
    for d in /opt/arp/vendor/comfyui_custom_nodes/*/; do \
        name="$(basename "$d")"; \
        cp -r "$d" "/opt/comfyui/custom_nodes/$name"; \
        if [ -f "/opt/comfyui/custom_nodes/$name/requirements.txt" ]; then \
            pip install -r "/opt/comfyui/custom_nodes/$name/requirements.txt" \
              || echo "WARN: requirements for custom node $name failed"; \
        fi; \
    done; \
    pip install scikit-image einops tqdm matplotlib

RUN chmod +x /opt/arp/docker/entrypoint.sh

ENV ARP_ROOT=/opt/arp \
    COMFY_DIR=/opt/comfyui \
    ARP_WORKSPACE=/workspace

EXPOSE 8765 8188
WORKDIR /opt/arp
ENTRYPOINT ["/opt/arp/docker/entrypoint.sh"]
