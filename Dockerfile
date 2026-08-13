FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install inference/runtime dependencies without Gradio, then pin the official
# ResembleAI source revision that contains Multilingual V3 support.
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir --no-deps \
      "git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2"

# Bake only the files required by Chatterbox Multilingual V3 into the image.
# This avoids depending on RunPod's cached-model mount at worker startup and
# prevents billed workers from downloading multi-GB model weights at runtime.
RUN python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ResembleAI/chatterbox",
    revision="main",
    local_dir="/models/chatterbox",
    allow_patterns=[
        "ve.pt",
        "t3_mtl23ls_v3.safetensors",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
        "Cangjie5_TC.json",
    ],
)
PY

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
