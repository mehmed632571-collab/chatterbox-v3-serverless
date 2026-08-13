FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install inference/runtime dependencies without Gradio. Then install the
# official ResembleAI source revision that contains Multilingual V3 support
# for ChatterboxMultilingualTTS.from_local(..., t3_model="v3").
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir --no-deps \
      "git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
