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

# Install the inference/runtime dependencies without Gradio. Gradio is only the
# optional web UI for Chatterbox and conflicts with the current RunPod SDK over
# tomlkit. Serverless uses handler.py, so Gradio is not needed here.
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir --no-deps chatterbox-tts==0.1.7

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
