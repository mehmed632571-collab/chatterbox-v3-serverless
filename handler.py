import base64
import io
import os
import tempfile
import threading
from pathlib import Path

import runpod
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

MODEL_ID = os.environ.get("MODEL_NAME", "ResembleAI/chatterbox")
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "600"))
MAX_REFERENCE_BYTES = int(os.environ.get("MAX_REFERENCE_BYTES", str(8 * 1024 * 1024)))

# The endpoint is configured with RunPod model caching, so inference should never
# download model weights while a billed worker is running.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_MODEL_LOCK = threading.Lock()


def resolve_snapshot_path(model_id: str) -> str:
    if "/" not in model_id:
        raise ValueError(f"MODEL_NAME must be in org/name format, got: {model_id}")

    org, name = model_id.split("/", 1)
    model_root = Path(HF_CACHE_ROOT) / f"models--{org}--{name}"
    refs_main = model_root / "refs" / "main"
    snapshots_dir = model_root / "snapshots"

    if refs_main.is_file():
        snapshot_hash = refs_main.read_text(encoding="utf-8").strip()
        candidate = snapshots_dir / snapshot_hash
        if candidate.is_dir():
            return str(candidate)

    if snapshots_dir.is_dir():
        versions = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if versions:
            return str(versions[0])

    raise RuntimeError(
        f"RunPod cached model was not found for {model_id}. "
        "Set the endpoint Model field to ResembleAI/chatterbox."
    )


def load_model() -> ChatterboxMultilingualTTS:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this worker.")

    snapshot = resolve_snapshot_path(MODEL_ID)
    print(f"Loading Chatterbox Multilingual V3 from cached snapshot: {snapshot}")
    model = ChatterboxMultilingualTTS.from_local(
        snapshot,
        device="cuda",
        t3_model="v3",
    )
    print("Chatterbox Multilingual V3 ready.")
    return model


MODEL = load_model()


def clamp_float(value, default, low, high):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def decode_reference_audio(value: str | None) -> bytes | None:
    if not value:
        return None

    if value.startswith("data:"):
        try:
            value = value.split(",", 1)[1]
        except IndexError as exc:
            raise ValueError("Invalid reference_audio data URI.") from exc

    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("reference_audio must be base64-encoded WAV audio.") from exc

    if len(raw) > MAX_REFERENCE_BYTES:
        raise ValueError(
            f"Reference audio is too large. Maximum is {MAX_REFERENCE_BYTES // (1024 * 1024)} MB."
        )
    return raw


def synthesize(payload: dict) -> dict:
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("text is required.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text must be at most {MAX_TEXT_CHARS} characters.")

    language = str(payload.get("language", "tr")).lower().strip()
    if language not in MODEL.get_supported_languages():
        raise ValueError(f"Unsupported language: {language}")

    exaggeration = clamp_float(payload.get("exaggeration"), 0.5, 0.0, 1.5)
    cfg_weight = clamp_float(payload.get("cfg_weight"), 0.5, 0.0, 1.0)
    temperature = clamp_float(payload.get("temperature"), 0.8, 0.05, 2.0)
    repetition_penalty = clamp_float(payload.get("repetition_penalty"), 1.2, 1.0, 2.0)
    min_p = clamp_float(payload.get("min_p"), 0.05, 0.0, 1.0)
    top_p = clamp_float(payload.get("top_p"), 1.0, 0.05, 1.0)

    reference_bytes = decode_reference_audio(payload.get("reference_audio"))
    reference_path = None

    try:
        if reference_bytes:
            temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp.write(reference_bytes)
            temp.close()
            reference_path = temp.name

        # Chatterbox mutates its voice conditionals when a prompt voice is used,
        # so serialize generation within one worker to prevent voices mixing.
        with _MODEL_LOCK:
            wav = MODEL.generate(
                text=text,
                language_id=language,
                audio_prompt_path=reference_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

        audio = wav.squeeze().detach().cpu().numpy()
        buffer = io.BytesIO()
        sf.write(buffer, audio, MODEL.sr, format="WAV", subtype="PCM_16")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        return {
            "audio_base64": encoded,
            "mime_type": "audio/wav",
            "sample_rate": MODEL.sr,
            "language": language,
            "model": "Chatterbox Multilingual V3",
        }
    finally:
        if reference_path:
            try:
                os.unlink(reference_path)
            except OSError:
                pass


def handler(job):
    try:
        payload = job.get("input") or {}
        if not isinstance(payload, dict):
            raise ValueError("input must be a JSON object.")
        return synthesize(payload)
    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}")
        return {
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


runpod.serverless.start({"handler": handler})
