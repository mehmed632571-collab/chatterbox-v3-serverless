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
BAKED_MODEL_PATH = Path(os.environ.get("CHATTERBOX_MODEL_PATH", "/models/chatterbox"))
HF_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "600"))
MAX_REFERENCE_BYTES = int(os.environ.get("MAX_REFERENCE_BYTES", str(8 * 1024 * 1024)))

# Production workers never download model weights at runtime.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_MODEL_LOCK = threading.Lock()
_REQUIRED_MODEL_FILES = {
    "ve.pt",
    "t3_mtl23ls_v3.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
}


def _is_complete_model_dir(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in _REQUIRED_MODEL_FILES)


def _resolve_cached_snapshot(model_id: str) -> Path | None:
    if "/" not in model_id:
        return None

    org, name = model_id.split("/", 1)
    model_root = HF_CACHE_ROOT / f"models--{org}--{name}"
    refs_main = model_root / "refs" / "main"
    snapshots_dir = model_root / "snapshots"

    if refs_main.is_file():
        snapshot_hash = refs_main.read_text(encoding="utf-8").strip()
        candidate = snapshots_dir / snapshot_hash
        if _is_complete_model_dir(candidate):
            return candidate

    if snapshots_dir.is_dir():
        for candidate in sorted(p for p in snapshots_dir.iterdir() if p.is_dir()):
            if _is_complete_model_dir(candidate):
                return candidate

    return None


def resolve_model_path() -> Path:
    # Primary path: model baked into the Docker image at build time.
    if _is_complete_model_dir(BAKED_MODEL_PATH):
        return BAKED_MODEL_PATH

    # Secondary path: keep compatibility with RunPod cached-model mounts if
    # they are available on a future worker host.
    cached = _resolve_cached_snapshot(MODEL_ID)
    if cached is not None:
        return cached

    missing = sorted(
        name for name in _REQUIRED_MODEL_FILES if not (BAKED_MODEL_PATH / name).is_file()
    )
    raise RuntimeError(
        "Chatterbox Multilingual V3 model files are unavailable. "
        f"Baked path: {BAKED_MODEL_PATH}; missing: {missing}"
    )


def load_model() -> ChatterboxMultilingualTTS:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this worker.")

    model_path = resolve_model_path()
    print(f"Loading Chatterbox Multilingual V3 from: {model_path}")
    model = ChatterboxMultilingualTTS.from_local(
        str(model_path),
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


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
