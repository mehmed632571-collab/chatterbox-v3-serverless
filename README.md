# Chatterbox V3 Serverless

RunPod Serverless worker for **ResembleAI Chatterbox Multilingual V3**, prepared primarily for Turkish TTS.

## What this worker does

- Loads `ResembleAI/chatterbox` from RunPod's Hugging Face model cache.
- Forces the multilingual **V3** T3 checkpoint (`t3_model="v3"`).
- Defaults to Turkish (`tr`).
- Accepts an optional base64 WAV reference voice.
- Exposes Chatterbox controls including exaggeration, CFG weight and temperature.
- Returns a base64 WAV file.

## RunPod deployment

Create a **Serverless / Queue** endpoint and import this GitHub repository.

Recommended endpoint settings for the first test:

- Branch: `main`
- Dockerfile: `Dockerfile`
- Minimum workers: `0`
- Maximum workers: `1`
- Model cache / Model: `ResembleAI/chatterbox`
- Use a CUDA GPU with sufficient VRAM; start with a 16 GB-or-larger option for testing.
- Do not attach the LTX Network Volume; this worker uses RunPod model caching instead.

## Minimal Turkish test request

```json
{
  "input": {
    "text": "Bebeğim, geldin mi? Ben de seni bekliyordum. Seni çok özledim.",
    "language": "tr",
    "exaggeration": 0.7,
    "cfg_weight": 0.3
  }
}
```

## Reference voice

For voice cloning / conditioning, encode a short clean WAV file as base64 and send it as `reference_audio`:

```json
{
  "input": {
    "text": "Bu gece seni bekledim.",
    "language": "tr",
    "reference_audio": "<BASE64_WAV>",
    "exaggeration": 0.7,
    "cfg_weight": 0.3
  }
}
```

The response includes `audio_base64`, `mime_type`, `sample_rate`, `language`, and `model`.

## Notes

The worker intentionally runs Hugging Face in offline mode after RunPod has populated the endpoint's cached model. This avoids downloading model weights while a billed worker is already running.

Build trigger: initial RunPod Serverless deployment.
