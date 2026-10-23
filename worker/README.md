# PetsAlive GPU workers (RunPod serverless, built for L4)

One image, two endpoints:

| endpoint | start command | does |
|---|---|---|
| **generation** | default (`python -u handler.py`) | SDXL + LCM-LoRA (8 steps, no CFG, 1024px) + per-pet LoRA. All pet LoRAs stay resident in slots (`multi_lora.py`) |
| **training** | `python -u train.py` | DreamBooth-LoRA with our own loop (`train_fast.py`: 512px, batch 4, lr 4e-4, 200 images; bf16 on the L4), then a Qwen2-VL-2B pet description |

The settings come from the experiments in `../experiments/` (runs 1–10, all on a Kaggle T4). See `../ROADMAP.md`.

## Deploy on RunPod
1. **GitHub integration:** Serverless → New Endpoint → GitHub repo `eliasubz/Stable_Diffusion_on_RunPod`, branch `main`, Dockerfile `worker/Dockerfile`, build context `worker`. Weights are baked in at build time, so expect a long first build.
2. **Generation endpoint:** GPU **L4 (24 GB)**, active/min workers **0**, max 1–2, idle timeout 5 s, execution timeout 300 s, container disk ≥ 40 GB, FlashBoot on.
3. **Training endpoint:** same image and GPU, container start command `python -u train.py`, execution timeout 900 s, and the R2 env vars below so it returns a `lora_url` the generation endpoint can download.
4. Put the endpoint IDs and API key into the proxy server's `.env` (`RUNPOD_ENDPOINT_ID`, `RUNPOD_TRAIN_ENDPOINT_ID`, `RUNPOD_API_KEY`).

Training env (any S3-compatible bucket, e.g. Cloudflare R2): `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_PUBLIC_URL`.

## API
**Training input:** `images` (3–10, base64 or URLs), `class_word` (cat / dog / pet), `token` (default `sks`), `run_id`; optional overrides `resolution`, `batch_size`, `learning_rate`, `rank`, `images_seen`, `seed`.
**Training output:** `lora_url` (or `lora_path`), `instance_prompt`, `description`, `precision`, `timings_ms` {cold_load, prep, train, describe, upload, total}.

**Generation input:** `prompt` (the server inserts `sks cat, <description>,`), `lora` (URL or path), `lora_scale` (0.9), `seed`, `width`/`height` (1024), `steps` (8).
**Generation output:** `image_base64`, `seed`, `timings_ms` {cold_load, lora_load, inference, encode}.

## Files
- `handler.py`, `train.py`: the two RunPod handlers
- `train_fast.py`: training loop (cached latents, fused AdamW, auto bf16/fp16, profiling helpers)
- `multi_lora.py`: batched multi-LoRA layers for the UNet
- `describe.py`: Qwen2-VL-2B pet description
- `common.py`: image loading, S3 upload, fp16 LoRA conversion
- `train_diffusers.py`: the old diffusers-script trainer (runs 1–2)
- `run*.py`, `*.ipynb`, `eval/`: experiments
