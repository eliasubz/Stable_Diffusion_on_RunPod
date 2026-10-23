"""RunPod training worker (production): pet photos in, DreamBooth-LoRA + pet description out.

Uses our own training loop (train_fast.py) with the config from runs 6-7: cached VAE latents and
prompt, fused AdamW, no gradient checkpointing, 512px, batch 4, lr 4e-4, 200 images seen. Precision
is picked automatically (bf16 on the L4, fp16 + loss scaling on a T4). After training, Qwen2-VL-2B
describes the pet (run 4) so the server can add the description to every prompt.

Models are loaded once per worker, so warm jobs skip loading.

Input:
  images (list of base64 or http(s) URLs, 3-10, required), class_word ("cat" / "dog" / "pet"),
  token (rare identifier, default "sks"), run_id (names the uploaded LoRA),
  optional overrides: resolution, batch_size, learning_rate, rank, images_seen, seed
Output:
  lora_url (when S3_* env is set) or lora_path, instance_prompt, description, precision,
  timings_ms {cold_load, prep, train, describe, upload, total}
"""
import gc
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA initializes

import runpod  # noqa: E402
import torch  # noqa: E402

import train_fast as tf  # noqa: E402
from common import load_image, to_fp16, upload  # noqa: E402
from describe import PetDescriber  # noqa: E402

PRODUCTION = {"resolution": 512, "batch_size": 4, "learning_rate": 4e-4, "rank": 8}
IMAGES_SEEN = 200
OVERRIDABLE = ("resolution", "batch_size", "learning_rate", "rank", "seed")
LORA_DIR = os.environ.get("LORA_DIR", "/tmp/loras")

_t0 = time.perf_counter()
COMPONENTS = tf.load_components(fp32_vae=True)
# The VAE and text encoders are only needed to cache latents and the prompt at the start of a job:
# they wait on the CPU and visit the GPU for ~1 s, leaving ~2 GB more for training.
ENCODERS = ("vae", "te1", "te2")


def encoders_to(device):
    for k in ENCODERS:
        COMPONENTS[k].to(device)
    if device == "cpu":
        torch.cuda.empty_cache()


encoders_to("cpu")
DESCRIBER = PetDescriber()  # stays on the CPU between jobs
COLD_LOAD_MS = (time.perf_counter() - _t0) * 1000
_first_job = True


def train(inp: dict, progress=None) -> dict:
    global _first_job
    srcs = inp.get("images") or []
    if not 3 <= len(srcs) <= 10:
        return {"error": "send between 3 and 10 images"}

    t_start = time.perf_counter()
    timings = {"cold_load": COLD_LOAD_MS if _first_job else 0.0}
    _first_job = False
    class_word = inp.get("class_word", "pet")
    instance_prompt = f"a photo of {inp.get('token', 'sks')} {class_word}"
    run_id = inp.get("run_id") or time.strftime("%Y%m%d-%H%M%S")

    t = time.perf_counter()
    images = [load_image(s) for s in srcs]
    timings["prep"] = (time.perf_counter() - t) * 1000

    cfg = tf.Config(cache_latents=True, cache_text=True, optimizer="adamw_fused", grad_ckpt=False, precision="auto",
                    **{**PRODUCTION, **{k: inp[k] for k in OVERRIDABLE if k in inp}})
    steps = math.ceil(int(inp.get("images_seen", IMAGES_SEEN)) / cfg.batch_size)
    t = time.perf_counter()
    encoders_to("cuda")
    trainer = tf.Trainer(images, instance_prompt, cfg, COMPONENTS)  # caches latents + prompt embeddings
    encoders_to("cpu")
    precision = "bf16" if trainer.amp_dtype == torch.bfloat16 else "fp16"
    for s in range(1, steps + 1):
        trainer.step()
        if progress and (s % 10 == 0 or s == steps):
            progress(f"training {s}/{steps}")
    lora_path = trainer.save(os.path.join(LORA_DIR, run_id))
    torch.cuda.synchronize()
    timings["train"] = (time.perf_counter() - t) * 1000
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    to_fp16(lora_path)

    if progress:
        progress("describing")
    description, raw, describe_s = DESCRIBER.describe(images, class_word)
    timings["describe"] = describe_s * 1000

    t = time.perf_counter()
    lora_url = upload(lora_path, f"loras/{run_id}.safetensors")
    timings["upload"] = (time.perf_counter() - t) * 1000
    timings["total"] = (time.perf_counter() - t_start) * 1000

    out = {"instance_prompt": instance_prompt, "description": description, "description_raw": raw,
           "precision": precision, "steps": steps, "config": {**PRODUCTION, **{k: inp[k] for k in OVERRIDABLE if k in inp}},
           "size_mb": round(os.path.getsize(lora_path) / 1e6, 1), "timings_ms": timings}
    if lora_url:
        out["lora_url"] = lora_url
    else:
        out["lora_path"] = lora_path
    print(f"[train] {run_id}: {steps} steps ({precision}) in {timings['train'] / 1000:.0f} s, "
          f"described in {describe_s:.0f} s: {description!r}", flush=True)
    return out


def handler(job):
    # Progress shows up in the RunPod /status response while the job runs
    return train(job["input"], progress=lambda msg: runpod.serverless.progress_update(job, msg))


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
