"""RunPod generation worker (production): SDXL + LCM-LoRA (8 steps) + per-pet LoRA.

Production config from runs 3-8: the LCM-LoRA is fused into the base weights once at startup,
generation is 8 steps without CFG at 1024px, and every pet LoRA stays resident in its own slot
(multi_lora.py, run 8: +33% throughput vs. swapping, ~26 MB per pet). Slots are reused
least-recently-used. The server builds the prompt, including the pet token and its description
("a sks cat, British Shorthair, gray, ..., as a pirate").

Input:
  prompt (str, required), lora (URL or local path to the pet's LoRA, optional),
  lora_scale (default 0.9), seed, width, height (default 1024), steps (default 8)
Output:
  image_base64 (PNG), seed, timings_ms {cold_load, lora_load, inference, encode}
"""
import base64
import collections
import hashlib
import io
import os
import time

import requests
import runpod
import torch
from diffusers import AutoencoderKL, LCMScheduler, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

import multi_lora as ml

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
VAE_ID = os.environ.get("VAE_ID", "madebyollin/sdxl-vae-fp16-fix")  # the stock VAE NaNs in fp16
LCM_LORA = ("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors")
LORA_RANK = int(os.environ.get("LORA_RANK", 8))
LORA_SLOTS = int(os.environ.get("LORA_SLOTS", 32))  # resident pets per worker
LORA_CACHE = os.environ.get("LORA_CACHE", "/tmp/loras")

# Loaded once per container so warm requests skip this cost.
_t0 = time.perf_counter()
vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.load_lora_weights(hf_hub_download(*LCM_LORA), adapter_name="lcm")
pipe.fuse_lora(adapter_names=["lcm"])
pipe.unload_lora_weights()  # keeps the fused LCM weights
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
selector, wrapped = ml.wrap_unet(pipe.unet, rank=LORA_RANK, max_adapters=LORA_SLOTS + 1)  # slot 0 = no LoRA
COLD_LOAD_MS = (time.perf_counter() - _t0) * 1000
_first_request = True

slots = collections.OrderedDict()  # (lora source, scale) -> slot, least recently used first


def fetch_lora(src: str) -> str:
    if not (src.startswith("http://") or src.startswith("https://")):
        return src
    os.makedirs(LORA_CACHE, exist_ok=True)
    path = os.path.join(LORA_CACHE, hashlib.sha1(src.encode()).hexdigest() + ".safetensors")
    if not os.path.exists(path):
        r = requests.get(src, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def slot_for(src, scale: float) -> int:
    """Slot holding this pet's LoRA, loading it (and evicting the least recently used pet) if needed."""
    if not src:
        return 0
    key = (src, scale)
    if key in slots:
        slots.move_to_end(key)
        return slots[key]
    if len(slots) < LORA_SLOTS:
        slot = len(slots) + 1
    else:
        _, slot = slots.popitem(last=False)
    ml.load_into_slot(wrapped, fetch_lora(src), slot, scale=scale)
    slots[key] = slot
    return slot


def handler(job):
    global _first_request
    inp = job["input"]
    prompt = inp.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    timings = {"cold_load": COLD_LOAD_MS if _first_request else 0.0}
    _first_request = False

    t = time.perf_counter()
    slot = slot_for(inp.get("lora"), float(inp.get("lora_scale", 0.9)))
    selector.idx = torch.tensor([slot], device="cuda")
    timings["lora_load"] = (time.perf_counter() - t) * 1000

    seed = int(inp.get("seed", torch.seed() % 2**32))
    torch.cuda.synchronize()
    t = time.perf_counter()
    image = pipe(
        prompt=prompt,
        num_inference_steps=int(inp.get("steps", 8)),
        guidance_scale=1.0,  # LCM: no classifier-free guidance
        width=int(inp.get("width", 1024)),
        height=int(inp.get("height", 1024)),
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    torch.cuda.synchronize()
    timings["inference"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    timings["encode"] = (time.perf_counter() - t) * 1000

    return {"image_base64": b64, "seed": seed, "timings_ms": timings}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
