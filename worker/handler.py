"""RunPod serverless worker: SDXL + IP-Adapter (zero-shot pet identity conditioning).

Input:
  prompt (str, required), pet_image (base64 or http(s) URL, optional),
  negative_prompt, ip_adapter_scale (0-1), steps, guidance_scale, width, height, seed
Output:
  image_base64 (PNG), seed, timings_ms {cold_load, fetch_input, inference, encode}
"""
import base64
import io
import os
import time

import requests
import runpod
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
IP_ADAPTER_REPO = os.environ.get("IP_ADAPTER_REPO", "h94/IP-Adapter")
IP_ADAPTER_WEIGHT = os.environ.get("IP_ADAPTER_WEIGHT", "ip-adapter_sdxl.bin")
# fp16-safe VAE: the stock SDXL VAE upcasts to fp32 for decode, which OOMs a 16GB T4 at 1024px.
VAE_ID = os.environ.get("VAE_ID", "madebyollin/sdxl-vae-fp16-fix")
# LOW_VRAM=1 for <=16GB GPUs (Colab T4): decode in tiles to cap VAE memory.
LOW_VRAM = os.environ.get("LOW_VRAM") == "1"
# CPU_OFFLOAD=1 parks idle submodules in system RAM. Last resort: needs ~16GB+ RAM and is much slower.
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD") == "1"

# Loaded once per container so warm requests skip this cost.
_t0 = time.perf_counter()
vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="sdxl_models", weight_name=IP_ADAPTER_WEIGHT)
if CPU_OFFLOAD:
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")
if LOW_VRAM:
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
COLD_LOAD_MS = (time.perf_counter() - _t0) * 1000
_first_request = True


def load_image(src: str) -> Image.Image:
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, timeout=30)
        r.raise_for_status()
        data = r.content
    else:
        data = base64.b64decode(src.split(",", 1)[-1])  # tolerate data: URIs
    return Image.open(io.BytesIO(data)).convert("RGB")


def handler(job):
    global _first_request
    inp = job["input"]
    prompt = inp.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    timings = {"cold_load": COLD_LOAD_MS if _first_request else 0.0}
    _first_request = False

    t = time.perf_counter()
    pet = load_image(inp["pet_image"]) if inp.get("pet_image") else None
    timings["fetch_input"] = (time.perf_counter() - t) * 1000

    # Scale 0 disables the adapter for text-only requests.
    pipe.set_ip_adapter_scale(float(inp.get("ip_adapter_scale", 0.6)) if pet else 0.0)
    ip_image = pet or Image.new("RGB", (224, 224))

    seed = int(inp.get("seed", torch.seed() % 2**32))
    gen = torch.Generator("cuda").manual_seed(seed)

    torch.cuda.synchronize()
    t = time.perf_counter()
    image = pipe(
        prompt=prompt,
        negative_prompt=inp.get("negative_prompt", "blurry, deformed, low quality"),
        ip_adapter_image=ip_image,
        num_inference_steps=int(inp.get("steps", 30)),
        guidance_scale=float(inp.get("guidance_scale", 6.0)),
        width=int(inp.get("width", 1024)),
        height=int(inp.get("height", 1024)),
        generator=gen,
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
