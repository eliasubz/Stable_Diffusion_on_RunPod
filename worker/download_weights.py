"""Bake weights into the image at build time to cut cold starts."""
import os

from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download(
    os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"),
    allow_patterns=["*.json", "*.txt", "*fp16.safetensors", "tokenizer*/*"],
)
snapshot_download(os.environ.get("VAE_ID", "madebyollin/sdxl-vae-fp16-fix"))
for f in ["sdxl_models/ip-adapter_sdxl.bin",
          "sdxl_models/image_encoder/config.json",
          "sdxl_models/image_encoder/model.safetensors"]:
    hf_hub_download("h94/IP-Adapter", f)
