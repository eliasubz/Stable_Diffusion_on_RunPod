"""Bake weights into the image at build time so cold starts don't download anything."""
import os

from huggingface_hub import hf_hub_download, snapshot_download

# SDXL base in fp16 (UNet, text encoders, tokenizers, scheduler, and the stock VAE used for training)
snapshot_download(
    os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"),
    allow_patterns=["*.json", "*.txt", "*fp16.safetensors", "tokenizer*/*"],
)
snapshot_download(os.environ.get("VAE_ID", "madebyollin/sdxl-vae-fp16-fix"), allow_patterns=["*.json", "*.safetensors"])
hf_hub_download("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors")  # generation: 8-step LCM
snapshot_download("Qwen/Qwen2-VL-2B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])  # pet description
