"""DreamBooth-LoRA (SDXL) training loop with switchable speedups and per-phase timing.

Matches diffusers' train_dreambooth_lora_sdxl.py (v0.32.2) where it matters: UNet LoRA on
to_q/to_k/to_v/to_out.0 with alpha = rank and gaussian init, epsilon-prediction MSE, uniform
timesteps, AdamW (wd 1e-4), grad clipping at 1.0, fp16 autocast with loss scaling, same saved
LoRA format. What can be switched on or off:

  cache_latents   encode the photos with the VAE once and sample from the cached latent distribution
                  (the script runs the fp32 VAE encoder every step)
  cache_text      encode the fixed instance prompt once (the script already does this)
  optimizer       "adamw" | "adamw_fused" | "adamw8bit" (bitsandbytes, falls back to adamw_fused)
  grad_ckpt       gradient checkpointing in the UNet (saves memory, costs recompute)
  batch_size      photos per step
  fp16_lora       the "cast fix": the LoRA weights in the UNet are fp16, so autocast doesn't re-cast them on
                  every use; fp32 master copies live in the optimizer, and grads / weights are copied
                  between the two with one foreach op each per step (classic mixed precision with masters)
  compile         torch.compile the UNet forward
  precision       "auto" (bf16 where supported, e.g. L4 / Ampere+, else fp16 + loss scaling as on a T4),
                  "bf16" or "fp16"

`Trainer.step()` times each phase with CUDA syncs, and `profile_step()` runs torch.profiler over
steps and groups the GPU time into UNet forward / backward / optimizer and by op type
(attention, GroupNorm, matmul, conv, elementwise).
"""
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import ImageOps
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")


@dataclass
class Config:
    resolution: int = 768
    rank: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    cache_latents: bool = True
    cache_text: bool = True
    optimizer: str = "adamw_fused"
    grad_ckpt: bool = True
    batch_size: int = 1
    fp16_lora: bool = False
    compile: bool = False
    precision: str = "auto"
    seed: int = 0


class Trainer:
    def __init__(self, images, instance_prompt, cfg: Config, components=None):
        """images: list of PIL images. components: reuse loaded models across configs (benchmarking)."""
        self.cfg = cfg
        self.dev = "cuda"
        torch.manual_seed(cfg.seed)
        c = components or load_components()
        self.c = c
        self.unet, self.vae = c["unet"], c["vae"]
        self.scheduler = c["scheduler"]
        # Native bf16 needs Ampere+ (sm80; the L4 is sm89). torch.cuda.is_bf16_supported() also returns True
        # on a T4 via slow emulation (smoke test: 3.2 s/step instead of ~0.6), so check the capability.
        native_bf16 = torch.cuda.get_device_capability()[0] >= 8
        bf16 = cfg.precision == "bf16" or (cfg.precision == "auto" and native_bf16)
        self.amp_dtype = torch.bfloat16 if bf16 else torch.float16
        if self.unet.dtype != self.amp_dtype:
            self.unet.to(self.amp_dtype)  # frozen base in the compute dtype; LoRA params are cast to fp32 below

        # LoRA exactly like the diffusers script; trainable params in fp32, frozen UNet in fp16
        self.unet.requires_grad_(False)
        if "pet" in getattr(self.unet, "peft_config", {}):
            self.unet.delete_adapters("pet")
        self.unet.add_adapter(LoraConfig(r=cfg.rank, lora_alpha=cfg.rank, init_lora_weights="gaussian",
                                         target_modules=["to_k", "to_q", "to_v", "to_out.0"]), adapter_name="pet")
        self.unet.set_adapter("pet")
        cast_training_params(self.unet, dtype=torch.float32)
        self.params = [p for p in self.unet.parameters() if p.requires_grad]
        if cfg.grad_ckpt:
            self.unet.enable_gradient_checkpointing()
        else:
            self.unet.disable_gradient_checkpointing()
        self.unet.train()

        to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.pixels = torch.stack([to_tensor(ImageOps.fit(im, (cfg.resolution,) * 2)) for im in images]).to(self.dev)
        self.instance_prompt = instance_prompt
        r = cfg.resolution
        self.add_time_ids = torch.tensor([[r, r, 0, 0, r, r]], device=self.dev, dtype=self.amp_dtype)

        self.latent_mean = self.latent_std = None
        if cfg.cache_latents:
            # One photo at a time: a batched fp32 VAE encode at 768px doesn't fit in 15 GB
            means, stds = [], []
            with torch.no_grad():
                for px in self.pixels:
                    dist = self.vae.encode(px[None].to(self.vae.dtype)).latent_dist
                    means.append(dist.mean.float())
                    stds.append(dist.std.float())
            self.latent_mean, self.latent_std = torch.cat(means), torch.cat(stds)
        self.text = self.encode_text() if cfg.cache_text else None

        if cfg.fp16_lora:
            # fp32 masters for the optimizer; the UNet's LoRA weights become fp16 working copies
            self.masters = [p.detach().clone().float() for p in self.params]
            for m in self.masters:
                m.grad = torch.zeros_like(m)
            with torch.no_grad():
                for p in self.params:
                    p.data = p.data.to(self.amp_dtype)
            self.opt, self.optimizer_name = make_optimizer(self.masters, cfg)
        else:
            self.masters = None
            self.opt, self.optimizer_name = make_optimizer(self.params, cfg)
        self.unet_fwd = torch.compile(self.unet) if cfg.compile else self.unet
        # Loss scaling only matters for fp16; bf16 has fp32's exponent range
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
        self.n_params = sum(p.numel() for p in self.params)

    @torch.no_grad()
    def encode_text(self):
        c = self.c
        embeds, pooled = [], None
        for tok, enc in [(c["tok1"], c["te1"]), (c["tok2"], c["te2"])]:
            ids = tok(self.instance_prompt, padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(self.dev)
            out = enc(ids, output_hidden_states=True)
            pooled = out[0]  # the last encoder's pooled projection is SDXL's text_embeds
            embeds.append(out.hidden_states[-2])
        return torch.cat(embeds, dim=-1), pooled

    def sample_latents(self, idx):
        sf = self.vae.config.scaling_factor
        if self.latent_mean is not None:
            m, s = self.latent_mean[idx], self.latent_std[idx]
            return (m + s * torch.randn_like(m)) * sf
        with torch.no_grad():
            return self.vae.encode(self.pixels[idx].to(self.vae.dtype)).latent_dist.sample().float() * sf

    def step(self, timed=False):
        """One optimizer step on a random photo. Returns (loss, {phase: ms}) with timed=True."""
        t, ms = {}, {}

        def mark(name):
            if timed:
                torch.cuda.synchronize()
                t[name] = time.perf_counter()

        mark("start")
        B = self.cfg.batch_size
        idx = torch.randint(0, len(self.pixels), (B,), device=self.dev)
        with record_function("vae_encode"):
            latents = self.sample_latents(idx)
        mark("vae")
        with record_function("text_encode"):
            prompt_embeds, pooled = self.text if self.text is not None else self.encode_text()
            if B > 1:
                prompt_embeds, pooled = prompt_embeds.expand(B, -1, -1), pooled.expand(B, -1)
        mark("text")

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=self.dev)
        noisy = self.scheduler.add_noise(latents, noise, timesteps)
        with record_function("unet_forward"), torch.autocast("cuda", dtype=self.amp_dtype):
            pred = self.unet_fwd(noisy, timesteps, prompt_embeds,
                                 added_cond_kwargs={"text_embeds": pooled, "time_ids": self.add_time_ids.expand(B, -1)}).sample
            loss = F.mse_loss(pred.float(), noise.float())
        mark("forward")
        with record_function("backward"):
            self.scaler.scale(loss).backward()
        mark("backward")
        with record_function("optimizer"):
            if self.masters is not None:
                # fp16 grads -> fp32 master grads, step on the masters, masters -> fp16 working weights
                torch._foreach_copy_([m.grad for m in self.masters], [p.grad for p in self.params])
                for p in self.params:
                    p.grad = None
            opt_params = self.masters if self.masters is not None else self.params
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(opt_params, self.cfg.max_grad_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
            if self.masters is not None:
                with torch.no_grad():
                    torch._foreach_copy_([p.data for p in self.params], self.masters)
            else:
                self.opt.zero_grad(set_to_none=True)
        mark("optimizer")

        if timed:
            names = ["start", "vae", "text", "forward", "backward", "optimizer"]
            ms = {n: (t[n] - t[p]) * 1000 for p, n in zip(names, names[1:])}
            ms["total"] = (t["optimizer"] - t["start"]) * 1000
        return loss.item() if timed else None, ms

    def save(self, out_dir, weight_name="pytorch_lora_weights.safetensors"):
        """Save in diffusers' LoRA format (loadable with pipe.load_lora_weights)."""
        os.makedirs(out_dir, exist_ok=True)
        layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet, adapter_name="pet"))
        StableDiffusionXLPipeline.save_lora_weights(out_dir, unet_lora_layers=layers, weight_name=weight_name)
        return os.path.join(out_dir, weight_name)


def load_components(fp32_vae=True):
    """fp32_vae=True matches the diffusers script (stock SDXL VAE, kept in fp32 to avoid NaNs)."""
    dev, f16 = "cuda", torch.float16
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", variant="fp16").to(dev, torch.float32 if fp32_vae else f16)
    return {
        "unet": UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", variant="fp16", torch_dtype=f16).to(dev),
        "vae": vae.requires_grad_(False),
        "te1": CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", variant="fp16", torch_dtype=f16).to(dev).requires_grad_(False),
        "te2": CLIPTextModelWithProjection.from_pretrained(MODEL_ID, subfolder="text_encoder_2", variant="fp16", torch_dtype=f16).to(dev).requires_grad_(False),
        "tok1": AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", use_fast=False),
        "tok2": AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2", use_fast=False),
        "scheduler": DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler"),
    }


def make_optimizer(params, cfg):
    kw = dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    if cfg.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, **kw), "adamw8bit"
        except Exception as e:  # bitsandbytes often lags new CUDA / Python builds
            print(f"[train_fast] 8-bit Adam unavailable ({type(e).__name__}: {e}); using fused AdamW", flush=True)
            return torch.optim.AdamW(params, fused=True, **kw), "adamw_fused (8bit fallback)"
    if cfg.optimizer == "adamw_fused":
        return torch.optim.AdamW(params, fused=True, **kw), "adamw_fused"
    return torch.optim.AdamW(params, **kw), "adamw"


def benchmark(trainer, steps=20, warmup=5):
    """Mean per-phase ms over `steps` timed steps after `warmup` untimed ones."""
    for _ in range(warmup):
        trainer.step()
    torch.cuda.reset_peak_memory_stats()
    runs = [trainer.step(timed=True)[1] for _ in range(steps)]
    out = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
    out["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out


# Op-name → category for the profiler breakdown (self GPU time of aten ops, forward + backward)
CATEGORIES = [
    ("attention", ("scaled_dot_product", "efficient_attention", "flash_attention", "_fused_sdp", "cudnn_attention")),
    ("groupnorm", ("group_norm",)),
    ("layernorm", ("layer_norm",)),
    ("matmul", ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::linear", "aten::matmul")),
    ("conv", ("convolution", "conv2d", "cudnn_conv")),
    ("optimizer", ("_foreach", "adam", "_fused_adamw", "optimizer_update", "clip_grad", "_amp_")),
    ("casts", ("aten::copy_", "aten::_to_copy")),
    ("compiled (triton)", ("triton_",)),
]


def categorize(name):
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k.lower() in n for k in keys):
            return cat
    return "elementwise/other"


def profile_step(trainer, steps=3, warmup=5, trace_path=None):
    """Profile `steps` training steps. Returns ({phase: GPU ms/step}, {op category: GPU ms/step}, top ops)."""
    for _ in range(warmup):
        trainer.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(steps):
            trainer.step()
        torch.cuda.synchronize()
    if trace_path:
        prof.export_chrome_trace(trace_path)

    avg = prof.key_averages()
    dev_time = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)  # noqa: E731
    self_dev = lambda e: getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)  # noqa: E731

    phases = {e.key: dev_time(e) / 1000 / steps for e in avg
              if e.key in ("vae_encode", "text_encode", "unet_forward", "backward", "optimizer")}
    cats, top = {}, []
    for e in avg:
        # aten ops only: raw CUDA kernel rows (e.g. cuDNN's "_5x_cudnn_...") are children of aten ops,
        # so counting them too would double-count their time
        # torch.compile's fused kernels show up as top-level "triton_*" rows, not aten ops
        if not (e.key.startswith("aten::") or e.key.startswith("triton_")):
            continue
        t = self_dev(e) / 1000 / steps
        if t <= 0:
            continue
        cats[categorize(e.key)] = cats.get(categorize(e.key), 0) + t
        top.append((e.key, round(t, 2)))
    top.sort(key=lambda x: -x[1])
    return phases, dict(sorted(cats.items(), key=lambda x: -x[1])), top[:25]


def train(images, instance_prompt, steps, cfg: Config, out_dir, log_every=25, components=None):
    """Full training run. Returns (lora_path, {step: seconds since start}) for time-to-LoRA."""
    trainer = Trainer(images, instance_prompt, cfg, components)
    t0, times = time.perf_counter(), {}
    for s in range(1, steps + 1):
        trainer.step()  # with batch_size > 1, `steps` counts optimizer steps, not images
        if s % log_every == 0 or s == steps:
            torch.cuda.synchronize()
            times[s] = round(time.perf_counter() - t0, 1)
            print(f"[train_fast] {s}/{steps}  {times[s] / s:.3f} s/step", flush=True)
    return trainer.save(out_dir), times, trainer

