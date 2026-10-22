"""Run 3: inference speed vs. quality for personalized SDXL (week 3).

Starts from run 2's winner (Boss's 250-step DreamBooth-LoRA) and measures what each speedup costs:
LoRA fusing, fewer steps, step-distillation LoRAs (LCM, SDXL-Lightning) stacked on the pet LoRA,
and torch.compile. Latency is measured warm (one warm-up image per config is excluded). Identity
is scored on held-out photos (cropped to the pet), prompt adherence with CLIP-T, as in run 2.

Runs unattended (Kaggle script kernel or any 16GB+ GPU) and writes results as it goes.
"""
import gc
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

# ---- experiment config -------------------------------------------------------------------------
PET_CLASS = "cat"
SUBJECT = "sks cat"
HOLDOUT = {"Boss4.jpeg", "Boss8.jpeg"}
PET_LORA = "boss-step250.safetensors"  # run 2: identity plateaus at 200-300 steps; 250 picked from the images
PET_LORA_SCALE = 0.9
SCENES = [
    "as a pirate captain on a ship, wearing a tricorn hat, funny, cinematic",
    "as an astronaut on the moon, wearing a space helmet, cinematic",
    "as a chef in a busy kitchen, wearing a chef hat, funny",
    "sitting on a throne like a king, wearing a crown, oil painting",
]
SEEDS = [1, 2, 3]
SIZE = 1024  # production resolution (and what the distillation LoRAs were trained at)
NEGATIVE = "blurry, deformed, low quality"

# name: (speed LoRA or None, scheduler, steps, guidance, fuse, compile)
CONFIGS = {
    "base25": (None, "euler", 25, 6.0, False, False),
    "base25_fused": (None, "euler", 25, 6.0, True, False),
    "base12_fused": (None, "euler", 12, 6.0, True, False),
    "lcm4": ("lcm", "lcm", 4, 1.0, True, False),
    "lcm8": ("lcm", "lcm", 8, 1.0, True, False),
    "lightning4": ("lightning4", "trailing", 4, 0.0, True, False),
    "lightning8": ("lightning8", "trailing", 8, 0.0, True, False),
    "lightning4_compiled": ("lightning4", "trailing", 4, 0.0, True, True),  # keep last: compile mutates the UNet
}
SPEED_LORAS = {
    "lcm": ("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors"),
    "lightning4": ("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors"),
    "lightning8": ("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_lora.safetensors"),
}

OUT = os.environ.get("RUN3_OUT", "/kaggle/working/run3" if os.path.isdir("/kaggle/working") else "run3")
os.makedirs(f"{OUT}/images", exist_ok=True)


def log(msg):
    print(f"[run3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find(name):
    for root in ["/kaggle/input", os.path.dirname(os.path.abspath(__file__))]:
        hits = glob.glob(f"{root}/**/{name}", recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


# ---- setup ---------------------------------------------------------------------------------------
t_run = time.perf_counter()
WORK = "/tmp/worker"
shutil.copytree(os.path.dirname(find("train.py")), WORK, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("test_images", "*.ipynb"))
sys.path.insert(0, WORK)
os.chdir(WORK)
log("installing pinned requirements")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import (  # noqa: E402
    AutoencoderKL,
    EulerDiscreteScheduler,
    LCMScheduler,
    StableDiffusionXLPipeline,
)
from huggingface_hub import hf_hub_download  # noqa: E402

gpu = torch.cuda.get_device_name(0)
log(f"GPU {gpu}, torch {torch.__version__}")
info = {"gpu": gpu, "torch": torch.__version__, "size": SIZE, "seeds": SEEDS, "scenes": SCENES,
        "pet_lora": PET_LORA, "pet_lora_scale": PET_LORA_SCALE, "configs": CONFIGS}

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipe.set_progress_bar_config(disable=True)
base_scheduler_config = pipe.scheduler.config
SCHEDULERS = {
    "euler": lambda: EulerDiscreteScheduler.from_config(base_scheduler_config),
    "lcm": lambda: LCMScheduler.from_config(base_scheduler_config),
    # SDXL-Lightning expects trailing timesteps
    "trailing": lambda: EulerDiscreteScheduler.from_config(base_scheduler_config, timestep_spacing="trailing"),
}
pet_lora_path = find(PET_LORA)


def set_loras(speed):
    """Load the pet LoRA (+ optional speed LoRA) fresh, so fusing never leaks between configs."""
    pipe.unfuse_lora()
    pipe.unload_lora_weights()
    pipe.load_lora_weights(pet_lora_path, adapter_name="pet")
    names, weights = ["pet"], [PET_LORA_SCALE]
    if speed:
        repo, fname = SPEED_LORAS[speed]
        pipe.load_lora_weights(hf_hub_download(repo, fname), adapter_name=speed)
        names.append(speed)
        weights.append(1.0)
    pipe.set_adapters(names, adapter_weights=weights)
    return names


def generate(prompt, steps, guidance, seed):
    g = torch.Generator("cuda").manual_seed(seed)
    torch.cuda.synchronize()
    t = time.perf_counter()
    img = pipe(prompt=prompt, negative_prompt=NEGATIVE if guidance > 1 else None, num_inference_steps=steps,
               guidance_scale=guidance, width=SIZE, height=SIZE, generator=g).images[0]
    torch.cuda.synchronize()
    return img, (time.perf_counter() - t) * 1000


# ---- 1. generate -------------------------------------------------------------------------------------
rows, cfg_info = [], {}
for name, (speed, sched, steps, guidance, fuse, compile_unet) in CONFIGS.items():
    try:
        names = set_loras(speed)
        if fuse:
            pipe.fuse_lora(adapter_names=names)
        pipe.scheduler = SCHEDULERS[sched]()
        compile_s = 0.0
        if compile_unet:
            pipe.unet = torch.compile(pipe.unet)
        torch.cuda.reset_peak_memory_stats()

        # Warm-up (not recorded). With torch.compile this is where compilation happens.
        _, warm_ms = generate(f"a {SUBJECT} {SCENES[0]}", steps, guidance, seed=0)
        if compile_unet:
            compile_s = warm_ms / 1000
        for i, scene in enumerate(SCENES):
            for seed in SEEDS:
                img, ms = generate(f"a {SUBJECT} {scene}", steps, guidance, seed)
                path = f"{OUT}/images/{name}_{i}_{seed}.png"
                img.save(path)
                rows.append({"config": name, "scene": i, "seed": seed, "prompt": f"a {PET_CLASS} {scene}",
                             "path": path, "latency_ms": ms})
        cfg_info[name] = {"peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                          "warmup_s": round(warm_ms / 1000, 1), "compile_s": round(compile_s, 1)}
        lat = [r["latency_ms"] for r in rows if r["config"] == name]
        log(f"{name}: p50 {np.median(lat):.0f} ms, peak {cfg_info[name]['peak_mem_gb']} GB, warm-up {warm_ms / 1000:.1f} s")
    except Exception:
        log(f"FAILED {name}\n{traceback.format_exc()}")
    json.dump(rows, open(f"{OUT}/generations.json", "w"), indent=1)
    info["per_config"] = cfg_info
    json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)

# ---- 2. score ----------------------------------------------------------------------------------------
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

del pipe
gc.collect()
torch.cuda.empty_cache()

from eval.identity import IdentityScorer  # noqa: E402

scorer = IdentityScorer(crop_label=f"a photo of a {PET_CLASS}")
photo_dir = os.path.dirname(find("Boss1.jpeg"))
scorer.set_references([Image.open(f"{photo_dir}/{p}").convert("RGB") for p in sorted(HOLDOUT)])
for r in rows:
    r.update(scorer.score(Image.open(r["path"]).convert("RGB"), r["prompt"]))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/results.csv", index=False)

g = df.groupby("config", sort=False)
summary = pd.DataFrame({
    "steps": [CONFIGS[c][2] for c in g.groups],
    "p50_ms": g.latency_ms.median(),
    "p95_ms": g.latency_ms.quantile(0.95),
    "dino_crop_mean": g.dino_crop.mean(), "dino_crop_std": g.dino_crop.std(),
    "clip_t_mean": g.clip_t.mean(), "clip_t_std": g.clip_t.std(),
    "n": g.size(),
}).round(4)
summary["speedup"] = (summary.loc["base25", "p50_ms"] / summary.p50_ms).round(2) if "base25" in summary.index else float("nan")
for k in ["peak_mem_gb", "compile_s"]:
    summary[k] = [cfg_info.get(c, {}).get(k) for c in summary.index]
summary.to_csv(f"{OUT}/summary.csv")
log("summary:\n" + summary.to_string())

# ---- 3. plots ----------------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric, label in [(axes[0], "dino_crop", "identity (DINO, pet crop, held-out)"),
                          (axes[1], "clip_t", "prompt adherence (CLIP-T)")]:
    sem = summary[f"{metric}_std"] / summary.n ** 0.5
    ax.errorbar(summary.p50_ms / 1000, summary[f"{metric}_mean"], yerr=sem, fmt="o", capsize=3)
    for c, r in summary.iterrows():
        ax.annotate(c, (r.p50_ms / 1000, r[f"{metric}_mean"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("p50 latency per image, seconds (log) ←")
    ax.set_ylabel(f"{label} →")
    ax.grid(alpha=0.25)
fig.suptitle(f"SDXL + pet LoRA: speed vs. quality ({gpu}, {SIZE}px)")
plt.tight_layout()
plt.savefig(f"{OUT}/speed_quality.png", dpi=120)

names = list(summary.index)
fig, axes = plt.subplots(len(names), len(SCENES), figsize=(3 * len(SCENES), 3 * len(names)))
for r, c in enumerate(names):
    for s in range(len(SCENES)):
        f = f"{OUT}/images/{c}_{s}_{SEEDS[0]}.png"
        if os.path.exists(f):
            axes[r, s].imshow(Image.open(f))
        axes[r, s].axis("off")
    axes[r, 0].set_title(f"{c} ({summary.loc[c, 'p50_ms'] / 1000:.1f} s)", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/grid.png", dpi=50)

info["total_min"] = round((time.perf_counter() - t_run) / 60, 1)
json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)
log(f"done in {info['total_min']} min")
