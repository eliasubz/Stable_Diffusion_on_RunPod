"""Run 4: auto pet descriptions + modern step-distillation LoRAs (week 3).

Run 3 showed fast samplers dilute the pet LoRA: the scene's "default cat" wins (fluffy Persian king,
bicolor pirate). Two fixes are tested here:
  1. A vision-language model (Qwen2-VL-2B) describes the pet from its photos once ("grey British
     shorthair, orange eyes, ..."), and the description is added to every prompt.
  2. Newer distillation LoRAs (Hyper-SD, DMD2) next to LCM and Lightning.
Every method runs with and without the description. Besides mean identity, the worst image and the
failure rate (identity < FAIL_BELOW) are reported, since bad cases matter most in an app.

Runs unattended (Kaggle script kernel or any 16GB+ GPU) and writes results as it goes.
"""
import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

# ---- experiment config -------------------------------------------------------------------------
PET_CLASS = "cat"
SUBJECT = "sks cat"
HOLDOUT = {"Boss4.jpeg", "Boss8.jpeg"}
PET_LORA = "boss-step250.safetensors"
PET_LORA_SCALE = 0.9
SCENES = [
    "as a pirate captain on a ship, wearing a tricorn hat, funny, cinematic",
    "as an astronaut on the moon, wearing a space helmet, cinematic",
    "as a chef in a busy kitchen, wearing a chef hat, funny",
    "sitting on a throne like a king, wearing a crown, oil painting",
]
SEEDS = [1, 2, 3]
SIZE = 1024
NEGATIVE = "blurry, deformed, low quality"
FAIL_BELOW = 0.4  # identity (DINO crop) below this looks like a different cat

VLM_ID = "Qwen/Qwen2-VL-2B-Instruct"
VLM_PROMPT = (
    "These photos all show the same cat. Describe only the cat's appearance for an image generation "
    "prompt: likely breed, coat color and pattern, fur length, eye color and build. Ignore the background. "
    "Answer with one short comma-separated phrase of at most 15 words, without mentioning the photos."
)
FALLBACK_DESCRIPTION = None  # set to a string to skip the VLM

# method: (speed LoRA or None, scheduler, steps, guidance, fixed timesteps or None)
METHODS = {
    "base25": (None, "euler", 25, 6.0, None),
    "lcm8": ("lcm", "lcm", 8, 1.0, None),
    "lightning4": ("lightning4", "trailing", 4, 0.0, None),
    "hyper8": ("hyper8", "ddim_trailing", 8, 0.0, None),
    "dmd2_4": ("dmd2", "lcm", 4, 0.0, [999, 749, 499, 249]),  # timesteps from the DMD2 README
}
# repo, substrings the file name must contain (resolved against the repo listing, fp16 preferred)
SPEED_LORAS = {
    "lcm": ("latent-consistency/lcm-lora-sdxl", ["pytorch_lora_weights.safetensors"]),
    "lightning4": ("ByteDance/SDXL-Lightning", ["sdxl_lightning_4step_lora"]),
    "hyper8": ("ByteDance/Hyper-SD", ["Hyper-SDXL-8steps-lora"]),  # not the -CFG- variant
    "dmd2": ("tianweiy/DMD2", ["dmd2_sdxl_4step_lora"]),
}

OUT = os.environ.get("RUN4_OUT", "/kaggle/working/run4" if os.path.isdir("/kaggle/working") else "run4")
os.makedirs(f"{OUT}/images", exist_ok=True)


def log(msg):
    print(f"[run4 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402

gpu = torch.cuda.get_device_name(0)
log(f"GPU {gpu}, torch {torch.__version__}")
photo_dir = os.path.dirname(find("Boss1.jpeg"))
photos = sorted(glob.glob(f"{photo_dir}/*.jp*g"))
train_photos = [p for p in photos if os.path.basename(p) not in HOLDOUT]
info = {"gpu": gpu, "torch": torch.__version__, "size": SIZE, "seeds": SEEDS, "scenes": SCENES,
        "pet_lora": PET_LORA, "pet_lora_scale": PET_LORA_SCALE, "methods": METHODS, "fail_below": FAIL_BELOW}


def save_info():
    json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)


# ---- 1. describe the pet once (as the app would at upload) ----------------------------------------
def describe_pet(paths):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    t = time.perf_counter()
    model = Qwen2VLForConditionalGeneration.from_pretrained(VLM_ID, torch_dtype=torch.float16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(VLM_ID, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    load_s = time.perf_counter() - t

    images = [Image.open(p).convert("RGB") for p in paths]
    messages = [{"role": "user", "content": [*({"type": "image"} for _ in images), {"type": "text", "text": VLM_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to("cuda")
    t = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    gen_s = time.perf_counter() - t
    raw = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    del model, processor, inputs, out
    gc.collect()
    torch.cuda.empty_cache()
    # Keep it prompt-shaped: one line, no trailing period, no leading "the cat is"
    desc = re.sub(r"^(the |this )?cat (is|has|appears to be)\s*", "", raw.strip().splitlines()[0], flags=re.I)
    desc = " ".join(desc.rstrip(".").split()[:20])
    return desc, raw, round(load_s, 1), round(gen_s, 2)


if FALLBACK_DESCRIPTION:
    description = FALLBACK_DESCRIPTION
    info["description"] = {"text": description, "source": "manual"}
else:
    try:
        description, raw, load_s, gen_s = describe_pet(train_photos[:4])
        info["description"] = {"text": description, "raw": raw, "source": VLM_ID, "load_s": load_s, "generate_s": gen_s}
    except Exception:
        log(f"VLM failed, falling back to a manual description\n{traceback.format_exc()}")
        description = "grey British shorthair, orange eyes, short dense fur, stocky build"
        info["description"] = {"text": description, "source": "manual fallback"}
log(f"pet description: {description!r}")
save_info()

# ---- 2. load SDXL ---------------------------------------------------------------------------------
from diffusers import (  # noqa: E402
    AutoencoderKL,
    DDIMScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    StableDiffusionXLPipeline,
)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipe.set_progress_bar_config(disable=True)
base_cfg = pipe.scheduler.config
SCHEDULERS = {
    "euler": lambda: EulerDiscreteScheduler.from_config(base_cfg),
    "lcm": lambda: LCMScheduler.from_config(base_cfg),
    "trailing": lambda: EulerDiscreteScheduler.from_config(base_cfg, timestep_spacing="trailing"),  # Lightning
    "ddim_trailing": lambda: DDIMScheduler.from_config(base_cfg, timestep_spacing="trailing"),  # Hyper-SD
}
pet_lora_path = find(PET_LORA)
repo_files = {}


def resolve_lora(key):
    repo, parts = SPEED_LORAS[key]
    if repo not in repo_files:
        repo_files[repo] = HfApi().list_repo_files(repo)
    hits = [f for f in repo_files[repo] if f.endswith(".safetensors") and all(p in f for p in parts)]
    if not hits:
        raise FileNotFoundError(f"no file matching {parts} in {repo}")
    hits.sort(key=lambda f: ("fp16" not in f, len(f)))  # prefer fp16, then the plainest name
    info.setdefault("speed_lora_files", {})[key] = f"{repo}/{hits[0]}"
    return hf_hub_download(repo, hits[0])


def set_loras(speed):
    pipe.unfuse_lora()
    pipe.unload_lora_weights()
    pipe.load_lora_weights(pet_lora_path, adapter_name="pet")
    names, weights = ["pet"], [PET_LORA_SCALE]
    if speed:
        pipe.load_lora_weights(resolve_lora(speed), adapter_name=speed)
        names.append(speed)
        weights.append(1.0)
    pipe.set_adapters(names, adapter_weights=weights)
    pipe.fuse_lora(adapter_names=names)


def generate(prompt, steps, guidance, timesteps, seed):
    g = torch.Generator("cuda").manual_seed(seed)
    kwargs = {"timesteps": timesteps} if timesteps else {"num_inference_steps": steps}
    torch.cuda.synchronize()
    t = time.perf_counter()
    img = pipe(prompt=prompt, negative_prompt=NEGATIVE if guidance > 1 else None, guidance_scale=guidance,
               width=SIZE, height=SIZE, generator=g, **kwargs).images[0]
    torch.cuda.synchronize()
    return img, (time.perf_counter() - t) * 1000


# ---- 3. generate ----------------------------------------------------------------------------------
rows, cfg_info = [], {}
for method, (speed, sched, steps, guidance, timesteps) in METHODS.items():
    try:
        set_loras(speed)
        pipe.scheduler = SCHEDULERS[sched]()
    except Exception:
        log(f"FAILED to set up {method}\n{traceback.format_exc()}")
        continue
    for with_desc in (False, True):
        name = f"{method}+desc" if with_desc else method
        subject = f"{SUBJECT}, {description}," if with_desc else SUBJECT
        try:
            torch.cuda.reset_peak_memory_stats()
            generate(f"a {subject} {SCENES[0]}", steps, guidance, timesteps, seed=0)  # warm-up, not recorded
            for i, scene in enumerate(SCENES):
                for seed in SEEDS:
                    img, ms = generate(f"a {subject} {scene}", steps, guidance, timesteps, seed)
                    path = f"{OUT}/images/{name}_{i}_{seed}.png"
                    img.save(path)
                    # CLIP-T is scored against the plain scene prompt, so the description can't inflate it
                    rows.append({"config": name, "method": method, "desc": with_desc, "scene": i, "seed": seed,
                                 "prompt": f"a {PET_CLASS} {scene}", "path": path, "latency_ms": ms})
            cfg_info[name] = {"peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            lat = [r["latency_ms"] for r in rows if r["config"] == name]
            log(f"{name}: p50 {np.median(lat):.0f} ms")
        except Exception:
            log(f"FAILED {name}\n{traceback.format_exc()}")
        json.dump(rows, open(f"{OUT}/generations.json", "w"), indent=1)
        info["per_config"] = cfg_info
        save_info()

# ---- 4. score -------------------------------------------------------------------------------------
import pandas as pd  # noqa: E402

del pipe
gc.collect()
torch.cuda.empty_cache()

from eval.identity import IdentityScorer  # noqa: E402

scorer = IdentityScorer(crop_label=f"a photo of a {PET_CLASS}")
scorer.set_references([Image.open(f"{photo_dir}/{p}").convert("RGB") for p in sorted(HOLDOUT)])
for r in rows:
    r.update(scorer.score(Image.open(r["path"]).convert("RGB"), r["prompt"]))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/results.csv", index=False)

g = df.groupby("config", sort=False)
summary = pd.DataFrame({
    "method": g.method.first(), "desc": g.desc.first(),
    "steps": [METHODS[m][2] for m in g.method.first()],
    "p50_ms": g.latency_ms.median(),
    "dino_crop_mean": g.dino_crop.mean(), "dino_crop_std": g.dino_crop.std(),
    "dino_crop_min": g.dino_crop.min(),
    "fail_rate": g.dino_crop.apply(lambda s: (s < FAIL_BELOW).mean()),
    "clip_t_mean": g.clip_t.mean(), "clip_t_std": g.clip_t.std(),
    "n": g.size(),
}).round(4)
if "base25" in summary.index:
    summary["speedup"] = (summary.loc["base25", "p50_ms"] / summary.p50_ms).round(2)
summary["peak_mem_gb"] = [cfg_info.get(c, {}).get("peak_mem_gb") for c in summary.index]
summary.to_csv(f"{OUT}/summary.csv")
# Per-scene identity: shows which scenes still fail
df.pivot_table(index="config", columns="scene", values="dino_crop", aggfunc="mean", sort=False).round(3).to_csv(f"{OUT}/per_scene.csv")
log("summary:\n" + summary.to_string())

# ---- 5. plots -------------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (metric, label) in zip(axes, [("dino_crop_mean", "identity: mean DINO (pet crop, held-out) →"),
                                      ("fail_rate", "← failure rate (identity < %.1f)" % FAIL_BELOW)]):
    for method in METHODS:
        pair = summary[summary.method == method]
        if pair.empty:
            continue
        x, y = pair.p50_ms / 1000, pair[metric]
        ax.plot(x, y, "-", color="#bbbbbb", zorder=1)
        ax.scatter(x[~pair.desc], y[~pair.desc], color="#888888", zorder=2, label="plain prompt" if method == "base25" else None)
        ax.scatter(x[pair.desc], y[pair.desc], color="#1f77b4", zorder=3, label="+ auto description" if method == "base25" else None)
        ax.annotate(method, (x.iloc[-1], y.iloc[-1]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("p50 latency per image, seconds (log) ←")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
fig.suptitle(f"Fast samplers + pet LoRA, with and without an auto pet description ({gpu}, {SIZE}px)")
plt.tight_layout()
plt.savefig(f"{OUT}/speed_quality.png", dpi=120)

names = list(summary.index)
fig, axes = plt.subplots(len(names), len(SCENES), figsize=(3 * len(SCENES), 3 * len(names)))
for r, c in enumerate(names):
    for s in range(len(SCENES)):
        f = f"{OUT}/images/{c}_{s}_{SEEDS[1]}.png"
        if os.path.exists(f):
            axes[r, s].imshow(Image.open(f))
        axes[r, s].axis("off")
    axes[r, 0].set_title(f"{c} ({summary.loc[c, 'p50_ms'] / 1000:.1f} s)", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/grid.png", dpi=50)

info["total_min"] = round((time.perf_counter() - t_run) / 60, 1)
save_info()
log(f"done in {info['total_min']} min")
