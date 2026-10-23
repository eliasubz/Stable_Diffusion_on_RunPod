"""Run 6: where does a DreamBooth-LoRA training step spend its time, and how much can we cut?

1. Benchmark: our training loop (train_fast.py) configured like diffusers' script, then each
   speedup added one at a time: cached VAE latents, fused AdamW, 8-bit Adam, no grad checkpointing.
   An extra "no cache" config also re-encodes the prompt every step to show what text caching saves.
2. torch.profiler on the script-equivalent config and the fastest one: GPU time per phase (UNet
   forward / backward / optimizer / VAE / text) and per op type (attention, GroupNorm, matmul, ...).
3. A full 250-step training with the fastest config (time-to-LoRA).
4. Check that the faster LoRA is as good: generate with the production setup (LCM-8 + description)
   from both the new LoRA and run 2's script-trained 250-step LoRA, same seeds, and score identity.

Runs unattended on Kaggle and writes results as it goes.
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

PET_CLASS = "cat"
INSTANCE_PROMPT = "a photo of sks cat"
HOLDOUT = {"Boss4.jpeg", "Boss8.jpeg"}
RESOLUTION = 768
TRAIN_STEPS = 250
BENCH_STEPS, BENCH_WARMUP = 20, 5
SCRIPT_LORA = "boss-step250.safetensors"  # run 2, trained with the diffusers script
DESCRIPTION = "British Shorthair, gray, short, yellow"  # run 4/5 description v1
SCENES = [
    "as a pirate captain on a ship, wearing a tricorn hat, funny, cinematic",
    "as an astronaut on the moon, wearing a space helmet, cinematic",
    "as a chef in a busy kitchen, wearing a chef hat, funny",
    "sitting on a throne like a king, wearing a crown, oil painting",
]
SEEDS = [1, 2, 3]

# name: Config overrides (resolution / rank / lr fixed across configs)
BENCH = {
    "no_cache": dict(cache_latents=False, cache_text=False, optimizer="adamw", grad_ckpt=True),
    "script_equiv": dict(cache_latents=False, cache_text=True, optimizer="adamw", grad_ckpt=True),
    "+cached_latents": dict(cache_latents=True, cache_text=True, optimizer="adamw", grad_ckpt=True),
    "+fused_adamw": dict(cache_latents=True, cache_text=True, optimizer="adamw_fused", grad_ckpt=True),
    "+8bit_adam": dict(cache_latents=True, cache_text=True, optimizer="adamw8bit", grad_ckpt=True),
    "+no_grad_ckpt": dict(cache_latents=True, cache_text=True, optimizer="adamw_fused", grad_ckpt=False),
}

OUT = os.environ.get("RUN6_OUT", "/kaggle/working/run6" if os.path.isdir("/kaggle/working") else "run6")
os.makedirs(f"{OUT}/images", exist_ok=True)


def log(msg):
    print(f"[run6 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
shutil.copy(find("train_fast.py"), WORK)
sys.path.insert(0, WORK)
os.chdir(WORK)
log("installing pinned requirements (+ latest bitsandbytes for 8-bit Adam)")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
bnb = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.46"], capture_output=True, text=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import train_fast as tf  # noqa: E402

gpu = torch.cuda.get_device_name(0)
photo_dir = os.path.dirname(find("Boss1.jpeg"))
train_photos = [p for p in sorted(glob.glob(f"{photo_dir}/*.jp*g")) if os.path.basename(p) not in HOLDOUT]
images = [Image.open(p).convert("RGB") for p in train_photos]
info = {"gpu": gpu, "torch": torch.__version__, "resolution": RESOLUTION, "rank": 8, "train_photos": len(images),
        "bitsandbytes_install": "ok" if bnb.returncode == 0 else bnb.stderr[-500:], "bench": {}, "profiles": {}}
log(f"GPU {gpu}, torch {torch.__version__}, {len(images)} training photos")


def save_info():
    json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)


def free():
    """Release cached GPU memory. Callers must `del` their own references first."""
    gc.collect()
    torch.cuda.empty_cache()


t = time.perf_counter()
comps = tf.load_components(fp32_vae=True)
info["load_components_s"] = round(time.perf_counter() - t, 1)


def fresh_unet():
    """A clean UNet per config, so LoRA / checkpointing state never leaks between configs."""
    comps.pop("unet", None)
    gc.collect()
    torch.cuda.empty_cache()
    comps["unet"] = tf.UNet2DConditionModel.from_pretrained(tf.MODEL_ID, subfolder="unet", variant="fp16",
                                                            torch_dtype=torch.float16).to("cuda")


# ---- 1. benchmark each speedup ----------------------------------------------------------------------
for name, overrides in BENCH.items():
    try:
        fresh_unet()
        cfg = tf.Config(resolution=RESOLUTION, **overrides)
        trainer = tf.Trainer(images, INSTANCE_PROMPT, cfg, comps)
        res = tf.benchmark(trainer, steps=BENCH_STEPS, warmup=BENCH_WARMUP)
        # Config keys are prefixed: "optimizer" is already the optimizer phase's ms
        res.update({"optimizer_used": trainer.optimizer_name, "trainable_params": trainer.n_params,
                    **{f"cfg_{k}": v for k, v in overrides.items()}})
        info["bench"][name] = res
        log(f"{name}: {res['total']:.0f} ms/step (vae {res['vae']:.0f}, text {res['text']:.0f}, fwd {res['forward']:.0f}, "
            f"bwd {res['backward']:.0f}, opt {res['optimizer']:.1f}) peak {res['peak_mem_gb']:.1f} GB [{trainer.optimizer_name}]")
        del trainer
        free()
    except torch.OutOfMemoryError:
        info["bench"][name] = {"error": "out of memory", **{f"cfg_{k}": v for k, v in overrides.items()}}
        log(f"{name}: OUT OF MEMORY")
        trainer = None  # drop the half-built trainer so its memory doesn't leak into the next config
        free()
    except Exception:
        info["bench"][name] = {"error": traceback.format_exc()[-800:], **{f"cfg_{k}": v for k, v in overrides.items()}}
        log(f"FAILED {name}\n{traceback.format_exc()}")
        trainer = None
        free()
    save_info()

ok = {k: v for k, v in info["bench"].items() if "total" in v}
if not ok:
    raise SystemExit("no benchmark config succeeded; see run_info.json")
fastest = min(ok, key=lambda k: ok[k]["total"])
base = ok.get("script_equiv", {}).get("total")
info["fastest"] = fastest
info["speedup_vs_script"] = round(base / ok[fastest]["total"], 2) if base else None
log(f"fastest: {fastest} ({ok[fastest]['total']:.0f} ms/step, {info['speedup_vs_script']}x vs script_equiv)")
save_info()

# ---- 2. profile the script-equivalent and the fastest config -----------------------------------------
for name in dict.fromkeys(["script_equiv", fastest]):
    try:
        fresh_unet()
        trainer = tf.Trainer(images, INSTANCE_PROMPT, tf.Config(resolution=RESOLUTION, **BENCH[name]), comps)
        phases, cats, top = tf.profile_step(trainer, steps=3, warmup=5, trace_path=f"{OUT}/trace_{name.strip('+')}.json")
        info["profiles"][name] = {"phases_ms": phases, "categories_ms": cats, "top_ops_ms": top}
        log(f"profile {name}: phases {({k: round(v) for k, v in phases.items()})}")
        log(f"profile {name}: op categories {({k: round(v) for k, v in cats.items()})}")
        del trainer
        free()
    except Exception:
        log(f"FAILED profiling {name}\n{traceback.format_exc()}")
        trainer = None
        free()
    save_info()

# ---- 3. full training with the fastest config ---------------------------------------------------------
fresh_unet()
fast_cfg = tf.Config(resolution=RESOLUTION, **BENCH[fastest])
t = time.perf_counter()
fast_lora, step_times, trainer = tf.train(images, INSTANCE_PROMPT, TRAIN_STEPS, fast_cfg, f"{OUT}/lora_fast", components=comps)
info["full_train"] = {"config": fastest, "steps": TRAIN_STEPS, "total_s": round(time.perf_counter() - t, 1),
                      "step_times_s": step_times, "s_per_step": round(step_times[TRAIN_STEPS] / TRAIN_STEPS, 3),
                      "note": "excludes model loading (load_components_s); run 2's script took 86 s + 1.76 s/step"}
log(f"full training: {TRAIN_STEPS} steps in {step_times[TRAIN_STEPS]:.0f} s")
save_info()
del trainer
comps.clear()
free()

# ---- 4. same production generation from both LoRAs ----------------------------------------------------
from diffusers import AutoencoderKL, LCMScheduler, StableDiffusionXLPipeline  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(tf.MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16",
                                                 use_safetensors=True).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
lcm = hf_hub_download("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors")
rows = []
for name, path in [("script_lora", find(SCRIPT_LORA)), ("fast_lora", fast_lora)]:
    pipe.unfuse_lora()
    pipe.unload_lora_weights()
    pipe.load_lora_weights(path, adapter_name="pet")
    pipe.load_lora_weights(lcm, adapter_name="lcm")
    pipe.set_adapters(["pet", "lcm"], adapter_weights=[0.9, 1.0])
    pipe.fuse_lora(adapter_names=["pet", "lcm"])
    for i, scene in enumerate(SCENES):
        for seed in SEEDS:
            img = pipe(prompt=f"a sks cat, {DESCRIPTION}, {scene}", num_inference_steps=8, guidance_scale=1.0,
                       width=1024, height=1024, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            p = f"{OUT}/images/{name}_{i}_{seed}.png"
            img.save(p)
            rows.append({"lora": name, "scene": i, "seed": seed, "prompt": f"a {PET_CLASS} {scene}", "path": p})
    log(f"generated {name}")
del pipe
free()

import pandas as pd  # noqa: E402

from eval.identity import IdentityScorer  # noqa: E402

scorer = IdentityScorer(crop_label=f"a photo of a {PET_CLASS}")
scorer.set_references([Image.open(f"{photo_dir}/{p}").convert("RGB") for p in sorted(HOLDOUT)])
for r in rows:
    r.update(scorer.score(Image.open(r["path"]).convert("RGB"), r["prompt"]))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/quality.csv", index=False)
q = df.groupby("lora")[["dino_crop", "clip_t"]].agg(["mean", "std"]).round(4)
info["quality"] = json.loads(q.to_json())
log("quality (LCM-8 + description):\n" + q.to_string())

# ---- 5. tables and plots ------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

bench = pd.DataFrame(info["bench"]).T
bench.to_csv(f"{OUT}/bench.csv")
okb = bench[bench["total"].notna()] if "total" in bench else bench.iloc[0:0]
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
phase_cols = [c for c in ["vae", "text", "forward", "backward", "optimizer"] if c in okb]
left = [0] * len(okb)
for col, color in zip(phase_cols, ["#9467bd", "#8c564b", "#1f77b4", "#ff7f0e", "#2ca02c"]):
    vals = okb[col].astype(float).tolist()
    axes[0].barh(okb.index, vals, left=left, color=color, label=col)
    left = [a + b for a, b in zip(left, vals)]
for y, (n, r) in enumerate(okb.iterrows()):
    axes[0].text(float(r["total"]) + 15, y, f"{float(r['total']):.0f} ms · {float(r['peak_mem_gb']):.1f} GB", va="center", fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xlabel("ms per training step (wall clock, synced)")
axes[0].set_title(f"Training step by phase ({gpu}, {RESOLUTION}px, rank 8)")
axes[0].legend(fontsize=8, loc="lower right")
axes[0].set_xlim(0, max(left) * 1.3 if left else 1)

profs = info["profiles"]
cats = sorted({c for p in profs.values() for c in p["categories_ms"]})
w = 0.8 / max(1, len(profs))
for k, (pname, p) in enumerate(profs.items()):
    axes[1].bar([i + k * w for i in range(len(cats))], [p["categories_ms"].get(c, 0) for c in cats], width=w, label=pname)
axes[1].set_xticks([i + w * (len(profs) - 1) / 2 for i in range(len(cats))])
axes[1].set_xticklabels(cats, rotation=20, ha="right")
axes[1].set_ylabel("GPU ms per step (self time, fwd + bwd)")
axes[1].set_title("GPU time by op type (torch.profiler)")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/training_profile.png", dpi=120)

fig, axes = plt.subplots(2, len(SCENES), figsize=(3 * len(SCENES), 6.4))
for r, name in enumerate(["script_lora", "fast_lora"]):
    for s in range(len(SCENES)):
        axes[r, s].imshow(Image.open(f"{OUT}/images/{name}_{s}_{SEEDS[1]}.png"))
        axes[r, s].axis("off")
    axes[r, 0].set_title(name, loc="left", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/grid.png", dpi=60)

shutil.copy(fast_lora, f"{OUT}/boss-fast-step{TRAIN_STEPS}.safetensors")
info["total_min"] = round((time.perf_counter() - t_run) / 60, 1)
save_info()
log(f"done in {info['total_min']} min")
