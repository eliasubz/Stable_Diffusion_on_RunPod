"""Run 7: faster training steps (cast fix, torch.compile) and time-to-plateau (batch size, LR, resolution).

Part 1, step speed (768px, batch 1): run 6's fastest config, + the cast fix (fp16 LoRA weights with
fp32 masters), + torch.compile. Compile time is measured separately. Then torch.profiler again on
the fastest one.

Part 2, time to plateau: the metric that matters is seconds of training until the LoRA produces
Boss, not ms per step. Each config trains up to 300 images seen, saving the LoRA at 100/200/300
images. Every snapshot then generates the same images (Euler 25 steps, 768px, no description, as in
run 2), scored on held-out photos. Configs vary batch size, learning rate and training resolution.

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
SCENES = [
    "as a pirate captain on a ship, wearing a tricorn hat, funny, cinematic",
    "as an astronaut on the moon, wearing a space helmet, cinematic",
    "as a chef in a busy kitchen, wearing a chef hat, funny",
    "sitting on a throne like a king, wearing a crown, oil painting",
]
EVAL_SEED = 1
EVAL_SIZE, EVAL_STEPS = 768, 25
PLATEAU = 0.60  # run 2: identity (DINO crop, held-out) plateaus at ~0.62 with Euler-25 at 768px
SNAPSHOT_IMAGES = [100, 200, 300]

# Part 1: Config overrides at 768px, batch 1 (cached latents + text, fused AdamW, no grad ckpt = run 6's best)
FAST = dict(cache_latents=True, cache_text=True, optimizer="adamw_fused", grad_ckpt=False)
STEP_BENCH = {
    "run6_fastest": dict(FAST),
    "+cast_fix": dict(FAST, fp16_lora=True),
    "+cast_fix+compile": dict(FAST, fp16_lora=True, compile=True),
}
# Part 2: (resolution, batch size, learning rate); grad checkpointing is turned on only if needed
SWEEP = {
    "768_bs1_lr1e-4": (768, 1, 1e-4),
    "768_bs1_lr2e-4": (768, 1, 2e-4),
    "768_bs1_lr4e-4": (768, 1, 4e-4),
    "768_bs2_lr2e-4": (768, 2, 2e-4),
    "768_bs4_lr4e-4": (768, 4, 4e-4),
    "512_bs1_lr1e-4": (512, 1, 1e-4),
    "512_bs1_lr2e-4": (512, 1, 2e-4),
    "512_bs4_lr4e-4": (512, 4, 4e-4),
}

OUT = os.environ.get("RUN7_OUT", "/kaggle/working/run7" if os.path.isdir("/kaggle/working") else "run7")
LORA_TMP = "/tmp/run7_loras"
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(LORA_TMP, exist_ok=True)


def log(msg):
    print(f"[run7 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import train_fast as tf  # noqa: E402

gpu = torch.cuda.get_device_name(0)
photo_dir = os.path.dirname(find("Boss1.jpeg"))
train_photos = [p for p in sorted(glob.glob(f"{photo_dir}/*.jp*g")) if os.path.basename(p) not in HOLDOUT]
images = [Image.open(p).convert("RGB") for p in train_photos]
info = {"gpu": gpu, "torch": torch.__version__, "plateau": PLATEAU, "step_bench": {}, "sweep": {}}
log(f"GPU {gpu}, torch {torch.__version__}, {len(images)} training photos")


def save_info():
    json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)


def free():
    gc.collect()
    torch.cuda.empty_cache()


comps = tf.load_components(fp32_vae=True)


def fresh_unet():
    comps.pop("unet", None)
    free()
    torch._dynamo.reset()
    comps["unet"] = tf.UNet2DConditionModel.from_pretrained(tf.MODEL_ID, subfolder="unet", variant="fp16",
                                                            torch_dtype=torch.float16).to("cuda")


# ---- part 1: step speed ---------------------------------------------------------------------------
for name, overrides in STEP_BENCH.items():
    trainer = None
    try:
        fresh_unet()
        trainer = tf.Trainer(images, INSTANCE_PROMPT, tf.Config(resolution=768, **overrides), comps)
        torch.cuda.synchronize()
        t = time.perf_counter()
        trainer.step()  # first step: includes compilation with torch.compile
        torch.cuda.synchronize()
        first_s = time.perf_counter() - t
        res = tf.benchmark(trainer, steps=20, warmup=5)
        res.update({"first_step_s": round(first_s, 1), **{f"cfg_{k}": v for k, v in overrides.items()}})
        info["step_bench"][name] = res
        log(f"{name}: {res['total']:.0f} ms/step (fwd {res['forward']:.0f}, bwd {res['backward']:.0f}, opt {res['optimizer']:.1f}), "
            f"peak {res['peak_mem_gb']:.1f} GB, first step {first_s:.1f} s")
    except Exception:
        info["step_bench"][name] = {"error": traceback.format_exc()[-1500:]}
        log(f"FAILED {name}\n{traceback.format_exc()}")
    trainer = None
    free()
    save_info()

ok = {k: v for k, v in info["step_bench"].items() if "total" in v}
best_step = min(ok, key=lambda k: ok[k]["total"]) if ok else None
info["best_step_config"] = best_step

# Profile the fastest step config, and the cast fix alone if that isn't the fastest (to see the casts go away)
for name in dict.fromkeys([n for n in [best_step, "+cast_fix"] if n in ok]):
    trainer = None
    try:
        fresh_unet()
        trainer = tf.Trainer(images, INSTANCE_PROMPT, tf.Config(resolution=768, **STEP_BENCH[name]), comps)
        _, cats, top = tf.profile_step(trainer, steps=3, warmup=5, trace_path=f"{OUT}/trace_{name.strip('+')}.json")
        info.setdefault("profiles", {})[name] = {"categories_ms": cats, "top_ops_ms": top}
        log(f"profile {name}: {({k: round(v) for k, v in cats.items()})}")
    except Exception:
        log(f"FAILED profiling {name}\n{traceback.format_exc()}")
    trainer = None
    free()
    save_info()

# ---- part 2: time to plateau ------------------------------------------------------------------------
SWEEP_BASE = dict(FAST, fp16_lora="+cast_fix" in ok)


def run_sweep_config(name, res, bs, lr):
    """Train with snapshots; retry with gradient checkpointing if it doesn't fit."""
    for grad_ckpt in (False, True):
        trainer = None
        try:
            fresh_unet()
            cfg = tf.Config(**dict(SWEEP_BASE, grad_ckpt=grad_ckpt), resolution=res, batch_size=bs, learning_rate=lr)
            trainer = tf.Trainer(images, INSTANCE_PROMPT, cfg, comps)
            torch.cuda.synchronize()
            t0, snaps, step = time.perf_counter(), {}, 0
            for n_img in SNAPSHOT_IMAGES:
                while step * bs < n_img:
                    trainer.step()
                    step += 1
                torch.cuda.synchronize()
                train_s = time.perf_counter() - t0
                path = trainer.save(LORA_TMP, weight_name=f"{name}_img{n_img}.safetensors")
                snaps[n_img] = {"train_s": round(train_s, 1), "steps": step, "path": path}
                t0 += time.perf_counter() - t0 - train_s  # don't count saving as training time
            peak = torch.cuda.max_memory_allocated() / 1e9
            trainer = None
            free()
            return {"grad_ckpt": grad_ckpt, "peak_mem_gb": round(peak, 1), "snapshots": snaps}
        except torch.OutOfMemoryError:
            log(f"{name}: out of memory with grad_ckpt={grad_ckpt}")
            trainer = None
            free()
    return {"error": "out of memory even with gradient checkpointing"}


for name, (res, bs, lr) in SWEEP.items():
    try:
        torch.cuda.reset_peak_memory_stats()
        out = run_sweep_config(name, res, bs, lr)
        out.update({"resolution": res, "batch_size": bs, "learning_rate": lr})
        info["sweep"][name] = out
        if "snapshots" in out:
            s = out["snapshots"]
            log(f"{name}: " + ", ".join(f"{n} img in {v['train_s']:.0f} s" for n, v in s.items())
                + f" (grad_ckpt={out['grad_ckpt']}, peak {out['peak_mem_gb']} GB)")
    except Exception:
        info["sweep"][name] = {"error": traceback.format_exc()[-1500:]}
        log(f"FAILED {name}\n{traceback.format_exc()}")
    save_info()

comps.clear()
free()

# ---- evaluate every snapshot with the same generation ------------------------------------------------
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline  # noqa: E402

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(tf.MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16",
                                                 use_safetensors=True).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
rows = []
for name, out in info["sweep"].items():
    for n_img, snap in out.get("snapshots", {}).items():
        try:
            pipe.unload_lora_weights()
            pipe.load_lora_weights(snap["path"], adapter_name="pet")
            pipe.set_adapters(["pet"], adapter_weights=[0.9])
            for i, scene in enumerate(SCENES):
                img = pipe(prompt=f"a sks cat {scene}", negative_prompt="blurry, deformed, low quality",
                           num_inference_steps=EVAL_STEPS, guidance_scale=6.0, width=EVAL_SIZE, height=EVAL_SIZE,
                           generator=torch.Generator("cuda").manual_seed(EVAL_SEED)).images[0]
                p = f"{OUT}/images/{name}_img{n_img}_{i}.png"
                img.save(p)
                rows.append({"config": name, "images_seen": n_img, "train_s": snap["train_s"], "scene": i,
                             "prompt": f"a {PET_CLASS} {scene}", "path": p})
        except Exception:
            log(f"FAILED eval {name} @ {n_img}\n{traceback.format_exc()}")
    log(f"evaluated {name}")
del pipe
free()

import pandas as pd  # noqa: E402

from eval.identity import IdentityScorer  # noqa: E402

scorer = IdentityScorer(crop_label=f"a photo of a {PET_CLASS}")
scorer.set_references([Image.open(f"{photo_dir}/{p}").convert("RGB") for p in sorted(HOLDOUT)])
for r in rows:
    r.update(scorer.score(Image.open(r["path"]).convert("RGB"), r["prompt"]))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/results.csv", index=False)
curve = df.groupby(["config", "images_seen"], sort=False).agg(
    train_s=("train_s", "first"), dino_crop=("dino_crop", "mean"), dino_crop_min=("dino_crop", "min"),
    clip_t=("clip_t", "mean")).round(4).reset_index()
curve.to_csv(f"{OUT}/curve.csv", index=False)

plateau = {}
for name, g in curve.groupby("config", sort=False):
    hit = g[g.dino_crop >= PLATEAU]
    plateau[name] = {"seconds_to_plateau": float(hit.train_s.iloc[0]) if len(hit) else None,
                     "images_to_plateau": int(hit.images_seen.iloc[0]) if len(hit) else None,
                     "best_identity": float(g.dino_crop.max())}
info["time_to_plateau"] = plateau
log("curve:\n" + curve.to_string())
log("time to plateau:\n" + json.dumps(plateau, indent=1))

# ---- plots --------------------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(10, 6))
for name, g in curve.groupby("config", sort=False):
    style = "-" if name.startswith("768") else "--"
    ax.plot(g.train_s, g.dino_crop, style, marker="o", label=name)
ax.axhline(PLATEAU, color="#999999", ls=":", lw=1, label=f"plateau ({PLATEAU})")
ax.set_xlabel("training seconds (excluding model loading) →")
ax.set_ylabel("identity: DINO crop vs held-out photos →")
ax.set_title(f"Time to plateau: batch size × learning rate × resolution ({gpu})")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"{OUT}/time_to_plateau.png", dpi=120)

finals = [n for n in SWEEP if (n, SNAPSHOT_IMAGES[-1]) in set(zip(df.config, df.images_seen))]
if finals:
    fig, axes = plt.subplots(len(finals), len(SCENES), figsize=(3 * len(SCENES), 3 * len(finals)), squeeze=False)
    for r, name in enumerate(finals):
        for c in range(len(SCENES)):
            f = f"{OUT}/images/{name}_img{SNAPSHOT_IMAGES[-1]}_{c}.png"
            if os.path.exists(f):
                axes[r, c].imshow(Image.open(f))
            axes[r, c].axis("off")
        axes[r, 0].set_title(f"{name} @ {SNAPSHOT_IMAGES[-1]} images", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/grid.png", dpi=50)

info["total_min"] = round((time.perf_counter() - t_run) / 60, 1)
save_info()
log(f"done in {info['total_min']} min")
