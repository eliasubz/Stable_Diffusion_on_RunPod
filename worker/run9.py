"""Run 9: quantization for training. QLoRA-style frozen UNet + INT8 GEMM feasibility on a T4.

Part A, QLoRA: the frozen UNet is loaded in fp16, int8 (bitsandbytes LLM.int8) or nf4 (4-bit), and only
the LoRA trains in fp32. Measures memory and ms/step at the production training config (512px,
batch 4) and at 768px batch 1/2 without gradient checkpointing (768/bs2 didn't fit in fp16 in run 7).
Then a full production training (200 images) per base precision, with the LoRA evaluated on the normal
fp16 base (Euler 25, 768px, no description, one seed) against held-out photos.

Part B, INT8 GEMM feasibility for INT8 training: torch._int_mm vs fp16 matmul at the UNet's real linear
shapes (forward x·Wᵀ and the input-gradient g·W), with and without the cost of quantizing the
activations per token. If INT8 isn't clearly faster here, INT8 training on a T4 can't pay off.

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
PRODUCTION = dict(resolution=512, batch_size=4, learning_rate=4e-4)  # run 7: plateau in ~56 s
TRAIN_IMAGES = 200
BASES = ["fp16", "int8", "nf4"]
BENCH_SHAPES = {"512_bs4": (512, 4), "768_bs1": (768, 1), "768_bs2": (768, 2)}

OUT = os.environ.get("RUN9_OUT", "/kaggle/working/run9" if os.path.isdir("/kaggle/working") else "run9")
os.makedirs(f"{OUT}/images", exist_ok=True)


def log(msg):
    print(f"[run9 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
log("installing pinned requirements + bitsandbytes")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
bnb_install = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.46"], capture_output=True, text=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import train_fast as tf  # noqa: E402

gpu = torch.cuda.get_device_name(0)
photo_dir = os.path.dirname(find("Boss1.jpeg"))
train_photos = [p for p in sorted(glob.glob(f"{photo_dir}/*.jp*g")) if os.path.basename(p) not in HOLDOUT]
images = [Image.open(p).convert("RGB") for p in train_photos]
info = {"gpu": gpu, "torch": torch.__version__, "production": PRODUCTION, "train_images": TRAIN_IMAGES,
        "bnb_install": "ok" if bnb_install.returncode == 0 else bnb_install.stderr[-500:],
        "gemm": {}, "qlora_bench": {}, "qlora_train": {}}
log(f"GPU {gpu}, torch {torch.__version__}")


def save_info():
    json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)


def free():
    gc.collect()
    torch.cuda.empty_cache()


# ---- part B first (fast, and decides whether INT8 training is worth pursuing on this GPU) -------------
def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def quant_rows(x):
    """Symmetric per-row int8 quantization (absmax), as W8A8 does for activations."""
    scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 127
    return (x / scale).round().clamp(-127, 127).to(torch.int8), scale


# Linear shapes in SDXL's transformer blocks: tokens at the 2 attention resolutions × (in, out)
LAYERS = {"attn 640": (640, 640), "attn 1280": (1280, 1280), "ff-in 640": (640, 5120), "ff-in 1280": (1280, 10240),
          "ff-out 640": (2560, 640), "ff-out 1280": (5120, 1280), "cross k/v 640": (2048, 640)}
try:
    for res, bs in [(512, 4), (768, 1)]:
        lat = res // 8
        for level, down in [("hi-res blocks", 2), ("lo-res blocks", 4)]:
            M = bs * (lat // down) ** 2
            for lname, (K, N) in LAYERS.items():
                if "640" in lname and down == 4 or "1280" in lname and down == 2:
                    continue  # 640-ch layers live at the higher resolution, 1280-ch at the lower one
                x16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
                w16 = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.02
                g16 = torch.randn(M, N, device="cuda", dtype=torch.float16)
                x8, _ = quant_rows(x16)
                g8, _ = quant_rows(g16)
                w8t = quant_rows(w16)[0].t().contiguous()  # [K, N] for x·Wᵀ
                w8 = quant_rows(w16.t().contiguous())[0].t().contiguous()  # [N, K] for g·W
                r = {"M": M, "K": K, "N": N,
                     "fwd_fp16_ms": bench(lambda: x16 @ w16.t()),
                     "fwd_int8_ms": bench(lambda: torch._int_mm(x8, w8t)),
                     "fwd_int8_incl_quant_ms": bench(lambda: torch._int_mm(quant_rows(x16)[0], w8t)),
                     "bwd_fp16_ms": bench(lambda: g16 @ w16),
                     "bwd_int8_incl_quant_ms": bench(lambda: torch._int_mm(quant_rows(g16)[0], w8))}
                r["fwd_speedup_raw"] = round(r["fwd_fp16_ms"] / r["fwd_int8_ms"], 2)
                r["fwd_speedup_incl_quant"] = round(r["fwd_fp16_ms"] / r["fwd_int8_incl_quant_ms"], 2)
                r["bwd_speedup_incl_quant"] = round(r["bwd_fp16_ms"] / r["bwd_int8_incl_quant_ms"], 2)
                info["gemm"][f"{res}px bs{bs} {level} {lname}"] = r
                log(f"GEMM {res}/bs{bs} {lname} M={M}: fwd {r['fwd_speedup_raw']}x raw, {r['fwd_speedup_incl_quant']}x incl. quant; "
                    f"bwd {r['bwd_speedup_incl_quant']}x incl. quant")
                del x16, w16, g16, x8, g8, w8t, w8
except Exception:
    info["gemm"]["error"] = traceback.format_exc()[-1500:]
    log(f"FAILED GEMM benchmark\n{traceback.format_exc()}")
free()
save_info()

# ---- part A: QLoRA ---------------------------------------------------------------------------------
from diffusers import BitsAndBytesConfig  # noqa: E402

comps = tf.load_components(fp32_vae=True)
comps.pop("unet")
free()


def load_unet(base):
    q = None
    if base == "int8":
        q = BitsAndBytesConfig(load_in_8bit=True)
    elif base == "nf4":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    unet = tf.UNet2DConditionModel.from_pretrained(tf.MODEL_ID, subfolder="unet", variant="fp16",
                                                   torch_dtype=torch.float16, quantization_config=q)
    if next(unet.parameters()).device.type != "cuda":
        unet = unet.to("cuda")  # bnb weights quantize on the move to the GPU
    return unet


def fresh(base):
    comps.pop("unet", None)
    free()
    comps["unet"] = load_unet(base)


FAST = dict(cache_latents=True, cache_text=True, optimizer="adamw_fused", grad_ckpt=False)
for base in BASES:
    for shape, (res, bs) in BENCH_SHAPES.items():
        name = f"{base} {shape}"
        trainer = None
        try:
            fresh(base)
            torch.cuda.reset_peak_memory_stats()
            trainer = tf.Trainer(images, INSTANCE_PROMPT, tf.Config(resolution=res, batch_size=bs, **FAST), comps)
            r = tf.benchmark(trainer, steps=10, warmup=3)
            r["ms_per_image"] = r["total"] / bs
            info["qlora_bench"][name] = r
            log(f"{name}: {r['total']:.0f} ms/step ({r['ms_per_image']:.0f} ms/image), peak {r['peak_mem_gb']:.1f} GB")
        except torch.OutOfMemoryError:
            info["qlora_bench"][name] = {"error": "out of memory"}
            log(f"{name}: OUT OF MEMORY")
        except Exception:
            info["qlora_bench"][name] = {"error": traceback.format_exc()[-1500:]}
            log(f"FAILED {name}\n{traceback.format_exc()}")
        trainer = None
        free()
        save_info()

# Full production training per base precision (200 images)
loras = {}
for base in BASES:
    trainer = None
    try:
        fresh(base)
        cfg = tf.Config(**FAST, **PRODUCTION)
        trainer = tf.Trainer(images, INSTANCE_PROMPT, cfg, comps)
        steps = TRAIN_IMAGES // cfg.batch_size
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(steps):
            trainer.step()
        torch.cuda.synchronize()
        train_s = time.perf_counter() - t
        loras[base] = trainer.save("/tmp/run9_loras", weight_name=f"boss_{base}.safetensors")
        info["qlora_train"][base] = {"steps": steps, "train_s": round(train_s, 1)}
        log(f"trained {base}: {steps} steps in {train_s:.0f} s")
    except Exception:
        info["qlora_train"][base] = {"error": traceback.format_exc()[-1500:]}
        log(f"FAILED training {base}\n{traceback.format_exc()}")
    trainer = None
    free()
    save_info()
comps.clear()
free()

# Evaluate every LoRA on the normal fp16 base
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline  # noqa: E402

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(tf.MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16",
                                                 use_safetensors=True).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
rows = []
for base, path in loras.items():
    pipe.unload_lora_weights()
    pipe.load_lora_weights(path, adapter_name="pet")
    pipe.set_adapters(["pet"], adapter_weights=[0.9])
    for i, scene in enumerate(SCENES):
        img = pipe(prompt=f"a sks cat {scene}", negative_prompt="blurry, deformed, low quality", num_inference_steps=25,
                   guidance_scale=6.0, width=768, height=768, generator=torch.Generator("cuda").manual_seed(EVAL_SEED)).images[0]
        p = f"{OUT}/images/{base}_{i}.png"
        img.save(p)
        rows.append({"base": base, "scene": i, "prompt": f"a {PET_CLASS} {scene}", "path": p})
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
if len(df):
    q = df.groupby("base", sort=False)[["dino_crop", "clip_t"]].mean().round(4)
    for b, r in q.iterrows():
        info["qlora_train"].setdefault(b, {}).update({k: float(v) for k, v in r.items()})
    log("quality on the fp16 base (1 seed):\n" + q.to_string())

# ---- plots --------------------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
g = {k: v for k, v in info["gemm"].items() if isinstance(v, dict) and "fwd_speedup_raw" in v}
labels = [k.replace(" blocks", "") for k in g]
x = range(len(g))
w = 0.27
for j, (key, lab) in enumerate([("fwd_speedup_raw", "forward, int8 GEMM only"),
                                 ("fwd_speedup_incl_quant", "forward incl. activation quant"),
                                 ("bwd_speedup_incl_quant", "input grad incl. quant")]):
    axes[0].bar([i + (j - 1) * w for i in x], [v[key] for v in g.values()], width=w, label=lab)
axes[0].axhline(1, color="#999999", lw=1)
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
axes[0].set_ylabel("speedup over fp16 matmul (×)")
axes[0].set_title(f"INT8 GEMM vs fp16 at SDXL linear shapes ({gpu})")
axes[0].legend(fontsize=8)

qb = {k: v for k, v in info["qlora_bench"].items() if "ms_per_image" in v}
axes[1].bar(list(qb), [v["ms_per_image"] for v in qb.values()],
            color=["#1f77b4" if k.startswith("fp16") else "#ff7f0e" if k.startswith("int8") else "#2ca02c" for k in qb])
for i, v in enumerate(qb.values()):
    axes[1].text(i, v["ms_per_image"], f"{v['ms_per_image']:.0f} ms\n{v['peak_mem_gb']:.1f} GB", ha="center", va="bottom", fontsize=7)
axes[1].set_ylabel("training ms per image (no grad checkpointing)")
axes[1].set_title("QLoRA: frozen UNet in fp16 / int8 / nf4")
axes[1].tick_params(axis="x", rotation=45, labelsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/quant_training.png", dpi=120)

if loras:
    fig, axes = plt.subplots(len(loras), len(SCENES), figsize=(3 * len(SCENES), 3 * len(loras)), squeeze=False)
    for r, base in enumerate(loras):
        for c in range(len(SCENES)):
            axes[r, c].imshow(Image.open(f"{OUT}/images/{base}_{c}.png"))
            axes[r, c].axis("off")
        axes[r, 0].set_title(f"LoRA trained on {base} base", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/grid.png", dpi=50)

info["total_min"] = round((time.perf_counter() - t_run) / 60, 1)
save_info()
log(f"done in {info['total_min']} min")
