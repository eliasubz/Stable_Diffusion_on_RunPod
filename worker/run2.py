"""Run 2: LoRA training-step sweep vs. IP-Adapter scale sweep.

One training run saves a LoRA every 50 steps; every checkpoint, three IP-Adapter scales and a
text-only baseline then generate the same scenes and seeds. Identity is scored on held-out photos
(never used for training or IP-Adapter conditioning), both full-image and cropped to the pet.

Runs unattended (Kaggle script kernel or any 16GB+ GPU). Writes everything to OUT as it goes, so a
timeout still leaves partial results.
"""
import base64
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
HOLDOUT = {"Boss4.jpeg", "Boss8.jpeg"}  # scoring only: varied backgrounds, face clearly visible
IP_PHOTO = "Boss2.jpeg"                  # IP-Adapter conditioning photo (a training photo)
TRAIN_CFG = {"steps": 500, "checkpoint_every": 50, "rank": 8, "learning_rate": 1e-4, "resolution": 768}
LORA_STEPS = [50, 100, 150, 200, 300, 400, 500]
LORA_SCALE = 0.9
IP_SCALES = [0.3, 0.45, 0.6]
SCENES = [
    "as a pirate captain on a ship, wearing a tricorn hat, funny, cinematic",
    "as an astronaut on the moon, wearing a space helmet, cinematic",
    "as a chef in a busy kitchen, wearing a chef hat, funny",
    "sitting on a throne like a king, wearing a crown, oil painting",
]
SEEDS = [1, 2, 3]
GEN = {"width": 768, "height": 768, "steps": 25}

OUT = os.environ.get("RUN2_OUT", "/kaggle/working/run2" if os.path.isdir("/kaggle/working") else "run2")
os.makedirs(f"{OUT}/images", exist_ok=True)


def log(msg):
    print(f"[run2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find(name):
    """Locate a file in the attached Kaggle dataset (mount paths vary) or next to this script."""
    for root in ["/kaggle/input", os.path.dirname(os.path.abspath(__file__))]:
        hits = glob.glob(f"{root}/**/{name}", recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


# ---- setup ---------------------------------------------------------------------------------------
t_run = time.perf_counter()
# Copy the worker code somewhere writable: train.py downloads the diffusers script next to itself.
src_dir = os.path.dirname(find("train.py"))
WORK = "/tmp/worker"
shutil.copytree(src_dir, WORK, dirs_exist_ok=True, ignore=shutil.ignore_patterns("test_images", "*.ipynb"))
shutil.copy(find("identity.py"), WORK)
sys.path.insert(0, WORK)
os.chdir(WORK)

log("installing pinned requirements")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

import torch  # noqa: E402

gpu = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
os.environ["LOW_VRAM"] = "1" if vram_gb < 20 else "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["LORA_DIR"] = f"{OUT}/loras"
log(f"GPU {gpu} ({vram_gb:.0f} GB), LOW_VRAM={os.environ['LOW_VRAM']}")

photo_dir = os.path.dirname(find("Boss1.jpeg"))
photos = sorted(glob.glob(f"{photo_dir}/*.jp*g"))
train_photos = [p for p in photos if os.path.basename(p) not in HOLDOUT]
holdout_photos = [p for p in photos if os.path.basename(p) in HOLDOUT]
b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()  # noqa: E731
log(f"{len(train_photos)} training photos, {len(holdout_photos)} held out")

info = {"gpu": gpu, "vram_gb": round(vram_gb, 1), "train_cfg": TRAIN_CFG, "train_photos": [os.path.basename(p) for p in train_photos],
        "holdout_photos": sorted(HOLDOUT), "ip_photo": IP_PHOTO, "gen": GEN, "seeds": SEEDS, "scenes": SCENES}

# ---- 1. train once, keeping a LoRA every 50 steps --------------------------------------------------
import train  # noqa: E402

train_out = train.train({**TRAIN_CFG, "images": [b64(p) for p in train_photos], "class_word": PET_CLASS, "run_id": "boss-run2"})
if "error" in train_out:
    print(train_out.get("log_tail", ""))
    raise SystemExit(f"training failed: {train_out['error']}")
loras = {int(k): v for k, v in train_out.get("checkpoints", {}).items()}
loras.setdefault(TRAIN_CFG["steps"], train_out["lora_path"])
info["train"] = {k: v for k, v in train_out.items() if k != "checkpoints"}
json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)
log(f"training done; LoRAs at steps {sorted(loras)}")

# ---- 2. generate --------------------------------------------------------------------------------
import handler  # noqa: E402  (loads SDXL + IP-Adapter onto the GPU)

subject = train_out["instance_prompt"].replace("a photo of ", "")  # "sks cat"
ip_b64 = b64(f"{photo_dir}/{IP_PHOTO}")
configs = [("text_only", 0, {})]
configs += [("ip_adapter", s, {"pet_image": ip_b64, "ip_adapter_scale": s}) for s in IP_SCALES]
configs += [("lora", st, {"lora": loras[st], "lora_scale": LORA_SCALE}) for st in LORA_STEPS if st in loras]

rows = []
for method, param, extra in configs:
    for i, scene in enumerate(SCENES):
        for seed in SEEDS:
            plain_prompt = f"a {PET_CLASS} {scene}"
            prompt = f"a {subject} {scene}" if method == "lora" else plain_prompt
            path = f"{OUT}/images/{method}_{param}_{i}_{seed}.png"
            try:
                t = time.perf_counter()
                out = handler.handler({"input": {"prompt": prompt, **extra, **GEN, "seed": seed}})
                wall = (time.perf_counter() - t) * 1000
                if "error" in out:
                    raise RuntimeError(out["error"])
                open(path, "wb").write(base64.b64decode(out["image_base64"]))
                rows.append({"method": method, "param": param, "scene": i, "seed": seed, "prompt": plain_prompt,
                             "path": path, "latency_ms": wall, **{f"t_{k}": v for k, v in out["timings_ms"].items()}})
            except Exception:
                log(f"FAILED {method} {param} scene {i} seed {seed}\n{traceback.format_exc()}")
    json.dump(rows, open(f"{OUT}/generations.json", "w"), indent=1)
    log(f"generated {method} {param}")

# ---- 3. score -----------------------------------------------------------------------------------
import gc  # noqa: E402

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

del handler.pipe
gc.collect()
torch.cuda.empty_cache()

from identity import IdentityScorer  # noqa: E402

scorer = IdentityScorer(crop_label=f"a photo of a {PET_CLASS}")
load = lambda p: Image.open(p).convert("RGB")  # noqa: E731
scorer.set_references([load(p) for p in holdout_photos])
for r in rows:
    r.update(scorer.score(load(r["path"]), r["prompt"]))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/results.csv", index=False)

step_times = {int(k): v for k, v in train_out.get("step_times_s", {}).items()}
metrics = ["dino_crop", "clip_i_crop", "dino", "clip_t", "t_inference", "t_lora_load"]
summary = df.groupby(["method", "param"])[metrics].agg(["mean", "std"]).round(4)
summary.columns = [f"{m}_{s}" for m, s in summary.columns]
summary["n"] = df.groupby(["method", "param"]).size()
summary["pet_found_rate"] = df.groupby(["method", "param"])["pet_found"].mean().round(2)
summary["train_min"] = [round(step_times.get(p, float("nan")) / 60, 2) if m == "lora" else 0
                        for m, p in summary.index]
summary.to_csv(f"{OUT}/summary.csv")
log("summary:\n" + summary[["dino_crop_mean", "dino_crop_std", "dino_mean", "clip_t_mean", "clip_t_std", "train_min", "n"]].to_string())

# ---- 4. plots -----------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.tradeoff_plot import plot_tradeoff  # noqa: E402

plot_tradeoff(summary, f"{OUT}/tradeoff.png", f"LoRA training steps vs. IP-Adapter scale ({gpu})")

# Grid: one row per config, one column per scene (seed 1)
cfgs = [(m, p) for m, p, _ in configs]
fig, axes = plt.subplots(len(cfgs), len(SCENES), figsize=(3 * len(SCENES), 3 * len(cfgs)))
for r, (m, p) in enumerate(cfgs):
    for c in range(len(SCENES)):
        f = f"{OUT}/images/{m}_{p}_{c}_{SEEDS[0]}.png"
        if os.path.exists(f):
            axes[r, c].imshow(Image.open(f))
        axes[r, c].axis("off")
    axes[r, 0].set_title(f"{m} {p}" if m != "text_only" else m, loc="left", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/grid.png", dpi=60)

info["total_min"] = round((time.perf_counter() - t_run) / 60, 1)
json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)
log(f"done in {info['total_min']} min")
