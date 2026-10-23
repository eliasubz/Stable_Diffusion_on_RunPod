"""Run 11: smoke test of the production worker code (train.py + handler.py) before building the RunPod image.

Runs both handlers exactly as RunPod would call them, on Boss's 6 training photos:
  1. train.handler: production training (train_fast, auto precision) + Qwen2-VL description
  2. handler.handler: LCM-8 generation with the new LoRA (twice, to hit the resident slot), plus no LoRA
Records timings, the description and the images. On a T4 the trainer uses fp16 + loss scaling; on an
L4 it picks bf16.
"""
import base64
import gc
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import time

HOLDOUT = {"Boss4.jpeg", "Boss8.jpeg"}
OUT = "/kaggle/working/run11" if os.path.isdir("/kaggle/working") else "run11"
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(f"[run11 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find(name):
    for root in ["/kaggle/input", os.path.dirname(os.path.abspath(__file__))]:
        hits = glob.glob(f"{root}/**/{name}", recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


WORK = "/tmp/worker"
shutil.copytree(os.path.dirname(find("handler.py")), WORK, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("test_images", "*.ipynb"))
sys.path.insert(0, WORK)
os.chdir(WORK)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

import torch  # noqa: E402

info = {"gpu": torch.cuda.get_device_name(0), "bf16_supported": torch.cuda.is_bf16_supported()}
photo_dir = os.path.dirname(find("Boss1.jpeg"))
photos = [p for p in sorted(glob.glob(f"{photo_dir}/*.jp*g")) if os.path.basename(p) not in HOLDOUT]
b64 = [base64.b64encode(open(p, "rb").read()).decode() for p in photos]

# 1. training worker
t = time.perf_counter()
import train  # noqa: E402  (loads SDXL components + Qwen2-VL)

info["train_import_s"] = round(time.perf_counter() - t, 1)
log(f"train.py loaded in {info['train_import_s']} s")
progress = []
# Batch 4 at 512px (production) peaks at ~14-15 GB, past a T4's limit; the L4 has 24 GB. Batch 2 here
# exercises the same code paths.
smoke = {"batch_size": 2} if torch.cuda.get_device_properties(0).total_memory < 20e9 else {}
info["overrides"] = smoke
out = train.train({"images": b64, "class_word": "cat", "run_id": "boss-smoke", **smoke}, progress=progress.append)
assert "error" not in out, out
info["train"] = out
info["train_progress"] = progress[-3:]
log(f"trained: {json.dumps({k: v for k, v in out.items() if k != 'description_raw'}, default=str)}")
lora = out["lora_path"]
shutil.copy(lora, f"{OUT}/boss-smoke.safetensors")

# Free the training worker's models (on RunPod it's a separate endpoint)
train.COMPONENTS.clear()
del train.DESCRIBER
gc.collect()
torch.cuda.empty_cache()

# 2. generation worker
t = time.perf_counter()
import handler  # noqa: E402

info["handler_import_s"] = round(time.perf_counter() - t, 1)
log(f"handler.py loaded in {info['handler_import_s']} s")
subject = f"a sks cat, {out['description']},"
reqs = [("lora_pirate", {"prompt": f"{subject} as a pirate captain on a ship, wearing a tricorn hat, funny, cinematic", "lora": lora, "seed": 1}),
        ("lora_astronaut", {"prompt": f"{subject} as an astronaut on the moon, wearing a space helmet, cinematic", "lora": lora, "seed": 1}),
        ("no_lora_astronaut", {"prompt": "a cat as an astronaut on the moon, wearing a space helmet, cinematic", "seed": 1})]
info["generate"] = {}
for name, inp in reqs:
    res = handler.handler({"input": inp})
    assert "error" not in res, res
    open(f"{OUT}/{name}.png", "wb").write(base64.b64decode(res["image_base64"]))
    info["generate"][name] = res["timings_ms"]
    log(f"{name}: {({k: round(v) for k, v in res['timings_ms'].items()})}")
info["slots"] = len(handler.slots)
json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)
log("done")
