"""Run 5: is it Boss, or just a grey British Shorthair? (LoRA vs. description ablation, LCM-8)

Run 4 showed an auto pet description removes identity failures. But DINO similarity to Boss may
just reward the breed the description names. This run separates the two:
  - configs: pet LoRA with/without description (v1 = run 4's text, v2 = attribute-bound text that
    shouldn't leak colors into the scene), and the description alone without the LoRA.
  - metric: identity *margin* = similarity to Boss's held-out photos minus similarity to real British
    Shorthairs from Oxford-IIIT Pet. Only the margin shows "this cat" rather than "this breed".
  - the metric is validated on real photos first: Boss's own photos should score a high margin,
    other British Shorthairs about zero.

Production config only (LCM-8, fused LoRAs, 1024px). Runs unattended on Kaggle and writes as it goes.
"""
import gc
import glob
import io
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
FAIL_BELOW = 0.4

DESC_V1 = "British Shorthair, gray, short, yellow"  # run 4's VLM output, verbatim
VLM_ID = "Qwen/Qwen2-VL-2B-Instruct"
# v2: every attribute names its body part, so colors stay on the cat. The example is deliberately a
# different-looking cat so the model can't copy Boss's attributes from the prompt.
VLM_PROMPT = (
    "These photos all show the same cat. Describe only the cat's appearance as a comma-separated list of "
    "short attribute phrases, each naming the body part it describes, starting with the likely breed. "
    "Example format: 'Siamese, short cream fur, dark brown face and ears, blue eyes, slim build'. "
    "Ignore the background. At most 15 words. Output only the list."
)

# Only the production sampler from run 4
METHODS = {"lcm8": ("lcm", "lcm", 8, 1.0, None)}
SPEED_LORAS = {"lcm": ("latent-consistency/lcm-lora-sdxl", ["pytorch_lora_weights.safetensors"])}
# name: (use pet LoRA, description key or None)
CONFIGS = {
    "lora": (True, None),
    "lora+desc_v1": (True, "v1"),
    "lora+desc_v2": (True, "v2"),
    "nolora+desc_v1": (False, "v1"),
    "nolora+desc_v2": (False, "v2"),
}

# Real British Shorthairs (other individuals) for the breed baseline
BREED_DATASET = ("timm/oxford-iiit-pet", "data/test-00000-of-00001.parquet")
BREED_LABEL = 9  # british_shorthair
N_BREED_REFS = 20     # breed reference set
N_BREED_PROBES = 10   # other BSH photos used only to validate the metric

OUT = os.environ.get("RUN5_OUT", "/kaggle/working/run5" if os.path.isdir("/kaggle/working") else "run5")
os.makedirs(f"{OUT}/images", exist_ok=True)


def log(msg):
    print(f"[run5 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
        "pet_lora": PET_LORA, "pet_lora_scale": PET_LORA_SCALE, "configs": CONFIGS, "fail_below": FAIL_BELOW}


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


try:
    desc_v2, raw, load_s, gen_s = describe_pet(train_photos[:4])
    info["description_v2"] = {"text": desc_v2, "raw": raw, "source": VLM_ID, "load_s": load_s, "generate_s": gen_s}
except Exception:
    log(f"VLM failed, falling back to a manual v2 description\n{traceback.format_exc()}")
    desc_v2 = "British Shorthair, short dense gray fur, round face, yellow eyes, stocky build"
    info["description_v2"] = {"text": desc_v2, "source": "manual fallback"}
DESCRIPTIONS = {"v1": DESC_V1, "v2": desc_v2}
info["description_v1"] = DESC_V1
log(f"descriptions: {DESCRIPTIONS}")
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


def set_loras(speed, with_pet=True):
    pipe.unfuse_lora()
    pipe.unload_lora_weights()
    names, weights = [], []
    if with_pet:
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
speed, sched, steps, guidance, timesteps = METHODS["lcm8"]
pipe.scheduler = SCHEDULERS[sched]()
rows, cfg_info = [], {}
for name, (with_pet, desc_key) in CONFIGS.items():
    try:
        set_loras(speed, with_pet=with_pet)
        pipe.scheduler = SCHEDULERS[sched]()
        token = SUBJECT if with_pet else PET_CLASS
        subject = f"{token}, {DESCRIPTIONS[desc_key]}," if desc_key else token
        torch.cuda.reset_peak_memory_stats()
        generate(f"a {subject} {SCENES[0]}", steps, guidance, timesteps, seed=0)  # warm-up
        for i, scene in enumerate(SCENES):
            for seed in SEEDS:
                img, ms = generate(f"a {subject} {scene}", steps, guidance, timesteps, seed)
                path = f"{OUT}/images/{name}_{i}_{seed}.png"
                img.save(path)
                rows.append({"config": name, "lora": with_pet, "desc": desc_key or "none", "scene": i, "seed": seed,
                             "prompt": f"a {PET_CLASS} {scene}", "full_prompt": f"a {subject} {scene}",
                             "path": path, "latency_ms": ms})
        cfg_info[name] = {"peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        log(f"{name}: p50 {np.median([r['latency_ms'] for r in rows if r['config'] == name]):.0f} ms")
    except Exception:
        log(f"FAILED {name}\n{traceback.format_exc()}")
    json.dump(rows, open(f"{OUT}/generations.json", "w"), indent=1)
    info["per_config"] = cfg_info
    save_info()

# ---- 4. score: similarity to Boss vs. similarity to the breed --------------------------------------
import pandas as pd  # noqa: E402

del pipe
gc.collect()
torch.cuda.empty_cache()

from eval.identity import IdentityScorer  # noqa: E402

scorer = IdentityScorer(crop_label=f"a photo of a {PET_CLASS}")
load = lambda p: Image.open(p).convert("RGB")  # noqa: E731
scorer.set_references([load(f"{photo_dir}/{p}") for p in sorted(HOLDOUT)])

breed_df = pd.read_parquet(hf_hub_download(BREED_DATASET[0], BREED_DATASET[1], repo_type="dataset"))
breed_df = breed_df[breed_df.label == BREED_LABEL].reset_index(drop=True)
breed_imgs = [Image.open(io.BytesIO(b["bytes"])).convert("RGB") for b in breed_df.image[: N_BREED_REFS + N_BREED_PROBES]]
info["breed"] = {"dataset": BREED_DATASET[0], "available": len(breed_df), "refs": N_BREED_REFS, "probes": N_BREED_PROBES}
log(f"{len(breed_df)} British Shorthair photos available")


def crop_dino(images):
    return scorer._dino([scorer.crop(im)[0] for im in images])


breed_ref_emb = crop_dino(breed_imgs[:N_BREED_REFS])


def breed_sim(image):
    return (crop_dino([image]) @ breed_ref_emb.T).mean().item()


# Validate the metric on real photos: Boss's training photos vs. other British Shorthairs
validation = []
for p in train_photos:
    im = load(p)
    s = scorer.score(im, "a photo of a cat")
    validation.append({"set": "boss_train_photos", "name": os.path.basename(p),
                       "boss": s["dino_crop"], "breed": breed_sim(im)})
for k, im in enumerate(breed_imgs[N_BREED_REFS:]):
    s = scorer.score(im, "a photo of a cat")
    validation.append({"set": "other_bsh_photos", "name": f"oxford_bsh_{k}", "boss": s["dino_crop"], "breed": breed_sim(im)})
val = pd.DataFrame(validation)
val["margin"] = val.boss - val.breed
val.to_csv(f"{OUT}/metric_validation.csv", index=False)
val_summary = val.groupby("set")[["boss", "breed", "margin"]].agg(["mean", "std"]).round(3)
info["metric_validation"] = json.loads(val_summary.to_json())
log("metric validation (real photos):\n" + val_summary.to_string())

for r in rows:
    im = load(r["path"])
    r.update(scorer.score(im, r["prompt"]))
    r["breed_sim"] = breed_sim(im)
    r["margin"] = r["dino_crop"] - r["breed_sim"]
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/results.csv", index=False)

g = df.groupby("config", sort=False)
summary = pd.DataFrame({
    "lora": g.lora.first(), "desc": g.desc.first(),
    "p50_ms": g.latency_ms.median(),
    "boss_sim_mean": g.dino_crop.mean(), "boss_sim_std": g.dino_crop.std(), "boss_sim_min": g.dino_crop.min(),
    "breed_sim_mean": g.breed_sim.mean(),
    "margin_mean": g.margin.mean(), "margin_std": g.margin.std(),
    "fail_rate": g.dino_crop.apply(lambda s: (s < FAIL_BELOW).mean()),
    "clip_t_mean": g.clip_t.mean(), "clip_t_std": g.clip_t.std(),
    "n": g.size(),
}).round(4)
summary.to_csv(f"{OUT}/summary.csv")
log("summary:\n" + summary.to_string())

# ---- 5. plots -------------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
colors = {"boss_train_photos": "#1f77b4", "other_bsh_photos": "#d62728"}
for name, grp in val.groupby("set"):
    ax.scatter(grp.breed, grp.boss, label=f"real: {name.replace('_', ' ')}", color=colors[name], marker="x")
for c, r in summary.iterrows():
    ax.scatter(r.breed_sim_mean, r.boss_sim_mean, color="#2ca02c" if r.lora else "#ff7f0e", s=60)
    ax.annotate(c, (r.breed_sim_mean, r.boss_sim_mean), textcoords="offset points", xytext=(5, 4), fontsize=8)
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, "--", color="#999999", lw=1, label="margin = 0 (breed only)")
ax.set_xlabel("similarity to other British Shorthairs (DINO crop)")
ax.set_ylabel("similarity to Boss, held-out (DINO crop)")
ax.set_title("above the diagonal = Boss, not just the breed")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1]
x = range(len(summary))
ax.bar(x, summary.margin_mean, yerr=summary.margin_std / summary.n ** 0.5, capsize=3,
       color=["#2ca02c" if lo else "#ff7f0e" for lo in summary.lora])
for setname, grp in val.groupby("set"):
    ax.axhline(grp.margin.mean(), ls="--", lw=1, color=colors[setname], label=f"real {setname.replace('_', ' ')}")
ax.set_xticks(list(x))
ax.set_xticklabels(summary.index, rotation=20, ha="right")
ax.set_ylabel("identity margin (Boss − breed)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")
fig.suptitle(f"LoRA vs. description: individual identity, LCM-8 ({gpu}, {SIZE}px)")
plt.tight_layout()
plt.savefig(f"{OUT}/margin.png", dpi=120)

names = list(summary.index)
fig, axes = plt.subplots(len(names), len(SCENES), figsize=(3 * len(SCENES), 3 * len(names)))
for r, c in enumerate(names):
    for s in range(len(SCENES)):
        f = f"{OUT}/images/{c}_{s}_{SEEDS[1]}.png"
        if os.path.exists(f):
            axes[r, s].imshow(Image.open(f))
        axes[r, s].axis("off")
    axes[r, 0].set_title(f"{c}  (margin {summary.loc[c, 'margin_mean']:+.3f})", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/grid.png", dpi=50)

info["total_min"] = round((time.perf_counter() - t_run) / 60, 1)
save_info()
log(f"done in {info['total_min']} min")
