"""Run 8: INT8 quantization and multi-LoRA serving for the production generation setup.

Production setup: SDXL + LCM-LoRA (8 steps, guidance 1) + pet LoRA (scale 0.9) + auto description, 1024px.

Part A, INT8 (torchao) on the UNet, with Boss's LoRA fused in first: fp16 vs. int8 weight-only vs.
W8A8 (int8 dynamic activations + int8 weights; the variant that can use Turing's INT8 tensor cores),
each with and without torch.compile. Reports latency, compile time, memory, identity on held-out
photos, and fidelity to the fp16 image with the same seed.

Part B, multi-LoRA serving: 16 requests from 8 users (Boss + 7 synthetic LoRAs with Boss's shapes
and per-tensor scale), LCM fused into the base weights for everyone:
  swap_fuse     per request: unload, load and fuse the user's LoRA, generate (batch 1)
  unfused_bs1   all LoRAs resident (multi_lora.py), adapter picked per request, batch 1
  multi_bsN     all LoRAs resident, N requests with different LoRAs in one batch
  same_lora_bsN upper bound: one fused LoRA for the whole batch
plus a check that Boss's image from the batched multi-LoRA path matches the fused path.

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
HOLDOUT = {"Boss4.jpeg", "Boss8.jpeg"}
PET_LORA = "boss-step250.safetensors"
PET_SCALE = 0.9
DESCRIPTION = "British Shorthair, gray, short, yellow"
SCENES = [
    "as a pirate captain on a ship, wearing a tricorn hat, funny, cinematic",
    "as an astronaut on the moon, wearing a space helmet, cinematic",
    "as a chef in a busy kitchen, wearing a chef hat, funny",
    "sitting on a throne like a king, wearing a crown, oil painting",
]
SEEDS = [1, 2, 3]
SIZE, STEPS = 1024, 8
QUANT = ["fp16", "fp16+compile", "int8wo", "int8wo+compile", "w8a8", "w8a8+compile"]
N_USERS, N_REQUESTS = 8, 16
BATCHES = [2, 4]

OUT = os.environ.get("RUN8_OUT", "/kaggle/working/run8" if os.path.isdir("/kaggle/working") else "run8")
os.makedirs(f"{OUT}/images", exist_ok=True)


def log(msg):
    print(f"[run8 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
log("installing pinned requirements + torchao")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
tao = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchao"], capture_output=True, text=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import AutoencoderKL, LCMScheduler, StableDiffusionXLPipeline  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402

import multi_lora as ml  # noqa: E402

gpu = torch.cuda.get_device_name(0)
info = {"gpu": gpu, "torch": torch.__version__, "size": SIZE, "steps": STEPS,
        "torchao_install": "ok" if tao.returncode == 0 else tao.stderr[-500:], "quant": {}, "serving": {}}
try:
    import torchao

    info["torchao"] = torchao.__version__
except Exception as e:
    info["torchao"] = f"import failed: {e}"
log(f"GPU {gpu}, torch {torch.__version__}, torchao {info['torchao']}")
photo_dir = os.path.dirname(find("Boss1.jpeg"))
pet_lora = find(PET_LORA)
lcm_lora = hf_hub_download("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors")
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def save_info():
    json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def load_pipe():
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16",
                                                     use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    return pipe


def fuse(pipe, loras):
    """Fuse [(path, adapter_name, weight)] into the base weights and drop the LoRA modules."""
    for path, name, _ in loras:
        pipe.load_lora_weights(path, adapter_name=name)
    pipe.set_adapters([n for _, n, _ in loras], adapter_weights=[w for _, _, w in loras])
    pipe.fuse_lora(adapter_names=[n for _, n, _ in loras])
    pipe.unload_lora_weights()  # keeps the fused weights


def gen(pipe, prompts, seeds):
    gens = [torch.Generator("cuda").manual_seed(s) for s in seeds]
    torch.cuda.synchronize()
    t = time.perf_counter()
    imgs = pipe(prompt=prompts, num_inference_steps=STEPS, guidance_scale=1.0, width=SIZE, height=SIZE,
                generator=gens).images
    torch.cuda.synchronize()
    return imgs, (time.perf_counter() - t) * 1000


PROMPT = lambda scene: f"a sks cat, {DESCRIPTION}, {scene}"  # noqa: E731


# ---- part A: INT8 ---------------------------------------------------------------------------------
def quantize(unet, kind):
    from torchao.quantization import quantize_

    try:  # torchao >= 0.10 config classes
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig

        cfg = Int8WeightOnlyConfig() if kind == "int8wo" else Int8DynamicActivationInt8WeightConfig()
    except ImportError:  # older function-style API
        from torchao.quantization import int8_dynamic_activation_int8_weight, int8_weight_only

        cfg = int8_weight_only() if kind == "int8wo" else int8_dynamic_activation_int8_weight()
    quantize_(unet, cfg)


def unet_mb(unet):
    return sum(p.numel() * p.element_size() for p in unet.parameters()) / 1e6


rows = []
for name in QUANT:
    pipe = None
    try:
        kind, compiled = name.split("+")[0], name.endswith("+compile")
        pipe = load_pipe()
        fuse(pipe, [(lcm_lora, "lcm", 1.0), (pet_lora, "pet", PET_SCALE)])
        mb_before = unet_mb(pipe.unet)
        if kind != "fp16":
            quantize(pipe.unet, kind)
        if compiled:
            pipe.unet = torch.compile(pipe.unet)
        torch.cuda.reset_peak_memory_stats()
        _, first_ms = gen(pipe, [PROMPT(SCENES[0])], [0])  # warm-up; includes compilation
        lat = []
        for i, scene in enumerate(SCENES):
            for seed in SEEDS:
                (img,), ms = gen(pipe, [PROMPT(scene)], [seed])
                lat.append(ms)
                p = f"{OUT}/images/quant_{name}_{i}_{seed}.png"
                img.save(p)
                rows.append({"part": "quant", "config": name, "scene": i, "seed": seed,
                             "prompt": f"a {PET_CLASS} {scene}", "path": p, "latency_ms": ms})
        info["quant"][name] = {"p50_ms": round(float(np.median(lat)), 1), "first_s": round(first_ms / 1000, 1),
                               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                               "unet_param_mb_fp16": round(mb_before, 0)}
        log(f"{name}: p50 {np.median(lat):.0f} ms, first {first_ms / 1000:.1f} s, peak {info['quant'][name]['peak_mem_gb']} GB")
    except Exception:
        info["quant"][name] = {"error": traceback.format_exc()[-1500:]}
        log(f"FAILED {name}\n{traceback.format_exc()}")
    pipe = None
    free()
    torch._dynamo.reset()
    save_info()

# ---- part B: multi-LoRA serving ---------------------------------------------------------------------
from safetensors.torch import load_file, save_file  # noqa: E402

boss_sd = load_file(pet_lora)
user_files = [pet_lora]
for u in range(1, N_USERS):
    f = f"/tmp/user{u}.safetensors"
    save_file(ml.random_like(boss_sd, seed=u), f)
    user_files.append(f)
rng = np.random.default_rng(0)
requests = [{"user": int(u), "scene": int(rng.integers(len(SCENES))), "seed": int(rng.integers(1000))}
            for u in rng.integers(0, N_USERS, N_REQUESTS)]
requests[0] = {"user": 0, "scene": 0, "seed": SEEDS[0]}  # Boss, for the correctness check
info["requests"] = requests


def record(name, total_s, per_request_ms, extra=None):
    info["serving"][name] = {"images_per_s": round(N_REQUESTS / total_s, 3), "total_s": round(total_s, 1),
                             "p50_request_ms": round(float(np.median(per_request_ms)), 1),
                             "p95_request_ms": round(float(np.percentile(per_request_ms, 95)), 1), **(extra or {})}
    log(f"{name}: {N_REQUESTS / total_s:.3f} img/s, p50 request {np.median(per_request_ms):.0f} ms")
    save_info()


# B1. swap + fuse per request, as a single-tenant worker would: load + fuse the user's LoRA,
# generate, then unfuse + unload (LCM stays fused into the base weights)
pipe = None
try:
    pipe = load_pipe()
    fuse(pipe, [(lcm_lora, "lcm", 1.0)])
    gen(pipe, [PROMPT(SCENES[0])], [0])
    lat, swaps = [], []
    t_all = time.perf_counter()
    for r in requests:
        torch.cuda.synchronize()
        t = time.perf_counter()
        pipe.load_lora_weights(user_files[r["user"]], adapter_name="pet")
        pipe.set_adapters(["pet"], adapter_weights=[PET_SCALE])
        pipe.fuse_lora(adapter_names=["pet"])
        torch.cuda.synchronize()
        swap_ms = (time.perf_counter() - t) * 1000
        (img,), ms = gen(pipe, [PROMPT(SCENES[r["scene"]])], [r["seed"]])
        t = time.perf_counter()
        pipe.unfuse_lora()
        pipe.unload_lora_weights()
        torch.cuda.synchronize()
        swap_ms += (time.perf_counter() - t) * 1000
        swaps.append(swap_ms)
        lat.append(swap_ms + ms)
        if r is requests[0]:
            img.save(f"{OUT}/images/serve_swap_fuse_boss.png")
    record("swap_fuse", time.perf_counter() - t_all, lat, {"p50_swap_ms": round(float(np.median(swaps)), 1)})
except Exception:
    info["serving"]["swap_fuse"] = {"error": traceback.format_exc()[-1500:]}
    log(f"FAILED swap_fuse\n{traceback.format_exc()}")
pipe = None
free()

# B2-B4. all LoRAs resident in one UNet
pipe = None
try:
    pipe = load_pipe()
    fuse(pipe, [(lcm_lora, "lcm", 1.0)])
    selector, wrapped = ml.wrap_unet(pipe.unet, rank=8, max_adapters=N_USERS + 1)  # slot 0 = no LoRA
    for u, f in enumerate(user_files):
        n = ml.load_into_slot(wrapped, f, slot=u + 1, scale=PET_SCALE)
    info["multi_lora_layers"] = n
    info["multi_lora_extra_mb"] = round(sum(m.A.numel() + m.B.numel() for m in wrapped.values()) * 2 / 1e6, 1)
    log(f"multi-LoRA: {len(wrapped)} wrapped layers, {n} loaded per user, +{info['multi_lora_extra_mb']} MB for {N_USERS} users")

    def serve(batch, same_lora=False):
        lat, t_all = [], time.perf_counter()
        for k in range(0, N_REQUESTS, batch):
            chunk = requests[k:k + batch]
            users = [0] * len(chunk) if same_lora else [r["user"] for r in chunk]
            selector.idx = torch.tensor([u + 1 for u in users], device="cuda")
            imgs, ms = gen(pipe, [PROMPT(SCENES[r["scene"]]) for r in chunk], [r["seed"] for r in chunk])
            lat += [ms] * len(chunk)  # every request in the batch waits for the whole batch
            if k == 0 and not same_lora:
                imgs[0].save(f"{OUT}/images/serve_multi_bs{batch}_boss.png")
        return time.perf_counter() - t_all, lat

    selector.idx = torch.tensor([1], device="cuda")
    gen(pipe, [PROMPT(SCENES[0])], [0])  # warm-up
    for name, batch, same in [("unfused_bs1", 1, False)] + [(f"multi_bs{b}", b, False) for b in BATCHES] \
            + [(f"same_lora_bs{b}", b, True) for b in BATCHES]:
        try:
            torch.cuda.reset_peak_memory_stats()
            total, lat = serve(batch, same)
            record(name, total, lat, {"batch": batch, "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        except torch.OutOfMemoryError:
            info["serving"][name] = {"error": "out of memory", "batch": batch}
            log(f"{name}: OUT OF MEMORY")
            free()
except Exception:
    info["serving"]["multi_lora"] = {"error": traceback.format_exc()[-1500:]}
    log(f"FAILED multi-LoRA\n{traceback.format_exc()}")
pipe = None
free()
save_info()

# ---- scoring ----------------------------------------------------------------------------------------
import pandas as pd  # noqa: E402

from eval.identity import IdentityScorer  # noqa: E402

scorer = IdentityScorer(crop_label=f"a photo of a {PET_CLASS}")
load = lambda p: Image.open(p).convert("RGB")  # noqa: E731
scorer.set_references([load(f"{photo_dir}/{p}") for p in sorted(HOLDOUT)])
for r in rows:
    r.update(scorer.score(load(r["path"]), r["prompt"]))
    ref = f"{OUT}/images/quant_fp16_{r['scene']}_{r['seed']}.png"
    if os.path.exists(ref):  # fidelity: DINO similarity of the full image to the fp16 image, same seed
        r["fidelity"] = (scorer._dino([load(r["path"])]) @ scorer._dino([load(ref)]).T).item()
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/quant_results.csv", index=False)
if len(df):
    q = df.groupby("config", sort=False)[["dino_crop", "clip_t", "fidelity"]].mean().round(4)
    for c, r in q.iterrows():
        info["quant"].setdefault(c, {}).update({k: float(v) for k, v in r.items()})
    log("quant quality:\n" + q.to_string())

# Boss through the batched multi-LoRA path vs. the fused path (same seed and prompt)
ref = f"{OUT}/images/serve_swap_fuse_boss.png"
if os.path.exists(ref):
    check = {}
    for b in [1] + BATCHES:
        f = f"{OUT}/images/serve_multi_bs{b}_boss.png"
        if os.path.exists(f):
            a, c = np.asarray(load(f), dtype=np.float32), np.asarray(load(ref), dtype=np.float32)
            check[f"bs{b}"] = {"mean_abs_pixel_diff": round(float(np.abs(a - c).mean()), 2),
                               "dino_sim_to_fused": round((scorer._dino([load(f)]) @ scorer._dino([load(ref)]).T).item(), 4)}
    info["multi_lora_vs_fused"] = check
    log(f"multi-LoRA vs fused (Boss): {check}")

# ---- plots ------------------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
qs = {k: v for k, v in info["quant"].items() if "p50_ms" in v}
axes[0].bar(list(qs), [v["p50_ms"] / 1000 for v in qs.values()], color="#1f77b4")
for i, v in enumerate(qs.values()):
    axes[0].text(i, v["p50_ms"] / 1000, f"{v['p50_ms'] / 1000:.2f} s\nfid {v.get('fidelity', float('nan')):.3f}",
                 ha="center", va="bottom", fontsize=8)
axes[0].set_ylabel("p50 seconds per image")
axes[0].set_title(f"INT8 (torchao) on the UNet, LCM-8, {SIZE}px ({gpu})")
axes[0].tick_params(axis="x", rotation=20)
sv = {k: v for k, v in info["serving"].items() if "images_per_s" in v}
axes[1].bar(list(sv), [v["images_per_s"] for v in sv.values()],
            color=["#d62728" if k == "swap_fuse" else "#2ca02c" if k.startswith("multi") else "#888888" for k in sv])
for i, v in enumerate(sv.values()):
    axes[1].text(i, v["images_per_s"], f"{v['images_per_s']:.2f}\np50 {v['p50_request_ms'] / 1000:.1f} s",
                 ha="center", va="bottom", fontsize=8)
axes[1].set_ylabel(f"throughput, images/s ({N_REQUESTS} requests, {N_USERS} users)")
axes[1].set_title("Multi-LoRA serving")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(f"{OUT}/int8_multilora.png", dpi=120)

cfgs = [c for c in QUANT if os.path.exists(f"{OUT}/images/quant_{c}_0_{SEEDS[0]}.png")]
if cfgs:
    fig, axes = plt.subplots(len(cfgs), len(SCENES), figsize=(3 * len(SCENES), 3 * len(cfgs)), squeeze=False)
    for r, c in enumerate(cfgs):
        for s in range(len(SCENES)):
            axes[r, s].imshow(Image.open(f"{OUT}/images/quant_{c}_{s}_{SEEDS[0]}.png"))
            axes[r, s].axis("off")
        axes[r, 0].set_title(c, loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/grid_quant.png", dpi=50)

info["total_min"] = round((time.perf_counter() - t_run) / 60, 1)
save_info()
log(f"done in {info['total_min']} min")
