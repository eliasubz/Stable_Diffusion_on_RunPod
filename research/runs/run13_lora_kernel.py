"""Run 13: fused Triton multi-LoRA kernel (lora_kernels.py) vs. the PyTorch path, on a T4.

A. Correctness per SDXL layer shape (1024px), batch 1 and a mixed batch of 2 (incl. slot 0 = no LoRA):
   max abs / relative error of the fused kernel vs. the PyTorch reference.
B. Tile sweep: shrink (block_t, block_k, split-K target) then expand (block_te, block_n), summed over all shapes.
C. Per-layer microbenchmark: LoRA cost on top of the base linear, PyTorch vs. Triton.
D. UNet step at 1024px: base, the wrapper with no LoRA, PyTorch LoRA, Triton LoRA, at bs1 and bs2 (two pets),
   measured round-robin (one step of each config per round) so T4 clock throttling hits all configs alike;
   plus the UNet-output difference between the paths.

Random rank-8 LoRAs (timing doesn't depend on values). Runs unattended on Kaggle, writes results.json + plot.
"""
import itertools
import json
import os
import sys

import torch
import triton
from diffusers import UNet2DConditionModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "worker"))
import lora_kernels as lk
import multi_lora as ml

RANK, SLOTS, ITERS, WARMUP = 8, 33, 20, 5
out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__}
dev, dt = "cuda", torch.float16
# (tokens, in, out) of the LoRA'd linears at 1024px
SHAPES = {"self_640": (4096, 640, 640), "self_1280": (1024, 1280, 1280),
          "cross_kv_640": (77, 2048, 640), "cross_kv_1280": (77, 2048, 1280)}


def save():
    json.dump(out, open("results.json", "w"), indent=1)


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms, includes launch overhead when CPU-bound


def layer(T, K, N, batch, slots):
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(batch, T, K, device=dev, dtype=dt, generator=g)
    A = (torch.randn(SLOTS, RANK, K, device=dev, dtype=dt, generator=g) * 0.05)
    B = (torch.randn(SLOTS, N, RANK, device=dev, dtype=dt, generator=g) * 0.05)
    A[0], B[0] = 0, 0
    scale = torch.rand(SLOTS, device=dev, dtype=dt) + 0.5
    W = torch.randn(N, K, device=dev, dtype=dt, generator=g) / K ** 0.5
    return x, A, B, scale, W, torch.tensor(slots, device=dev, dtype=torch.int32)


# A. correctness
out["correctness"] = {}
for (name, (T, K, N)), slots in itertools.product(SHAPES.items(), ([5], [7, 0], [3, 12])):
    x, A, B, scale, W, s = layer(T, K, N, len(slots), slots)
    y = x @ W.T
    ref = (y.float() + lk.reference_lora(x.float(), A.float(), B.float(), scale.float(), s.long()))
    got = lk.fused_lora_(y.clone(), x, A, B, scale, s).float()
    torch_fp16 = y + lk.reference_lora(x, A, B, scale, s.long())  # the current production path, for scale
    out["correctness"][f"{name}_{slots}"] = {
        "triton_max_abs": (got - ref).abs().max().item(),
        "pytorch_fp16_max_abs": (torch_fp16.float() - ref).abs().max().item(),
        "delta_rms": (ref - y.float()).pow(2).mean().sqrt().item(),
    }
print(json.dumps(out["correctness"], indent=1), flush=True)
worst = max(v["triton_max_abs"] - v["pytorch_fp16_max_abs"] for v in out["correctness"].values())
out["correctness_ok"] = worst < 1e-2
save()

# B. tiles, batch 1, summed over the four shapes; shrink first, then expand with the best shrink
def sweep_total(cfg):
    total = 0.0
    for T, K, N in SHAPES.values():
        x, A, B, scale, W, s = layer(T, K, N, 1, [5])
        y = x @ W.T
        total += timed(lambda: lk.fused_lora_(y, x, A, B, scale, s, **cfg), iters=30, warmup=3)
    return total


def sweep(grid, fixed):
    res = {}
    for vals in itertools.product(*grid.values()):
        cfg = {**fixed, **dict(zip(grid, vals))}
        try:
            res[json.dumps(cfg)] = sweep_total(cfg)
        except Exception:  # e.g. out of shared memory on the T4
            res[json.dumps(cfg)] = float("inf")
    return res, json.loads(min(res, key=res.get))


shrink_res, CFG = sweep({"block_t": (16, 32, 64), "block_k": (32, 64, 128), "min_programs": (80, 160, 320)},
                        {"block_te": 64, "block_n": 64})
expand_res, CFG = sweep({"block_te": (32, 64, 128), "block_n": (32, 64, 128)},
                        {k: CFG[k] for k in ("block_t", "block_k", "min_programs")})
out["tile_sweep_ms"], out["best_cfg"] = {**shrink_res, **expand_res}, CFG
print("best", CFG, min(expand_res.values()), flush=True)
save()

# C. per-layer LoRA cost on top of the base linear
out["per_layer"] = {}
for (name, (T, K, N)), slots in itertools.product(SHAPES.items(), ([5], [3, 12])):
    x, A, B, scale, W, s = layer(T, K, N, len(slots), slots)
    sl = s.long()
    base = timed(lambda: x @ W.T)
    torch_ms = timed(lambda: (x @ W.T) + lk.reference_lora(x, A, B, scale, sl))
    triton_ms = timed(lambda: lk.fused_lora_(x @ W.T, x, A, B, scale, s, **CFG))
    out["per_layer"][f"{name}_bs{len(slots)}"] = {"base_ms": base, "pytorch_lora_extra_ms": torch_ms - base,
                                                    "triton_lora_extra_ms": triton_ms - base}
print(json.dumps(out["per_layer"], indent=1), flush=True)
save()

# D. UNet step
import subprocess

ml.fused_lora_ = lambda *a: lk.fused_lora_(*a, **CFG)
unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=dt, variant="fp16"
).to(dev).eval()


def inputs(bs):
    g = torch.Generator(device=dev).manual_seed(1)
    return dict(
        sample=torch.randn(bs, 4, 128, 128, device=dev, dtype=dt, generator=g),
        timestep=torch.tensor(500, device=dev),
        encoder_hidden_states=torch.randn(bs, 77, 2048, device=dev, dtype=dt, generator=g),
        added_cond_kwargs=dict(text_embeds=torch.randn(bs, 1280, device=dev, dtype=dt, generator=g),
                               time_ids=torch.tensor([[1024, 1024, 0, 0, 1024, 1024]] * bs, device=dev, dtype=dt)),
    )


selector, wrapped = ml.wrap_unet(unet, rank=RANK, max_adapters=3)
for m in wrapped.values():
    m.A.data[1:].normal_(0, 0.01)
    m.B.data[1:].normal_(0, 0.01)
    m.scale[1:] = 0.9
parents = [(mod, name, child) for mod in unet.modules() for name, child in mod.named_children()
           if isinstance(child, ml.MultiLoRALinear)]


def set_mode(mode, idx):
    for mod, name, child in parents:  # "base" puts the plain nn.Linear back
        setattr(mod, name, child.base if mode == "base" else child)
    ml.USE_TRITON = mode == "triton"
    selector.idx = torch.tensor(idx, device=dev) if mode in ("pytorch", "triton") else None


def clocks():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except Exception as e:
        return repr(e)


MODES = ("base", "wrapped_no_lora", "pytorch", "triton")
steps, out["clocks"] = {}, {"start": clocks()}
with torch.no_grad():
    for bs, idx in ((1, [1]), (2, [1, 2]), (4, [1, 2, 1, 2])):  # bs4: fits on 24 GB (L4), not on a T4
        if bs == 4 and torch.cuda.get_device_properties(0).total_memory < 20e9:
            continue
        kw = inputs(bs)
        for mode in MODES:  # warm up and compile everything first
            set_mode(mode, idx)
            for _ in range(WARMUP):
                unet(**kw)
        times = {m: [] for m in MODES}
        for _ in range(ITERS):
            for mode in MODES:
                set_mode(mode, idx)
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                unet(**kw)
                e.record()
                torch.cuda.synchronize()
                times[mode].append(s.elapsed_time(e))
        for mode in MODES:
            steps[f"{mode}_bs{bs}"] = sorted(times[mode])[ITERS // 2]
        out["clocks"][f"after_bs{bs}"] = clocks()
        outs = {}
        for mode in ("pytorch", "triton"):
            set_mode(mode, idx)
            outs[mode] = unet(**kw).sample.float()
        a, b = outs["pytorch"], outs["triton"]
        out.setdefault("unet_output_pytorch_vs_triton", {})[f"bs{bs}"] = {
            "max_abs": (a - b).abs().max().item(),
            "cosine": torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()}
        out["unet_step_ms"] = steps
        print(json.dumps(steps, indent=1), flush=True)
        save()

base, pt, tr = steps["base_bs1"], steps["pytorch_bs1"], steps["triton_bs1"]
out["summary"] = {
    "wrapper_overhead_ms": steps["wrapped_no_lora_bs1"] - base,
    "lora_overhead_ms_pytorch": pt - base, "lora_overhead_ms_triton": tr - base,
    "overhead_removed_pct": 100 * (pt - tr) / (pt - base),
    "step_speedup": pt / tr, "s_saved_per_image_8_steps": 8 * (pt - tr) / 1000,
    "bs2_step_speedup": steps["pytorch_bs2"] / steps["triton_bs2"],
    "bs4_step_speedup": steps["pytorch_bs4"] / steps["triton_bs4"] if "triton_bs4" in steps else None,
}
print(json.dumps(out, indent=1), flush=True)
save()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["base\n(fused LoRA)", "PyTorch\nmulti-LoRA", "Triton\nmulti-LoRA"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(labels, [base, pt, tr], color=["#999", "#d62728", "#1f77b4"])
ax[0].set_ylabel("ms per UNet step (1024px, bs1)")
ax[0].set_title("SDXL UNet step on a T4")
names = list(out["per_layer"])
w = 0.4
ax[1].bar([i - w / 2 for i in range(len(names))], [out["per_layer"][n]["pytorch_lora_extra_ms"] for n in names], w, label="PyTorch", color="#d62728")
ax[1].bar([i + w / 2 for i in range(len(names))], [out["per_layer"][n]["triton_lora_extra_ms"] for n in names], w, label="Triton", color="#1f77b4")
ax[1].set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
ax[1].set_ylabel("LoRA ms on top of the base linear")
ax[1].set_title("Per-layer LoRA cost")
ax[1].legend()
fig.tight_layout()
fig.savefig("lora_kernel.png", dpi=150)
