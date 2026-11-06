"""Run 12: is the multi-LoRA path a bottleneck worth a kernel?

Times one SDXL UNet step at 1024px (the production unit of work, x8 per image) on the T4:
  base          plain UNet (= LoRA fused into the weights, same cost)
  wrapped_off   multi_lora.py wrapping, no adapter selected
  lora_bs1      one resident rank-8 LoRA, batch 1 (production)
  base_bs2 / lora_bs2   two different LoRAs in one batch
then profiles lora_bs1 and attributes CUDA time to the LoRA ops (gather, bmm, scale, add) vs. the rest,
plus the ideal per-layer cost if the LoRA were fused (read x once, read+write y once).

Random LoRA weights: timing doesn't depend on values. Runs unattended on Kaggle, writes results.json.
"""
import json
import time

import torch
from diffusers import UNet2DConditionModel

import multi_lora as ml

RANK, ITERS, WARMUP = 8, 20, 5
out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}

unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16, variant="fp16"
).to("cuda").eval()


def inputs(bs):
    return dict(
        sample=torch.randn(bs, 4, 128, 128, device="cuda", dtype=torch.float16),
        timestep=torch.tensor(500, device="cuda"),
        encoder_hidden_states=torch.randn(bs, 77, 2048, device="cuda", dtype=torch.float16),
        added_cond_kwargs=dict(
            text_embeds=torch.randn(bs, 1280, device="cuda", dtype=torch.float16),
            time_ids=torch.tensor([[1024, 1024, 0, 0, 1024, 1024]] * bs, device="cuda", dtype=torch.float16),
        ),
    )


@torch.no_grad()
def bench(name, bs):
    kw = inputs(bs)
    for _ in range(WARMUP):
        unet(**kw)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        unet(**kw)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    out[name] = {"bs": bs, "p50_ms": times[len(times) // 2], "min_ms": times[0], "ms_per_image": times[len(times) // 2] / bs}
    print(name, out[name], flush=True)


bench("base", 1)
bench("base_bs2", 2)

selector, wrapped = ml.wrap_unet(unet, rank=RANK, max_adapters=3)
for m in wrapped.values():  # random adapters in slots 1 and 2
    m.A.data[1:].normal_(0, 0.01)
    m.B.data[1:].normal_(0, 0.01)
out["lora_layers"] = len(wrapped)
out["layer_shapes"] = sorted({(m.base.in_features, m.base.out_features) for m in wrapped.values()})

selector.idx = None
bench("wrapped_off", 1)
selector.idx = torch.tensor([1], device="cuda")
bench("lora_bs1", 1)
selector.idx = torch.tensor([1, 2], device="cuda")
bench("lora_bs2", 2)

# Attribute lora_bs1 CUDA time per op. Ops only the LoRA path launches: index (gather), bmm, mul;
# add is shared with the base UNet (residuals), so it's split by comparing against a base profile.
from torch.profiler import ProfilerActivity, profile


@torch.no_grad()
def op_times(idx):
    selector.idx = idx
    kw = inputs(1)
    unet(**kw)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            unet(**kw)
        torch.cuda.synchronize()
    return {e.key: e.device_time_total / 3 / 1000 for e in p.key_averages()}  # ms per step


on, off = op_times(torch.tensor([1], device="cuda")), op_times(None)
diff = {k: on.get(k, 0) - off.get(k, 0) for k in set(on) | set(off)}
out["lora_bs1_extra_ms_by_op"] = dict(sorted(((k, round(v, 3)) for k, v in diff.items() if abs(v) > 0.05), key=lambda kv: -kv[1]))
out["lora_bs1_total_cuda_ms"] = sum(on.values())
out["wrapped_off_total_cuda_ms"] = sum(off.values())

# Ideal fused cost: per layer, read x once + read/write y once (tiny A/B and rank-r intermediate ignored)
tokens = {320: 0, 640: 64 * 64, 1280: 32 * 32}  # SDXL: 640-ch blocks at 64x64, 1280-ch at 32x32 (1024px)
bytes_ideal = 0
for m in wrapped.values():
    if m.base.in_features == 2048:  # cross-attn k/v run on the 77 text tokens: negligible
        continue
    n = tokens.get(m.base.out_features, 0)
    bytes_ideal += 2 * n * (m.base.in_features + 2 * m.base.out_features)
out["ideal_fused_GB_per_step"] = bytes_ideal / 1e9

base, lora = out["base"]["p50_ms"], out["lora_bs1"]["p50_ms"]
out["verdict"] = {
    "lora_overhead_ms_per_step": lora - base,
    "lora_overhead_pct": 100 * (lora - base) / lora,
    "lora_overhead_s_per_image_8_steps": 8 * (lora - base) / 1000,
    "bs2_overhead_pct": 100 * (out["lora_bs2"]["p50_ms"] - out["base_bs2"]["p50_ms"]) / out["lora_bs2"]["p50_ms"],
}
print(json.dumps(out, indent=1))
json.dump(out, open("results.json", "w"), indent=1)
