"""Run 10: is INT8 GEMM actually faster than fp16 on a T4, with the right kernel and layout?

Run 9 found torch._int_mm 1.5–2.8× *slower* than fp16 at SDXL shapes, with B passed row-major.
On paper Turing does 130 INT8 TOPS vs 65 fp16 TFLOPS. This microbenchmark checks every path:
  - fp16 cuBLAS (A @ B and A @ Wᵀ)
  - torch._int_mm with B row-major [K, N] (run 9) and column-major / "TN" (Wᵀ view of [N, K])
  - a Triton GEMM (int8 → int32 tl.dot, and fp16 → fp32 with the same kernel), autotuned
on a large square problem (each path's ceiling) and on SDXL's linear shapes. Also records which CUDA
kernels cuBLAS launches (IMMA tensor-core kernels or not) and checks results against fp32.
"""
import json
import os
import time
import traceback

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

OUT = os.environ.get("RUN10_OUT", "/kaggle/working/run10" if os.path.isdir("/kaggle/working") else "run10")
os.makedirs(OUT, exist_ok=True)
PEAK = {"fp16": 65.0, "int8": 130.0}  # T4 tensor-core peaks (TFLOPS / TOPS)

SHAPES = {
    "square 4096": (4096, 4096, 4096),
    "512/bs4 ff-in 640": (4096, 640, 5120),
    "512/bs4 ff-out 640": (4096, 2560, 640),
    "512/bs4 attn 640": (4096, 640, 640),
    "512/bs4 ff-in 1280": (1024, 1280, 10240),
    "768/bs1 ff-in 640": (2304, 640, 5120),
    "768/bs1 attn 640": (2304, 640, 640),
}


def log(msg):
    print(f"[run10 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def kernels(fn):
    """Names of the CUDA kernels one call launches."""
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.key for e in prof.key_averages() if e.device_type.name == "CUDA"})


# ---- Triton GEMM (same kernel for int8 and fp16) -------------------------------------------------------
CONFIGS = [triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s)
           for bm, bn, bk, w, s in [(128, 128, 64, 8, 2), (128, 128, 32, 4, 2), (128, 64, 64, 4, 2),
                                    (64, 128, 64, 4, 2), (64, 64, 64, 4, 2), (128, 256, 64, 8, 2)]]


@triton.autotune(configs=CONFIGS, key=["M", "N", "K", "INT8"])
@triton.jit
def gemm_kernel(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
                INT8: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BN)
    pm, pn = pid // grid_n, pid % grid_n
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr = a + rm[:, None] * sam + rk[None, :] * sak
    b_ptr = b + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.int32 if INT8 else tl.float32)
    for k in range(0, K, BK):
        x = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0)
        y = tl.load(b_ptr, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0)
        acc += tl.dot(x, y, out_dtype=tl.int32 if INT8 else tl.float32)
        a_ptr += BK * sak
        b_ptr += BK * sbk
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_mm(a, b, int8):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=torch.int32 if int8 else torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)  # noqa: E731
    gemm_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), INT8=int8)
    return c


# ---- run ------------------------------------------------------------------------------------------------
gpu = torch.cuda.get_device_name(0)
info = {"gpu": gpu, "torch": torch.__version__, "triton": triton.__version__, "capability": torch.cuda.get_device_capability(),
        "peak": PEAK, "results": {}, "kernels": {}}
log(f"GPU {gpu} sm{info['capability']}, torch {torch.__version__}, triton {triton.__version__}")

for sname, (M, K, N) in SHAPES.items():
    a16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w16 = torch.randn(N, K, device="cuda", dtype=torch.float16)  # nn.Linear layout [out, in]
    b16 = w16.t().contiguous()                                  # row-major [K, N]
    a8 = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    w8 = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
    b8_row = w8.t().contiguous()  # row-major [K, N]  (run 9)
    b8_col = w8.t()               # column-major view [K, N] ("TN")
    ref = a8.float() @ w8.float().t()

    variants = {
        "fp16 A@B (row-major B)": ("fp16", lambda: a16 @ b16),
        "fp16 A@Wt (TN)": ("fp16", lambda: a16 @ w16.t()),
        "int8 _int_mm row-major B": ("int8", lambda: torch._int_mm(a8, b8_row)),
        "int8 _int_mm TN (col-major B)": ("int8", lambda: torch._int_mm(a8, b8_col)),
        "triton fp16": ("fp16", lambda: triton_mm(a16, b16, int8=False)),
        "triton int8": ("int8", lambda: triton_mm(a8, b8_row, int8=True)),
    }
    flops = 2 * M * N * K
    res = {"M": M, "K": K, "N": N}
    for vname, (kind, fn) in variants.items():
        try:
            out = fn()
            if kind == "int8":
                # fp32 reference isn't exact past 2^24, so compare with a relative tolerance
                res[f"{vname} matches"] = bool(torch.allclose(out.float(), ref, rtol=1e-5, atol=64))
            ms = timeit(fn)
            tput = flops / ms / 1e9
            res[vname] = {"ms": round(ms, 4), "tera_ops": round(tput, 1), "pct_of_peak": round(100 * tput / PEAK[kind], 1)}
            if sname == "square 4096" or sname.startswith("512/bs4 ff-in 640"):
                info["kernels"][f"{sname} | {vname}"] = kernels(fn)
        except Exception as e:
            res[vname] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            log(f"{sname} {vname}: FAILED {type(e).__name__}: {str(e)[:200]}")
    base = res.get("fp16 A@Wt (TN)", {}).get("ms")
    for v in variants:
        if isinstance(res.get(v), dict) and "ms" in res[v] and base:
            res[v]["speedup_vs_fp16"] = round(base / res[v]["ms"], 2)
    info["results"][sname] = res
    log(f"{sname} (M={M} K={K} N={N}): " + ", ".join(
        f"{v}: {res[v]['tera_ops']} T ({res[v].get('speedup_vs_fp16', '-')}x)" for v in variants if "ms" in res.get(v, {})))
    json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)
    del a16, w16, b16, a8, w8, b8_row, b8_col, ref
    torch.cuda.empty_cache()

for k, v in info["kernels"].items():
    log(f"kernels {k}: {v}")

# ---- plot -----------------------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

vnames = ["fp16 A@Wt (TN)", "triton fp16", "int8 _int_mm row-major B", "int8 _int_mm TN (col-major B)", "triton int8"]
colors = ["#1f77b4", "#aec7e8", "#d62728", "#ff7f0e", "#2ca02c"]
fig, ax = plt.subplots(figsize=(14, 5))
names = list(info["results"])
w = 0.16
for j, (v, col) in enumerate(zip(vnames, colors)):
    vals = [info["results"][s].get(v, {}).get("tera_ops", 0) or 0 for s in names]
    ax.bar([i + (j - 2) * w for i in range(len(names))], vals, width=w, label=v, color=col)
ax.axhline(PEAK["fp16"], color="#1f77b4", ls=":", lw=1, label="fp16 peak 65 TFLOPS")
ax.axhline(PEAK["int8"], color="#2ca02c", ls=":", lw=1, label="int8 peak 130 TOPS")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylabel("achieved TFLOPS / TOPS")
ax.set_title(f"INT8 vs fp16 GEMM paths on {gpu}")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"{OUT}/gemm_paths.png", dpi=120)
log("done")
