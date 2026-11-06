"""Fused multi-LoRA kernels (Triton): y += scale[slot] * (x @ A[slot]^T) @ B[slot]^T, in place.

Punica / S-LoRA style shrink + expand, segmented by image (SGMV: every image is one segment of
1024-4096 tokens sharing one pet's LoRA). Replaces the PyTorch path in multi_lora.py, which launches
2 gathers, 2 bmms, a scale and an add per layer and writes the full-size delta to memory four times
(run 12: 86 ms of a 493 ms UNet step on a T4). Here:

  shrink   h[b, t, :] = scale[s] * x[b, t, :] @ A[s]^T      reads x once, writes a tiny [B, T, r] fp32
  expand   y[b, t, :] += h[b, t, :] @ B[s]^T                 reads and writes y once, in place

with s = slots[b] loaded inside the kernel (no gather kernels, no copied weights), and images on slot 0
("no LoRA") exit immediately. Rank r <= 16 is padded to 16 for tl.dot.

Shrink is split along K so short sequences still fill the GPU (77 text tokens, or 1024 tokens at
BLOCK_T 128, gave 1-8 programs for 40 SMs in run 13's first version): each split writes a partial
h, and expand sums the partials while loading them (no atomics, no zero-init).
"""
import torch
import triton
import triton.language as tl

R_PAD = 16


@triton.jit
def _shrink(x_ptr, a_ptr, scale_ptr, slots_ptr, h_ptr,
            T, K, R, K_PER_SPLIT,
            sxb, sxt, sxk, sas, sar, sak, shs, shb, sht,
            BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr):
    pid_t, b, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(slots_ptr + b)
    if slot == 0:
        return
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_r = tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_T, BLOCK_R), dtype=tl.float32)
    k_start = split * K_PER_SPLIT
    for k0 in range(k_start, tl.minimum(k_start + K_PER_SPLIT, K), BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + b * sxb + offs_t[:, None] * sxt + offs_k[None, :] * sxk,
                    mask=(offs_t[:, None] < T) & (offs_k[None, :] < K), other=0.0)
        a = tl.load(a_ptr + slot * sas + offs_r[None, :] * sar + offs_k[:, None] * sak,
                    mask=(offs_r[None, :] < R) & (offs_k[:, None] < K), other=0.0)  # A[slot]^T tile [K, R]
        acc += tl.dot(x, a)
    acc *= tl.load(scale_ptr + slot).to(tl.float32)
    tl.store(h_ptr + split * shs + b * shb + offs_t[:, None] * sht + offs_r[None, :], acc, mask=offs_t[:, None] < T)


@triton.jit
def _expand(h_ptr, b_ptr, slots_ptr, y_ptr,
            T, N, R, SPLITS,
            shs, shb, sht, sbs, sbn, sbr, syb, syt, syn,
            BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr):
    pid_t, pid_n, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(slots_ptr + b)
    if slot == 0:
        return
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    h = tl.zeros((BLOCK_T, BLOCK_R), dtype=tl.float32)
    for split in range(SPLITS):  # sum the shrink partials
        h += tl.load(h_ptr + split * shs + b * shb + offs_t[:, None] * sht + offs_r[None, :],
                     mask=offs_t[:, None] < T, other=0.0)
    w = tl.load(b_ptr + slot * sbs + offs_n[None, :] * sbn + offs_r[:, None] * sbr,
                mask=(offs_n[None, :] < N) & (offs_r[:, None] < R), other=0.0)  # B[slot]^T tile [R, N]
    delta = tl.dot(h.to(w.dtype), w)
    mask = (offs_t[:, None] < T) & (offs_n[None, :] < N)
    y_ptrs = y_ptr + b * syb + offs_t[:, None] * syt + offs_n[None, :] * syn
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    tl.store(y_ptrs, (y.to(tl.float32) + delta).to(y.dtype), mask=mask)


def fused_lora_(y, x, A, B, scale, slots, block_t=32, block_k=64, block_n=64, block_te=64, min_programs=320):
    """In place: y[b] += scale[slots[b]] * x[b] @ A[slots[b]]^T @ B[slots[b]]^T.

    y: [batch, T, out], x: [batch, T, in], A: [S, r, in], B: [S, out, r], scale: [S], slots: [batch] int.
    block_t/block_k tile the shrink, block_te/block_n the expand; K is split until the shrink has about
    `min_programs` programs. Defaults are run 13's best tiles on an L4. Returns y.
    """
    batch, T, K = x.shape
    N, R = B.shape[1], A.shape[1]
    assert R <= R_PAD, f"rank {R} > {R_PAD}"
    assert slots.numel() == batch, f"{slots.numel()} slots for a batch of {batch}"  # the kernel would read past slots
    tiles_t, k_blocks = triton.cdiv(T, block_t), triton.cdiv(K, block_k)
    splits = max(1, min(k_blocks, triton.cdiv(min_programs, tiles_t * batch)))
    k_per_split = triton.cdiv(k_blocks, splits) * block_k
    splits = triton.cdiv(K, k_per_split)
    h = torch.empty(splits, batch, T, R_PAD, device=x.device, dtype=torch.float32)
    _shrink[(tiles_t, batch, splits)](
        x, A, scale, slots, h, T, K, R, k_per_split,
        x.stride(0), x.stride(1), x.stride(2), A.stride(0), A.stride(1), A.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        BLOCK_T=block_t, BLOCK_K=block_k, BLOCK_R=R_PAD)
    _expand[(triton.cdiv(T, block_te), triton.cdiv(N, block_n), batch)](
        h, B, slots, y, T, N, R, splits,
        h.stride(0), h.stride(1), h.stride(2), B.stride(0), B.stride(1), B.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_T=block_te, BLOCK_N=block_n, BLOCK_R=R_PAD)
    return y


def reference_lora(x, A, B, scale, slots):
    """The PyTorch path from multi_lora.py, as the correctness reference: returns the delta."""
    a, b = A[slots], B[slots]
    delta = torch.bmm(torch.bmm(x, a.transpose(1, 2)), b.transpose(1, 2))
    return delta * scale[slots].view(-1, 1, 1)
