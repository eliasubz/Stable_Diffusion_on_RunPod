"""Batched multi-LoRA for the SDXL UNet: one forward pass, a different LoRA per sample.

The S-LoRA / Punica idea applied to diffusion: keep one base UNet in memory, stack every user's LoRA
weights (A: [K, r, in], B: [K, out, r]), and in each LoRA'd linear layer gather the A/B pair for each
sample of the batch and apply it with two batched matmuls:

    y = base(x) + scale * (x @ A[idx]^T) @ B[idx]^T        idx: [batch] user index per sample

Slot 0 is an all-zero LoRA ("no LoRA"). Loads diffusers-format LoRA files as saved by train_fast.py /
the diffusers training script (keys "unet.<module>.lora.down.weight" / ".lora.up.weight").
"""
import torch
import torch.nn as nn
from safetensors.torch import load_file

TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


class Selector:
    """Shared holder for the per-sample adapter index of the current batch."""

    idx = None


class MultiLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, max_adapters: int, selector: Selector):
        super().__init__()
        self.base, self.selector = base, selector
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.zeros(max_adapters, rank, base.in_features, device=dev, dtype=dt), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(max_adapters, base.out_features, rank, device=dev, dtype=dt), requires_grad=False)
        self.scale = torch.ones(max_adapters, device=dev, dtype=dt)

    def forward(self, x):
        y = self.base(x)
        idx = self.selector.idx
        if idx is None:
            return y
        a, b = self.A[idx], self.B[idx]  # [batch, r, in], [batch, out, r]
        squeeze = x.dim() == 2
        xs = x[:, None, :] if squeeze else x  # [batch, tokens, in]
        delta = torch.bmm(torch.bmm(xs, a.transpose(1, 2)), b.transpose(1, 2))
        delta = delta * self.scale[idx].view(-1, 1, 1)
        return y + (delta[:, 0] if squeeze else delta)


def wrap_unet(unet, rank: int, max_adapters: int):
    """Replace every target linear in the UNet with a MultiLoRALinear. Returns (selector, {path: module})."""
    selector, wrapped = Selector(), {}
    for name, module in list(unet.named_modules()):
        for child_name, child in list(module.named_children()):
            path = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(path.endswith(t) for t in TARGETS):
                m = MultiLoRALinear(child, rank, max_adapters, selector)
                setattr(module, child_name, m)
                wrapped[path] = m
    return selector, wrapped


def load_into_slot(wrapped, path_or_sd, slot: int, scale: float = 1.0):
    """Copy one diffusers-format LoRA into adapter slot `slot`. Returns the number of layers loaded."""
    sd = load_file(path_or_sd) if isinstance(path_or_sd, str) else path_or_sd
    for m in wrapped.values():  # a reused slot must not keep the previous user's weights
        m.A.data[slot].zero_()
        m.B.data[slot].zero_()
    n = 0
    for key, w in sd.items():
        if not key.endswith(".lora.down.weight"):
            continue
        path = key[len("unet."):-len(".lora.down.weight")]
        m = wrapped.get(path)
        if m is None:
            continue
        up = sd[key.replace(".lora.down.weight", ".lora.up.weight")]
        alpha_key = key.replace(".lora.down.weight", ".alpha")
        r = w.shape[0]
        if r > m.A.shape[1]:
            raise ValueError(f"LoRA rank {r} > slot rank {m.A.shape[1]}")
        alpha = float(sd[alpha_key]) if alpha_key in sd else r  # alpha = rank in our training
        m.A.data[slot, :r] = w.to(m.A.dtype)
        m.B.data[slot, :, :r] = up.to(m.B.dtype)
        m.scale[slot] = scale * alpha / r
        n += 1
    return n


def random_like(sd, seed: int):
    """A synthetic user LoRA with the same shapes and per-tensor scale as `sd` (for serving benchmarks)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in sd.items():
        t = v.float()
        out[k] = (torch.randn(t.shape, generator=g) * t.std().clamp_min(1e-6)).to(v.dtype) if t.dim() > 0 else v
    return out
