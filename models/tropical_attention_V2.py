# -*- coding: utf-8 -*-
"""
Tropical Attention V3 model for edge interdiction.

Keeps the same forward call:
    logits = model(edge_features, edge_bias=edge_bias, mask=mask)
"""

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tropical_gemm.pytorch as _tg
    _TENSORBFS_AVAILABLE = True
except Exception as _e:
    _tg = None
    _TENSORBFS_AVAILABLE = False
    _TENSORBFS_IMPORT_ERROR = _e

_REQUIRE_TENSORBFS = os.getenv("REQUIRE_TENSORBFS", "0") == "1"
_WARNED_FALLBACK = False


def print_tropical_backend():
    if not _TENSORBFS_AVAILABLE:
        print(f"TensorBFS tropical-gemm: NOT AVAILABLE ({_TENSORBFS_IMPORT_ERROR})")
        print("Using PyTorch fallback.")
        return

    funcs = sorted(name for name in dir(_tg) if "tropical" in name and "matmul" in name)
    gpu_available = getattr(_tg, "GPU_AVAILABLE", "unknown")
    print(f"TensorBFS tropical-gemm: AVAILABLE | GPU_AVAILABLE={gpu_available}")
    print("matmul functions found:", funcs)


def _torch_tropical_mm(a, b, mode):
    if mode == "maxplus":
        return (a.unsqueeze(-1) + b.unsqueeze(-3)).amax(dim=-2)
    if mode == "minplus":
        return (a.unsqueeze(-1) + b.unsqueeze(-3)).amin(dim=-2)
    raise ValueError(f"Unknown tropical matmul mode: {mode}")


def _get_tensorbfs_fn(mode, device_type, batched):
    if not _TENSORBFS_AVAILABLE:
        return None

    names = []

    if batched and device_type == "cuda":
        names += [
            f"tropical_{mode}_matmul_batched_gpu",
            f"tropical_{mode}_matmul_gpu_batched",
        ]

    if batched:
        names += [f"tropical_{mode}_matmul_batched"]

    if device_type == "cuda":
        names += [f"tropical_{mode}_matmul_gpu"]

    names += [f"tropical_{mode}_matmul"]

    for name in names:
        fn = getattr(_tg, name, None)
        if fn is not None:
            return fn

    return None


def _expand_leading(x, lead_shape):
    pad = len(lead_shape) - len(x.shape[:-2])

    if pad < 0:
        raise ValueError(
            f"Cannot broadcast shape {tuple(x.shape)} to leading shape {tuple(lead_shape)}"
        )

    return x.reshape((1,) * pad + tuple(x.shape)).expand(
        *lead_shape, *x.shape[-2:]
    )


def tropical_mm(a, b, mode="maxplus"):
    global _WARNED_FALLBACK

    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Bad tropical matmul shapes: {tuple(a.shape)} and {tuple(b.shape)}"
        )

    original_dtype = a.dtype

    a = a.contiguous().float()
    b = b.contiguous().float()

    if not _TENSORBFS_AVAILABLE:
        if _REQUIRE_TENSORBFS:
            raise RuntimeError(
                "REQUIRE_TENSORBFS=1 but tropical-gemm is not importable. "
                "Install with: python -m pip install 'tropical-gemm[torch]'"
            )

        if not _WARNED_FALLBACK:
            warnings.warn(
                "TensorBFS tropical-gemm not found; using PyTorch fallback.",
                RuntimeWarning,
            )
            _WARNED_FALLBACK = True

        return _torch_tropical_mm(a, b, mode).to(dtype=original_dtype)

    lead_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])

    m, k = a.shape[-2:]
    n = b.shape[-1]

    a3 = _expand_leading(a, lead_shape).reshape(-1, m, k).contiguous()
    b3 = _expand_leading(b, lead_shape).reshape(-1, k, n).contiguous()

    batch = a3.shape[0]

    fn_batched = _get_tensorbfs_fn(mode, a3.device.type, batched=True)

    if fn_batched is not None:
        try:
            out = fn_batched(a3, b3)
            return out.reshape(*lead_shape, m, n).to(dtype=original_dtype)
        except Exception:
            pass

    fn_2d = _get_tensorbfs_fn(mode, a3.device.type, batched=False)

    if fn_2d is None:
        if _REQUIRE_TENSORBFS:
            raise RuntimeError(f"TensorBFS installed, but no function found for {mode}.")
        return _torch_tropical_mm(a, b, mode).to(dtype=original_dtype)

    outs = [fn_2d(a3[i], b3[i]) for i in range(batch)]

    return torch.stack(outs, dim=0).reshape(*lead_shape, m, n).to(dtype=original_dtype)


class TropicalLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, x):
        return tropical_mm(x, self.W.transpose(0, 1), mode="maxplus")


class TropicalAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        device=None,
        tropical_proj=True,
        tropical_norm=False,
        symmetric=True,
    ):
        super().__init__()

        assert d_model % n_heads == 0

        if not symmetric:
            raise ValueError("This V3 implementation only supports symmetric=True.")

        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.tropical_proj = tropical_proj
        self.tropical_norm = tropical_norm
        self.symmetric = symmetric

        self.out = nn.Linear(d_model, d_model, bias=False)

        if self.tropical_proj:
            self.query_trop = TropicalLinear(self.d_k, self.d_k)
            self.key_trop = TropicalLinear(self.d_k, self.d_k)
            self.value_trop = TropicalLinear(self.d_k, self.d_k)

        if self.tropical_norm:
            self.lambda_param = nn.Parameter(torch.ones(1, 1, d_model, device=device))

        if device is not None:
            self.to(device)

    def normalize_tropical(self, x):
        return x - self.lambda_param

    def forward(self, x, edge_bias=None, mask=None):
        batch_size, seq_len, _ = x.size()
        x_dtype = x.dtype

        x_pos = torch.log1p(F.relu(x))

        if self.tropical_norm:
            x_pos = self.normalize_tropical(x_pos)

        q = x_pos
        k = x_pos
        v = x_pos

        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        bh = batch_size * self.n_heads

        q = q.reshape(bh, seq_len, self.d_k)
        k = k.reshape(bh, seq_len, self.d_k)
        v = v.reshape(bh, seq_len, self.d_k)

        if self.tropical_proj:
            q = self.query_trop(q)
            k = self.key_trop(k)
            v = self.value_trop(v)

        neg_k_t = (-k).transpose(1, 2).contiguous()

        max_diff = tropical_mm(q, neg_k_t, mode="maxplus")
        min_diff = tropical_mm(q, neg_k_t, mode="minplus")

        attn_scores = -(max_diff - min_diff)

        if edge_bias is not None:
            edge_bias = edge_bias.unsqueeze(1)
            edge_bias = edge_bias.repeat(1, self.n_heads, 1, 1)
            edge_bias = edge_bias.reshape(bh, seq_len, seq_len)

            attn_scores = attn_scores + edge_bias

        if mask is not None:
            head_mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1)
            head_mask = head_mask.reshape(bh, seq_len)

            pair_mask = head_mask.unsqueeze(1) & head_mask.unsqueeze(2)

            attn_scores = attn_scores.masked_fill(~pair_mask, -1e9)

        context = tropical_mm(attn_scores, v, mode="maxplus")

        context = (
            context.reshape(batch_size, self.n_heads, seq_len, self.d_k)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, self.d_model)
        )

        context = torch.expm1(context).to(dtype=x_dtype)

        output = self.out(context)

        return output, attn_scores


class TropicalInterdictionModel(nn.Module):
    def __init__(self, input_dim=8, d_model=64, n_heads=4, num_layers=2, device="cpu"):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList(
            [
                TropicalAttention(
                    d_model=d_model,
                    n_heads=n_heads,
                    device=device,
                    tropical_proj=True,
                    tropical_norm=False,
                    symmetric=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, edge_features, edge_bias=None, mask=None):
        x = self.input_proj(edge_features)

        for layer in self.layers:
            residual = x
            x, _ = layer(x, edge_bias=edge_bias, mask=mask)
            x = x + residual

        logits = self.classifier(x).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        return logits


TropicalAttentionV2 = TropicalAttention
TropicalInterdictionModelV2 = TropicalInterdictionModel