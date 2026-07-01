"""
Minimal drop-in TropicalAttentionV3 for Baran-phys/Tropical-Attention style code.

Drop this file next to the script that currently imports TropicalAttention.
If their script already says:

    from TropicalAttention import TropicalAttention

then no import change should be needed.

Install TensorBFS tropical-gemm first when possible:

    python -m pip install "tropical-gemm[torch]"

To force an error if TensorBFS is not available, run with:

    REQUIRE_TENSORBFS=1 python your_script.py
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
    """Small helper to confirm what backend the run is using."""
    if not _TENSORBFS_AVAILABLE:
        print(f"TensorBFS tropical-gemm: NOT AVAILABLE ({_TENSORBFS_IMPORT_ERROR})")
        print("Using PyTorch fallback. Fine for smoke tests; not recommended for large HPC runs.")
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
    raise ValueError(mode)


def _get_tensorbfs_fn(mode, device_type, batched):
    """Find a TensorBFS function if the installed version exposes one."""
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
    """Broadcast x over leading dimensions, preserving the last two matrix dims."""
    pad = len(lead_shape) - len(x.shape[:-2])
    if pad < 0:
        raise ValueError(f"Cannot broadcast shape {tuple(x.shape)} to leading shape {tuple(lead_shape)}")
    return x.reshape((1,) * pad + tuple(x.shape)).expand(*lead_shape, *x.shape[-2:])


def tropical_mm(a, b, mode="maxplus"):
    """
    Tropical matrix multiply.

    maxplus: C[i,j] = max_k A[i,k] + B[k,j]
    minplus: C[i,j] = min_k A[i,k] + B[k,j]

    Handles [..., M, K] x [..., K, N]. Uses TensorBFS if installed; otherwise PyTorch fallback.
    """
    global _WARNED_FALLBACK

    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"Bad tropical matmul shapes: {tuple(a.shape)} and {tuple(b.shape)}")

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
                "TensorBFS tropical-gemm not found; using PyTorch fallback. "
                "This may be memory-heavy for large sequence lengths.",
                RuntimeWarning,
            )
            _WARNED_FALLBACK = True
        return _torch_tropical_mm(a, b, mode)

    lead_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    m, k = a.shape[-2:]
    n = b.shape[-1]
    a3 = _expand_leading(a, lead_shape).reshape(-1, m, k).contiguous()
    b3 = _expand_leading(b, lead_shape).reshape(-1, k, n).contiguous()
    batch = a3.shape[0]

    # Prefer real batched TensorBFS functions if the installed version exposes them.
    fn_batched = _get_tensorbfs_fn(mode, a3.device.type, batched=True)
    if fn_batched is not None:
        try:
            out = fn_batched(a3, b3)
            return out.reshape(*lead_shape, m, n)
        except Exception:
            # Some versions expose the name but only support 2D. Fall through to 2D loop.
            pass

    # Otherwise loop over batch with the TensorBFS 2D function. Slower, but avoids [B,S,S,D] memory blowup.
    fn_2d = _get_tensorbfs_fn(mode, a3.device.type, batched=False)
    if fn_2d is None:
        if _REQUIRE_TENSORBFS:
            raise RuntimeError(f"TensorBFS installed, but no function found for {mode}.")
        return _torch_tropical_mm(a, b, mode)

    outs = [fn_2d(a3[i], b3[i]) for i in range(batch)]
    return torch.stack(outs, dim=0).reshape(*lead_shape, m, n)


class TropicalLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, x):
        return tropical_mm(x, self.W.transpose(0, 1), mode="maxplus").to(dtype=x.dtype)


class TropicalAttention(nn.Module):
    """
    Drop-in replacement for the original TropicalAttention class.

    Same constructor shape as Baran-phys/Tropical-Attention:
        TropicalAttention(d_model, n_heads, device, tropical_proj=True, tropical_norm=False, symmetric=True)

    V3 change:
        max(q-k) is computed as maxplus(q, -k^T)
        min(q-k) is computed as minplus(q, -k^T)

    This avoids explicitly materializing diff = q.unsqueeze(2) - k.unsqueeze(1).
    """

    def __init__(self, d_model, n_heads, device=None, tropical_proj=True, tropical_norm=False, symmetric=True):
        super().__init__()
        assert d_model % n_heads == 0
        if not symmetric:
            raise ValueError("This minimal V3 drop-in only implements symmetric=True.")

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

    def forward(self, x):
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

        # V3 TensorBFS path:
        # max_l(q_i,l - k_j,l) = maxplus(q, (-k)^T)
        # min_l(q_i,l - k_j,l) = minplus(q, (-k)^T)
        neg_k_t = (-k).transpose(1, 2).contiguous()
        max_diff = tropical_mm(q, neg_k_t, mode="maxplus")
        min_diff = tropical_mm(q, neg_k_t, mode="minplus")
        attn_scores = -(max_diff - min_diff)

        # Tropical aggregation over values: context_i,d = max_j(attn_scores_i,j + v_j,d)
        context = tropical_mm(attn_scores, v, mode="maxplus")

        context = (
            context.reshape(batch_size, self.n_heads, seq_len, self.d_k)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, self.d_model)
        )

        context = torch.expm1(context).to(dtype=x_dtype)
        output = self.out(context)
        return output, attn_scores


# Optional alias if they want to import the name explicitly.
TropicalAttentionV3 = TropicalAttention
