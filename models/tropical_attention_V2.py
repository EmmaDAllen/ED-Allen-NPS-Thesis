# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:50:16 2026

@author: emmallen
"""


"""tropical_attention_V2.py

Version 2 of the Tropical Attention Transformer for edge interdiction.

This implementation uses generalized tropical matrix multiplication for the model's
max-plus and min-plus operations. When available, the TensorBFS tropical-gemm package 
provides optimized implementations. Otherwise, the model automatically uses an equivalent
PyTorch fallback.

The mathematical attention mechanism remains consistent with Version 1: ordinary learned 
Q/K/V projections are followed by tropical projections, attention scores are based on symmetric
Hilbert projective distance, and context vectors are computed through max-plus aggregation.

The model retains the common forward interface:
    logits = model(
        edge_features,
        edge_bias=edge_bias,
        mask=mask,)"""

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


_TENSORBFS_IMPORT_ERROR = None

# attempt to load the optional TensorBFS tropical matrix-multiplication backend 
# model remains functional using the PyTorch fallback when the package is unavailable
try:
    import tropical_gemm.pytorch as _tg
    _TENSORBFS_AVAILABLE = True
except Exception as _e:
    _tg = None
    _TENSORBFS_AVAILABLE = False
    _TENSORBFS_IMPORT_ERROR = _e

# when REQUIRE_TENSORBFS=1, prevent silent fallback and require the optimized TensorBFS 
# backend to be available
_REQUIRE_TENSORBFS = os.getenv("REQUIRE_TENSORBFS", "0") == "1"

# track whether the fallback warning has already been issued so it
# appears only once during execution
_WARNED_FALLBACK = False


def print_tropical_backend():

    """Print diagnostic information about the active tropical backend.

    The function reports whether TensorBFS tropical-gemm was imported,
    whether its GPU backend is available, and which tropical matrix
    multiplication functions were detected."""

    if not _TENSORBFS_AVAILABLE:
        # report the import error and indicate that the PyTorch fallback will be used
        print(f"TensorBFS tropical-gemm: NOT AVAILABLE ({_TENSORBFS_IMPORT_ERROR})")
        print("Using PyTorch fallback.")
        return

    # identify all tropical matrix-multiplication functions exposed  by the installed TensorBFS module
    funcs = sorted(name for name in dir(_tg) if "tropical" in name and "matmul" in name)
    # TensorBFS may expose its GPU availability through GPU_AVAILABLE - use "unknown" when the 
    # installed version does not define it
    gpu_available = getattr(_tg, "GPU_AVAILABLE", "unknown")
    print(f"TensorBFS tropical-gemm: AVAILABLE | GPU_AVAILABLE={gpu_available}")
    print("matmul functions found:", funcs)


def _torch_tropical_mm(a, b, mode):

    """Compute tropical matrix multiplication using native PyTorch.

    For matrices A and B, max-plus multiplication computes

        C[i, j] = max_k(A[i, k] + B[k, j]),

    while min-plus multiplication computes

        C[i, j] = min_k(A[i, k] + B[k, j]).

    Leading batch dimensions are handled through PyTorch broadcasting.

    Parameters:
    a : torch.Tensor
        Left input with shape (..., m, k).
    b : torch.Tensor
        Right input with shape (..., k, n).
    mode : {"maxplus", "minplus"}
        Tropical semiring used for the reduction.

    Returns:
    torch.Tensor
        Tropical matrix product with shape (..., m, n)."""

    # max-plus multiplication replaces ordinary multiplication with
    # addition and ordinary summation with a maximum
    if mode == "maxplus":
        return (a.unsqueeze(-1) + b.unsqueeze(-3)).amax(dim=-2)

    # min-plus multiplication replaces ordinary multiplication with
    # addition and ordinary summation with a minimum
    if mode == "minplus":
        return (a.unsqueeze(-1) + b.unsqueeze(-3)).amin(dim=-2)

    
    raise ValueError(f"Unknown tropical matmul mode: {mode}")


def _get_tensorbfs_fn(mode, device_type, batched):

    """Find the most appropriate TensorBFS tropical-matmul function.

    Candidate function names are ordered from most specialized to most general. 
    Batched GPU implementations are preferred when the inputs are batched CUDA tensors."""

    # TensorBFS cannot supply a function when the package was not imported
    if not _TENSORBFS_AVAILABLE:
        return None

    names = []

    # prefer a batched GPU implementation when both capabilities are required by the input
    if batched and device_type == "cuda":
        names += [f"tropical_{mode}_matmul_batched_gpu", f"tropical_{mode}_matmul_gpu_batched",]

    # fall back successively to batched, GPU-only, and generic implementations
    if batched:
        names += [f"tropical_{mode}_matmul_batched"]
    if device_type == "cuda":
        names += [f"tropical_{mode}_matmul_gpu"]

    names += [f"tropical_{mode}_matmul"]

    # return the first function exposed by the installed TensorBFS version
    for name in names:
        fn = getattr(_tg, name, None)
        if fn is not None:
            return fn

    return None



def _expand_leading(x, lead_shape):

    """Broadcast a matrix tensor across a target set of leading dimensions.

    The final two dimensions are treated as matrix dimensions and are preserved. Any 
    missing leading dimensions are inserted as singleton dimensions before broadcasting.

    Parameters:
    x : torch.Tensor
        Tensor with shape (..., rows, columns).
    lead_shape : tuple
        Desired broadcast shape preceding the final matrix dimensions.

    Returns:
    torch.Tensor: Broadcast tensor with shape (*lead_shape, rows, columns)."""


    # determine how many leading singleton dimensions must be inserted
    # before broadcasting
    pad = len(lead_shape) - len(x.shape[:-2])

    # tensor cannot be broadcast to a target with fewer leading
    # dimensions than it already contains
    if pad < 0:
        raise ValueError(
            f"Cannot broadcast shape {tuple(x.shape)} to leading shape {tuple(lead_shape)}")

    # insert missing dimensions and broadcast across the requested leading shape while 
    # preserving the two matrix dimensions
    return x.reshape((1,) * pad + tuple(x.shape)).expand(*lead_shape, *x.shape[-2:])




def tropical_mm(a, b, mode="maxplus"):

    """Compute broadcast-compatible tropical matrix multiplication.

    The function attempts to use TensorBFS tropical-gemm when available. If an appropriate
    optimized implementation cannot be used, it falls back to an equivalent native PyTorch 
    implementation.

    Parameters:
    a : torch.Tensor
        Left operand with shape (..., m, k).
    b : torch.Tensor
        Right operand with shape (..., k, n).
    mode : {"maxplus", "minplus"}, default="maxplus"
        Tropical matrix multiplication operation.

    Returns:
    torch.Tensor = Tropical matrix product with shape (..., m, n).

    Raises: 
    ValueError if the inner matrix dimensions are incompatible.
    RuntimeError if TensorBFS is explicitly required but unavailable or does not expose a 
    required multiplication function."""

    global _WARNED_FALLBACK

    # inner matrix dimensions must agree, just as they must for ordinary matrix multiplication
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"Bad tropical matmul shapes: {tuple(a.shape)} and {tuple(b.shape)}")

    # preserve the original dtype so the final result can be returned consistently with the
    # model inputs
    original_dtype = a.dtype

    # TensorBFS operations are performed using contiguous float32 tensors
    a = a.contiguous().float()
    b = b.contiguous().float()

    if not _TENSORBFS_AVAILABLE:
        if _REQUIRE_TENSORBFS:
            raise RuntimeError(
                "REQUIRE_TENSORBFS=1 but tropical-gemm is not importable. "
                "Install with: python -m pip install 'tropical-gemm[torch]'")

        if not _WARNED_FALLBACK:
            warnings.warn(
                "TensorBFS tropical-gemm not found; using PyTorch fallback.",
                RuntimeWarning,)
            
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
        except Exception as exc:
            warnings.warn(
            f"TensorBFS batched {mode} multiplication failed. "
            f"Using another implementation. Error: {exc}",
            RuntimeWarning,)

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
    def __init__(self,d_model,n_heads,device=None,tropical_proj=True,tropical_norm=False,
                 symmetric=True,use_edge_bias=True):
        super().__init__()

        assert d_model % n_heads == 0

        if not symmetric:
            raise ValueError("This implementation only supports symmetric=True.")

        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.tropical_proj = tropical_proj
        self.tropical_norm = tropical_norm
        self.symmetric = symmetric
        self.use_edge_bias = use_edge_bias

        # Add ordinary learned Q, K, and V projections.
        self.query_linear = nn.Linear(d_model,d_model,bias=False,)
        self.key_linear = nn.Linear(d_model,d_model,bias=False,)
        self.value_linear = nn.Linear(d_model,d_model,bias=False,)

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

        # Ordinary Q, K, and V projections.
        q = self.query_linear(x)
        k = self.key_linear(x)
        v = self.value_linear(x)

        q = torch.log1p(F.relu(q))
        k = torch.log1p(F.relu(k))
        v = torch.log1p(F.relu(v))


        if self.tropical_norm:
            q = self.normalize_tropical(q)
            k = self.normalize_tropical(k)
            v = self.normalize_tropical(v)

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

        if self.use_edge_bias and edge_bias is not None:

            expanded_bias = edge_bias.unsqueeze(1).expand(-1,self.n_heads,-1,-1,)

            expanded_bias = expanded_bias.reshape(bh,seq_len,seq_len,)

            attn_scores = (attn_scores + expanded_bias)


        if mask is not None:
            head_mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1)
            head_mask = head_mask.reshape(bh, seq_len)

            pair_mask = head_mask.unsqueeze(1) & head_mask.unsqueeze(2)

            mask_value = torch.finfo(attn_scores.dtype).min

            attn_scores = attn_scores.masked_fill(~pair_mask, mask_value)

        context = tropical_mm(attn_scores, v, mode="maxplus")

        context = (context.reshape(batch_size, self.n_heads, seq_len, self.d_k)
            .permute(0, 2, 1, 3).contiguous()
            .reshape(batch_size, seq_len, self.d_model))

        context = torch.expm1(context).to(dtype=x_dtype)

        output = self.out(context)

        return output, attn_scores




class TropicalTransformerBlock(nn.Module):

    def __init__(self,d_model,n_heads,device="cpu",dropout=0.1,use_edge_bias=True,):

        super().__init__()

        self.attn = TropicalAttention(d_model=d_model,n_heads=n_heads,device=device,tropical_proj=True,
                                      tropical_norm=False,symmetric=True,use_edge_bias=use_edge_bias,)

        self.ff = nn.Sequential(
            nn.Linear(d_model,4 * d_model,),
            nn.ReLU(),
            nn.Linear(4 * d_model,d_model,),)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_bias=None, mask=None):

        residual = x

        attn_out, attn_scores = self.attn(x,edge_bias=edge_bias,mask=mask,)

        x = self.norm1(residual + self.dropout(attn_out))

        residual = x

        ff_out = self.ff(x)

        x = self.norm2(residual + self.dropout(ff_out))

        return x, attn_scores




class TropicalInterdictionModel(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2, dropout=0.1, 
                 device="cpu",use_edge_bias=True):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList([
            TropicalTransformerBlock(d_model=d_model,n_heads=n_heads,device=device,
                    dropout=dropout, use_edge_bias=use_edge_bias)
                for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),)

    def forward(self, edge_features, edge_bias=None, mask=None):
        x = self.input_proj(edge_features)

        for layer in self.layers:
            x, _ = layer(x, edge_bias=edge_bias, mask=mask)

        logits = self.classifier(x).squeeze(-1)

        if mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, mask_value)

        return logits


TropicalAttentionV2 = TropicalAttention
TropicalInterdictionModelV2 = TropicalInterdictionModel