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

    # use the native PyTorch implementation when TensorBFS could not
    # be imported, unless the environment explicitly requires it
    if not _TENSORBFS_AVAILABLE:
        if _REQUIRE_TENSORBFS:
            raise RuntimeError(
                "REQUIRE_TENSORBFS=1 but tropical-gemm is not importable. "
                "Install with: python -m pip install 'tropical-gemm[torch]'")

        # emit the fallback warning only once to avoid repeated messages
        # during every model forward pass
        if not _WARNED_FALLBACK:
            warnings.warn(
                "TensorBFS tropical-gemm not found; using PyTorch fallback.",
                RuntimeWarning,)
            
            _WARNED_FALLBACK = True

        return _torch_tropical_mm(a, b, mode).to(dtype=original_dtype)

    # determine a common broadcast shape for all leading batch
    # dimensions of the two matrix operands
    lead_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])

    # store the matrix dimensions:
    # a has shape (..., m, k), b has shape (..., k, n)
    m, k = a.shape[-2:]
    n = b.shape[-1]

    # broadcast both operands and collapse all leading dimensions into
    # one batch dimension expected by the backend
    a3 = _expand_leading(a, lead_shape).reshape(-1, m, k).contiguous()
    b3 = _expand_leading(b, lead_shape).reshape(-1, k, n).contiguous()

    batch = a3.shape[0]
    fn_batched = _get_tensorbfs_fn(mode, a3.device.type, batched=True)

    # prefer a batched TensorBFS function when one is available
    if fn_batched is not None:
        try:
            out = fn_batched(a3, b3)
            return out.reshape(*lead_shape, m, n).to(dtype=original_dtype)
        
        # if the optimized batched call fails at runtime, issue a warning
        # and attempt another compatible implementation
        except Exception as exc:
            warnings.warn(
            f"TensorBFS batched {mode} multiplication failed. "
            f"Using another implementation. Error: {exc}",
            RuntimeWarning,)

    # search for a non-batched TensorBFS function that can be applied
    # independently to each matrix pair
    fn_2d = _get_tensorbfs_fn(mode, a3.device.type, batched=False)

    # when no TensorBFS function is available, use native PyTorch unless
    # TensorBFS was explicitly required
    if fn_2d is None:
        if _REQUIRE_TENSORBFS:
            raise RuntimeError(f"TensorBFS installed, but no function found for {mode}.")
        return _torch_tropical_mm(a, b, mode).to(dtype=original_dtype)

    outs = [fn_2d(a3[i], b3[i]) for i in range(batch)]

    # apply the 2D backend separately to each matrix pair and restore
    # the original broadcast leading dimensions
    return torch.stack(outs, dim=0).reshape(*lead_shape, m, n).to(dtype=original_dtype)




class TropicalLinear(nn.Module):

    """Learnable max-plus tropical linear transformation.

    For input X and weight matrix W, this layer computes

        Y[..., i, j] = max_k(X[..., i, k] + W[j, k]).

    The operation is implemented through the shared tropical_mm()
    backend so that TensorBFS acceleration can be used when available."""

    def __init__(self, input_dim, output_dim):
        super().__init__()

        # store input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # learnable tropical weight matrix with shape (output_dim, input_dim)
        self.W = nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, x):

        """Apply the max-plus tropical linear transformation.

        The stored weight matrix is transposed from (output_dim, input_dim) to 
        (input_dim, output_dim) so it has the right-hand matrix orientation expected
        by tropical_mm()."""

        return tropical_mm(x, self.W.transpose(0, 1), mode="maxplus")




class TropicalAttention(nn.Module):

    """Multi-head tropical attention implemented with tropical matmul.

    The layer creates ordinary learned query, key, and value projections, maps them into 
    a nonnegative log-domain representation, and optionally applies learnable max-plus 
    tropical projections.

    Pairwise attention scores are the negative symmetric Hilbert projective distance between
    query and key representations. Context vectors are then computed using max-plus matrix 
    multiplication rather than softmax-weighted summation."""

    def __init__(self,d_model,n_heads,device=None,tropical_proj=True,tropical_norm=False,
                 symmetric=True,use_edge_bias=True):

        super().__init__()

        # ensure the hidden dimension can be divided evenly among heads
        assert d_model % n_heads == 0

        # currently implements only the symmetric form of the Hilbert projective distance
        if not symmetric:
            raise ValueError("This implementation only supports symmetric=True.")

        # store model dimensions and attention configuration
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.tropical_proj = tropical_proj
        self.tropical_norm = tropical_norm
        self.symmetric = symmetric
        self.use_edge_bias = use_edge_bias

        # create ordinary learned query, key, and value projections
        self.query_linear = nn.Linear(d_model,d_model,bias=False,)
        self.key_linear = nn.Linear(d_model,d_model,bias=False,)
        self.value_linear = nn.Linear(d_model,d_model,bias=False,)

        # recombine concatenated attention-head outputs in the ordinary vector space
        self.out = nn.Linear(d_model, d_model, bias=False)

        if self.tropical_proj:
            # create an additional max-plus projection for each of the query, key, and value representations
            self.query_trop = TropicalLinear(self.d_k, self.d_k)
            self.key_trop = TropicalLinear(self.d_k, self.d_k)
            self.value_trop = TropicalLinear(self.d_k, self.d_k)

        if self.tropical_norm:
            # optionally create a learned normalization offset for every hidden feature
            self.lambda_param = nn.Parameter(torch.ones(1, 1, d_model))


    def normalize_tropical(self, x):
        return x - self.lambda_param


    def forward(self, x, edge_bias=None, mask=None):

        """Apply multi-head tropical attention.

        Parameters:
        x : torch.Tensor
            Edge embeddings with shape (batch_size, num_edges, d_model).
        edge_bias : torch.Tensor, optional 
            Pairwise graph-structure bias with shape (batch_size, num_edges, num_edges).
        mask : torch.BoolTensor, optional
            Valid-edge mask with shape (batch_size, num_edges), where True indicates a 
            real edge and False indicates padding.

        Returns:
        output : torch.Tensor
            Updated edge embeddings with shape (batch_size, num_edges, d_model).
        attn_scores : torch.Tensor
            Unnormalized tropical attention scores with shape (batch_size * n_heads, num_edges, num_edges)."""

        # extract the batch size and number of edge tokens
        batch_size, seq_len, _ = x.size()

        # preserve the incoming dtype so the final context representation matches the rest of the model
        x_dtype = x.dtype

        # construct separate ordinary query (Q), key (K), and value (V) projections from the shared 
        # input edge embeddings
        q = self.query_linear(x)
        k = self.key_linear(x)
        v = self.value_linear(x)

        # map projected representations into a nonnegative log domain using log(1 + ReLU(x))
        q = torch.log1p(F.relu(q))
        k = torch.log1p(F.relu(k))
        v = torch.log1p(F.relu(v))

        # optionally subtract the learned tropical normalization offset
        if self.tropical_norm:
            q = self.normalize_tropical(q)
            k = self.normalize_tropical(k)
            v = self.normalize_tropical(v)

        # split the hidden dimension across attention heads: (batch_size, num_edges, d_model)
        # == (batch_size, n_heads, num_edges, d_k)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # merge the batch and head dimensions so each head is processed as an independent 
        # matrix in the tropical backend
        bh = batch_size * self.n_heads

        # reshape query, key, and value projections
        q = q.reshape(bh, seq_len, self.d_k)
        k = k.reshape(bh, seq_len, self.d_k)
        v = v.reshape(bh, seq_len, self.d_k)

        # apply learnable max-plus transformations independently to each head's query, key, 
        # and value representations
        if self.tropical_proj:
            q = self.query_trop(q)
            k = self.key_trop(k)
            v = self.value_trop(v)

        # negate and transpose the keys so tropical matrix multiplication computes all 
        # pairwise query-key coordinate differences
        neg_k_t = (-k).transpose(1, 2).contiguous()

        # compute the largest coordinate difference for every query-key pair using max-plus
        # matrix multiplication
        max_diff = tropical_mm(q, neg_k_t, mode="maxplus")

        # compute the smallest coordinate difference for every query-key pair using min-plus
        # matrix multiplication
        min_diff = tropical_mm(q, neg_k_t, mode="minplus")

        # symmetric Hilbert projective distance: d_H(q, k) = max_r(q_r - k_r) - min_r(q_r - k_r)
        # negating the distance converts proximity into an attention score == closer representations 
        # receive larger scores
        attn_scores = -(max_diff - min_diff)

        # replicate the graph-structure bias across all attention heads and reshape it to match 
        # the merged batch-head dimension
        if self.use_edge_bias and edge_bias is not None:

            expanded_bias = edge_bias.unsqueeze(1).expand(-1,self.n_heads,-1,-1,)
            expanded_bias = expanded_bias.reshape(bh,seq_len,seq_len,)

            # add graph structure directly to the unnormalized tropical attention scores
            attn_scores = (attn_scores + expanded_bias)

        # replicate the valid-edge mask across attention heads
        if mask is not None:

            # query-key pair is valid only when both positions correspond to real, non-padded edges
            head_mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1)
            head_mask = head_mask.reshape(bh, seq_len)
            pair_mask = head_mask.unsqueeze(1) & head_mask.unsqueeze(2)
            mask_value = torch.finfo(attn_scores.dtype).min

            # assign invalid pairs the lowest representable value so they cannot win the later
            # max-plus reduction
            attn_scores = attn_scores.masked_fill(~pair_mask, mask_value)

        # compute tropical context vectors: context[i,d] = max_j(attn_scores[i,j] + v[j,d])
        # this max-plus product replaces the softmax-weighted sum used by conventional 
        # Transformer attention
        context = tropical_mm(attn_scores, v, mode="maxplus")

        # restore separate batch and head dimensions, concatenate all heads, and recover shape
        # (batch_size, num_edges, d_model)
        context = (context.reshape(batch_size, self.n_heads, seq_len, self.d_k)
            .permute(0, 2, 1, 3).contiguous()
            .reshape(batch_size, seq_len, self.d_model))

        # map the aggregated context out of the log-domain representation and restore the input dtype
        context = torch.expm1(context).to(dtype=x_dtype)

        # apply the final ordinary output projection
        output = self.out(context)

        return output, attn_scores




class TropicalTransformerBlock(nn.Module):

    """Transformer-style encoder block using Version 2 tropical attention.

    Each block contains tropical multi-head attention, an ordinary
    position-wise feed-forward network, residual connections, layer
    normalization, and dropout.

    The feed-forward and residual architecture matches the baseline
    Transformers so model comparisons isolate the effect of the
    attention mechanism as closely as possible.
    """

    def __init__(self,d_model,n_heads,device="cpu",dropout=0.1,use_edge_bias=True,):

        super().__init__()

        # construct the TensorBFS-compatible tropical-attention layer
        self.attn = TropicalAttention(d_model=d_model,n_heads=n_heads,device=device,tropical_proj=True,
                                      tropical_norm=False,symmetric=True,use_edge_bias=use_edge_bias,)

        # use the same ordinary feed-forward architecture as the baseline Transformer models
        self.ff = nn.Sequential(
            nn.Linear(d_model,4 * d_model,),
            nn.ReLU(),
            nn.Linear(4 * d_model,d_model,),)

        # normalize each residual branch
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, edge_bias=None, mask=None):

        # apply tropical attention and the first residual connection
        residual = x
        attn_out, attn_scores = self.attn(x,edge_bias=edge_bias,mask=mask,)
        x = self.norm1(residual + self.dropout(attn_out))

        # apply the ordinary feed-forward network and second residual connection
        residual = x
        ff_out = self.ff(x)
        x = self.norm2(residual + self.dropout(ff_out))

        return x, attn_scores




class TropicalInterdictionModel(nn.Module):

    """Version 2 Tropical Attention Transformer for edge interdiction.

    Edge features are embedded into a shared hidden representation and
    processed through stacked Version 2 tropical Transformer blocks.
    The final classifier produces one unnormalized interdiction logit
    for every edge.

    Input shape:
        (batch_size, num_edges, input_dim)

    Output shape:
        (batch_size, num_edges)"""

    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2, dropout=0.1, 
                 device="cpu",use_edge_bias=True):
        
        super().__init__()

        # project raw edge features into the shared hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # stack multiple Version 2 tropical Transformer blocks
        self.layers = nn.ModuleList([
            TropicalTransformerBlock(d_model=d_model,n_heads=n_heads,device=device,
                    dropout=dropout, use_edge_bias=use_edge_bias)
                for _ in range(num_layers)])

        # map every final edge embedding to one interdiction logit
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),)

    def forward(self, edge_features, edge_bias=None, mask=None):

        # convert raw edge features into hidden edge embeddings
        x = self.input_proj(edge_features)

        # sequentially refine the embeddings through all tropical Transformer blocks 
        # attention scores are discarded because only edge logits are required for prediction
        for layer in self.layers:
            x, _ = layer(x, edge_bias=edge_bias, mask=mask)

        # produce one raw interdiction score per edge
        logits = self.classifier(x).squeeze(-1)

        if mask is not None:

            mask_value = torch.finfo(logits.dtype).min
            # assign padded positions the lowest representable value so they cannot be selected 
            # as interdicted edges
            logits = logits.masked_fill(~mask, mask_value)

        return logits


TropicalAttentionV2 = TropicalAttention
TropicalInterdictionModelV2 = TropicalInterdictionModel