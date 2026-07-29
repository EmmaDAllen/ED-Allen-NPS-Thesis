# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 06:25:24 2026

@author: emmallen
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeBiasAttention(nn.Module):

    """Multi-head self-attention with an additive graph-structure bias.

    edge_bias[b, i, j] is added directly to the attention score between
    edge-token i and edge-token j. Larger values increase attention."""

    def __init__(self, d_model, n_heads, dropout=0.1):

        super(EdgeBiasAttention, self).__init__()

        # hidden dimension must be divisible by the number of heads
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # learnable projections used to construct queries, keys, and values for attention
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)

        # recombine outputs from all attention heads
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, edge_bias=None, mask=None):

        batch_size, seq_len, d_model = x.shape

        # project edge embeddings into Q, K, V spaces
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        # split hidden dimension across attention heads
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # standard scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # inject graph structure into attention
        # higher edge_bias values increase attention scores
        if edge_bias is not None:
            scores = scores + edge_bias.unsqueeze(1)

        # prevent padded edges from receiving attention
        if mask is not None:
            pair_mask = mask.unsqueeze(1).unsqueeze(2) & mask.unsqueeze(1).unsqueeze(3)
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~pair_mask, mask_value)

        # convert scores into attention probabilities
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # aggregate information from all edges
        context = torch.matmul(attn_weights, V)

        # merge attention heads back together
        context = (context.permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, d_model))

        return self.out(context)

 
class EdgeBiasTransformerBlock(nn.Module):

    """Standard Transformer encoder block consisting of:

    1. Edge-biased multi-head attention
    2. Feed-forward network
    3. Residual connections
    4. Layer normalization

    Residual connections preserve information from earlier layers
    while allowing the network to learn refinements."""

    def __init__(self, d_model, n_heads, dropout=0.1):

        super(EdgeBiasTransformerBlock, self).__init__()

        self.attn = EdgeBiasAttention(d_model, n_heads, dropout=dropout)

        # position-wise feed forward network 
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model))

        # normalize hidde nrepresentations after each residual block
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_bias=None, mask=None):

        # multi-head attention residual block
        residual = x
        attn_out = self.attn(x, edge_bias=edge_bias, mask=mask)
        x = self.norm1(residual + self.dropout(attn_out))

        # feed-forward residual block
        residual = x
        ff_out = self.ff(x)
        x = self.norm2(residual + self.dropout(ff_out))

        return x


class EdgeBiasTransformerInterdictionModel(nn.Module):

    """Transformer baseline for network interdiction.

    Input: Edge feature vectors

    Output: One interdiction score (logit) per edge

    Graph structure is incorporated through edge_bias,
    which modifies attention scores between edges."""

    def __init__(self, input_dim, d_model=64, n_heads=4,
                 num_layers=2, dropout=0.1):

        super(EdgeBiasTransformerInterdictionModel, self).__init__()

        # project raw edge featrues into transformer hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # stack transformer encoder blocks
        self.layers = nn.ModuleList([
            EdgeBiasTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout)

            for _ in range(num_layers)])

        # predict one interdiction score per edge
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1))

    def forward(self, edge_features, edge_bias=None, mask=None):

        # convert raw edge features into hidden embeddings
        x = self.input_proj(edge_features)

        # apply stacked transformer layers
        for layer in self.layers:
            x = layer(x, edge_bias=edge_bias, mask=mask)

        # produce one logit per edge 
        logits = self.classifier(x).squeeze(-1)

        # ensure padded edges are not selected 
        if mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, mask_value)

        return logits