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

    """Standard dot-product attention with optional graph edge bias."""

    def __init__(self, d_model, n_heads):

        super(EdgeBiasAttention, self).__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)

        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, edge_bias=None, mask=None):

        batch_size, seq_len, d_model = x.shape

        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # Standard scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add graph edge-bias structure to attention scores
        if edge_bias is not None:
            scores = scores + edge_bias.unsqueeze(1)

        # Prevent attention to padded edges
        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, V)

        context = (
            context.permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, d_model)
        )

        return self.out(context)


class EdgeBiasTransformerBlock(nn.Module):

    """Transformer block using edge-biased attention."""

    def __init__(self, d_model, n_heads, dropout=0.1):

        super(EdgeBiasTransformerBlock, self).__init__()

        self.attn = EdgeBiasAttention(d_model, n_heads)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_bias=None, mask=None):

        # Attention residual block
        residual = x
        attn_out = self.attn(x, edge_bias=edge_bias, mask=mask)
        x = self.norm1(residual + self.dropout(attn_out))

        # Feed-forward residual block
        residual = x
        ff_out = self.ff(x)
        x = self.norm2(residual + self.dropout(ff_out))

        return x


class EdgeBiasTransformerInterdictionModel(nn.Module):

    """Standard Transformer baseline with graph edge bias."""

    def __init__(self, input_dim=7, d_model=64, n_heads=4,
                 num_layers=2, dropout=0.1):

        super(EdgeBiasTransformerInterdictionModel, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList([
            EdgeBiasTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, edge_features, edge_bias=None, mask=None):

        x = self.input_proj(edge_features)

        for layer in self.layers:
            x = layer(x, edge_bias=edge_bias, mask=mask)

        logits = self.classifier(x).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        return logits