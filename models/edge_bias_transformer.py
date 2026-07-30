# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 06:25:24 2026

@author: emmallen
"""

"""edge_bias_transformer.py

Transformer baseline with graph-structured attention bias.

This model extends the standard Transformer by adding an additive bias matrix to the attention 
scores before the softmax operation. The bias encodes graph structure so that pairs of edges
with stronger structural relationships receive higher attention weights. This provides a 
stronger graph-aware baseline while retaining the standard Transformer attention mechanism."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeBiasAttention(nn.Module):

    """Multi-head self-attention with additive edge bias.

    This layer computes standard scaled dot-product attention and incorporates graph 
    structure by adding an edge-bias matrix directly to the attention scores prior
    to the softmax operation.

    Larger edge-bias values increase the attention weight between corresponding
    pairs of edge tokens."""

    def __init__(self, d_model, n_heads, dropout=0.1):

        super(EdgeBiasAttention, self).__init__()

        # ensure the hidden dimension can be evenly divided among attention heads
        assert d_model % n_heads == 0

        # number of attention heads and dimension processed by each head
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # learnable linear projections used to construct queries, keys, and values for self-attention
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)

        # combine the outputs from all attention heads into a single embedding
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)


    def forward(self, x, edge_bias=None, mask=None):

        batch_size, seq_len, d_model = x.shape

        # project edge embeddings into Q (query), K (key), V (value) spaces
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        # reshape tensors so each attention head processes a separate portion of the hidden 
        # representation (splits hidden dimension across attention heads)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # compute standard scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

       # incorporate graph structure into the attention mechanism by using edge-bias matrix 
       # which is added directly to the attention scores before the softmax, encouraging structurally 
       # related edges to attend more strongly to one another.
        if edge_bias is not None:
            scores = scores + edge_bias.unsqueeze(1)

        # prevent padded edges from participating in (receiving) attention
        if mask is not None:
            pair_mask = mask.unsqueeze(1).unsqueeze(2) & mask.unsqueeze(1).unsqueeze(3)
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~pair_mask, mask_value)

        # normalize attention scores into attention probabilities
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # aggregate information from neighboring edge embeddings
        context = torch.matmul(attn_weights, V)

        # concatenate all attention heads into a single embedding 
        # (merge attention heads back together)
        context = (context.permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, d_model))

        return self.out(context)

 
class EdgeBiasTransformerBlock(nn.Module):

    """Single edge-bias Transformer encoder block.

    Each block consists of:

        1. Edge-biased multi-head self-attention
        2. Feed-forward network
        3. Residual connections
        4. Layer normalization

    Residual connections improve optimization by allowing each layer to learn refinements
    to the incoming representation rather than entirely new representations."""

    def __init__(self, d_model, n_heads, dropout=0.1):

        super(EdgeBiasTransformerBlock, self).__init__()

        self.attn = EdgeBiasAttention(d_model, n_heads, dropout=dropout)

        # position-wise feed-forward network applied independently to every edge embedding
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model))

        # normalize hidden representations after each residual connection
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_bias=None, mask=None):

        # apply the edge-biased self-attention residual block
        residual = x
        attn_out = self.attn(x, edge_bias=edge_bias, mask=mask)
        x = self.norm1(residual + self.dropout(attn_out))

        # apply the position-wise feed-forward residual block
        residual = x
        ff_out = self.ff(x)
        x = self.norm2(residual + self.dropout(ff_out))

        return x


class EdgeBiasTransformerInterdictionModel(nn.Module):

    """Transformer model with graph-aware attention bias.

    Edge features are embedded into a hidden representation and processed through 
    multiple Transformer encoder blocks that incorporate graph structure via an additive 
    attention bias. A classifier then predicts one interdiction logit for every edge.

    Input shape:
        (batch_size, num_edges, input_dim)

    Output shape:
        (batch_size, num_edges)"""

    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2, dropout=0.1):

        super(EdgeBiasTransformerInterdictionModel, self).__init__()

        # project raw edge features into transformer's hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # stack multiple edge-biased transformer encoder blocks
        self.layers = nn.ModuleList([
            EdgeBiasTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout)

            for _ in range(num_layers)])

        # predict one interdiction logit score for every edge
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1))

    def forward(self, edge_features, edge_bias=None, mask=None):

        # embed raw edge features into the transformer's hidden space
        x = self.input_proj(edge_features)

        # apply each transformer encoder block sequentially
        for layer in self.layers:
            x = layer(x, edge_bias=edge_bias, mask=mask)

        # compute one interdiction logit per edge 
        logits = self.classifier(x).squeeze(-1)

        # assign extremely negative values to padded edges so they cannot be selected during prediction
        if mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, mask_value)

        return logits