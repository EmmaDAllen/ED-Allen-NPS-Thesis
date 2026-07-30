# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:50:16 2026

@author: emmallen
"""

"""standard_transformer.py

Baseline Transformer model for network interdiction.

This model treats every edge in a graph as a token in a sequence. Edge features are 
projected into a learned embedding space, processed with a standard Transformer 
encoder, and mapped to a single interdiction score (logit) for each edge. Unlike the
Tropical Attention Transformer, this model uses PyTorch's standard self-attention 
mechanism and serves as a baseline for comparison."""

import torch
import torch.nn as nn


class StandardTransformerInterdictionModel(nn.Module):

    """Standard Transformer baseline for network interdiction.

    Parameters:
    input_dim : int = Number of input features describing each edge
    d_model : int, default=64 = Dimension of the hidden Transformer embedding
    n_heads : int, default=4 = Number of attention heads
    num_layers : int, default=2 = Number of stacked Transformer encoder layers
    dropout : float, default=0.1 = Dropout probability used throughout the encoder

    Notes:
    Input shape:(batch_size, num_edges, input_dim)

    Output shape:(batch_size, num_edges)

    Each output value is an unnormalized logit representing how likely the corresponding
    edge should be interdicted."""


    def __init__(self, input_dim, d_model=64, n_heads=4,num_layers=2, dropout=0.1):

        super(StandardTransformerInterdictionModel, self).__init__()

        # learn a dense embedding for each edge feature vector before applying self-attention
        self.input_proj = nn.Linear(input_dim, d_model)

        # define a single Transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,dim_feedforward=4 * d_model,
                                                   dropout=dropout,activation="relu",batch_first=True,norm_first=False)

        # stack multiple encoder blocks to capture increasingly complex relationships between edges
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)

        # map each encoded edge embedding to a single interdiction score
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1))

    def forward(self, edge_features, edge_bias=None, mask=None):

        """Perform a forward pass through the model.

        Parameters:
        edge_features : torch.Tensor
            Tensor of edge features with shape
            (batch_size, num_edges, input_dim).

        edge_bias : torch.Tensor, optional
            Included for interface compatibility with other Transformer
            variants. This model does not use edge bias.

        mask : torch.BoolTensor, optional
            Boolean tensor indicating valid (True) and padded (False)
            edges within each graph.

        Returns
        torch.Tensor: Interdiction logits for every edge with shape (batch_size, num_edges)."""

        # project raw edge features into the Transformer's hidden space
        x = self.input_proj(edge_features)

        # PyTorch expects padded positions to be marked as True, whereas the dataset mask
        # uses True for valid edges
        if mask is not None:
            padding_mask = ~mask
        else:
            padding_mask = None

        # allow every edge to attend to every other edge in the graph
        x = self.encoder(x,src_key_padding_mask=padding_mask)

        # produce one logit per edge
        logits = self.classifier(x).squeeze(-1)

        # assign extremely negative values to padded edges so they are never selected during prediction
        if mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, mask_value)

        return logits