# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:50:16 2026

@author: emmallen
"""

# standard_transformer.py

import torch
import torch.nn as nn


class StandardTransformerInterdictionModel(nn.Module):

    """Standard Transformer baseline for edge interdiction prediction."""

    def __init__(self, input_dim=7, d_model=64, n_heads=4,
                 num_layers=2, dropout=0.1):

        super(StandardTransformerInterdictionModel, self).__init__()

        # Project raw edge features into transformer hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # One transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )

        # Stack multiple encoder layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Predict one interdiction score per edge
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, edge_features, edge_bias=None, mask=None):

        # Convert edge features into hidden embeddings
        x = self.input_proj(edge_features)

        # Transformer expects True for padded positions in src_key_padding_mask
        if mask is not None:
            padding_mask = ~mask
        else:
            padding_mask = None

        # Apply standard transformer encoder
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # Predict edge scores
        logits = self.classifier(x).squeeze(-1)

        # Prevent padded edges from being selected
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        return logits