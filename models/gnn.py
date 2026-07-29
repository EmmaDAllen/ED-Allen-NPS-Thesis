# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:57:07 2026

@author: emmallen
"""

# graph_neural_network.py

import torch
import torch.nn as nn


class SimpleGNNLayer(nn.Module):

    """Simple edge-to-edge message passing layer."""

    def __init__(self, d_model):

        super(SimpleGNNLayer, self).__init__()

        self.self_linear = nn.Linear(d_model, d_model)
        self.neighbor_linear = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()

    def forward(self, x, adjacency, mask=None):

        # Convert edge bias into binary adjacency
        adjacency = (adjacency > 0).float()

        # Remove padded rows/columns from contributing messages
        if mask is not None:
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            adjacency = adjacency * pair_mask.float()

        # Count number of neighbors for averaging
        degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)

        # Average neighbor features
        neighbor_messages = torch.bmm(adjacency, x) / degree

        # Combine self features and neighbor messages
        out = self.self_linear(x) + self.neighbor_linear(neighbor_messages)

        return self.activation(out)


class GNNInterdictionModel(nn.Module):

    """Graph neural network baseline for edge interdiction prediction."""

    def __init__(self, input_dim, d_model=64, num_layers=2):

        super(GNNInterdictionModel, self).__init__()

        # Project raw edge features into hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Stack message passing layers
        self.layers = nn.ModuleList([
            SimpleGNNLayer(d_model)
            for _ in range(num_layers)
        ])

        # Predict one interdiction score per edge
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, edge_features, edge_bias=None, mask=None):

        if edge_bias is None:
            raise ValueError("GNNInterdictionModel requires edge_bias adjacency.")

        adjacency = edge_bias

        # Convert edge features into hidden embeddings
        x = self.input_proj(edge_features)

        # Apply message passing layers with residual connections
        for layer in self.layers:
            residual = x
            x = layer(x, adjacency=adjacency, mask=mask)
            x = x + residual

        # Predict edge scores
        logits = self.classifier(x).squeeze(-1)

        # Prevent padded edges from being selected
        if mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, mask_value)

        return logits