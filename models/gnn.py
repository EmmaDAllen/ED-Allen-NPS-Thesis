# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:57:07 2026

@author: emmallen
"""

"""graph_neural_network.py

Graph Neural Network (GNN) baseline for edge interdiction.

This model represents each edge as a learnable embedding and updates edge 
representations through message passing over an edge-adjacency graph. After 
several rounds of neighborhood aggregation, a classifier predicts one interdiction
 logit for every edge.

Unlike the Transformer models, information is exchanged only between adjacent edges
 rather than through global self-attention."""

import torch
import torch.nn as nn


class SimpleGNNLayer(nn.Module):

    """Single edge-to-edge message-passing layer.

    Each edge embedding is updated by combining its current representation with the 
    average representation of its neighboring edges. Neighborhood relationships are 
    defined by the supplied edge-adjacency matrix."""

    def __init__(self, d_model):

        super(SimpleGNNLayer, self).__init__()

        # learn a transformation for the edge's own features
        self.self_linear = nn.Linear(d_model, d_model)
        # learn a separate transformation for aggregated neighbor features
        self.neighbor_linear = nn.Linear(d_model, d_model)
        # apply a nonlinear activation after combining the two sources of information
        self.activation = nn.ReLU()

    def forward(self, x, adjacency, mask=None):

        """Perform one round of edge message passing.

        Parameters:
        x : torch.Tensor
            Edge embeddings with shape (batch_size, num_edges, d_model).

        adjacency : torch.Tensor
            Edge adjacency matrix with shape (batch_size, num_edges, num_edges).

        mask : torch.BoolTensor, optional
            Boolean tensor indicating valid (True) and padded (False) edge positions.

        Returns:
        torch.Tensor
            Updated edge embeddings with shape (batch_size, num_edges, d_model)."""

        # convert the weighted edge-bias matrix into a binary adjacency
        # matrix indicating whether two edges are neighbors
        adjacency = (adjacency > 0).float()

        # remove padded edges so they neither send nor receive messages
        if mask is not None:
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            adjacency = adjacency * pair_mask.float()

        # count the number of neighbors for each edge when computing
        # the average message
        degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)

        # aggregate neighboring edge embeddings by averaging
        neighbor_messages = torch.bmm(adjacency, x) / degree

        # combine the transformed self-embedding with the transformed
        # neighborhood information
        out = self.self_linear(x) + self.neighbor_linear(neighbor_messages)

        # apply a nonlinear activation before returning the updated edge embedding
        return self.activation(out)


class GNNInterdictionModel(nn.Module):

    """Graph Neural Network baseline for network interdiction.

    Raw edge features are embedded into a hidden representation and refined through
    multiple rounds of edge-to-edge message passing. A classifier then predicts one
    interdiction logit for every edge.

    Input shape: (batch_size, num_edges, input_dim)

    Output shape: (batch_size, num_edges)"""

    def __init__(self, input_dim, d_model=64, num_layers=2):

        super(GNNInterdictionModel, self).__init__()

        # project raw edge features into the shared hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # stack multiple message-passing layers so information can propagate across
        # the graph
        self.layers = nn.ModuleList([
            SimpleGNNLayer(d_model)
            for _ in range(num_layers)])

        # predict one interdiction logit per edge
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1))


    def forward(self, edge_features, edge_bias=None, mask=None):

        """Perform a forward pass through the GNN.

        Parameters:
        edge_features : torch.Tensor
            Edge feature tensor with shape (batch_size, num_edges, input_dim).

        edge_bias : torch.Tensor
            Edge adjacency matrix used for message passing.

        mask : torch.BoolTensor, optional
            Boolean tensor identifying valid (True) and padded (False) edges.

        Returns
        torch.Tensor
            Interdiction logits for every edge with shape (batch_size, num_edges)."""

        # message passing requires an edge-adjacency matrix
        if edge_bias is None:
            raise ValueError("GNNInterdictionModel requires edge_bias adjacency.")

        # use the supplied graph structure as the edge adjacency matrix
        edge_adjacency = edge_bias

        # embed raw edge features into the hidden representation
        x = self.input_proj(edge_features)

        # apply one message-passing layer and add a residual connection
        # to preserve information from the previous representation
        for layer in self.layers:
            residual = x
            x = layer(x, adjacency=edge_adjacency, mask=mask)
            x = x + residual

        # compute one interdiction logit for every edge
        logits = self.classifier(x).squeeze(-1)

        # assign extremely negative values to padded edges so they
        # cannot be selected during prediction
        if mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, mask_value)

        return logits