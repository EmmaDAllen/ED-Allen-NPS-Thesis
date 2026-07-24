# -*- coding: utf-8 -*-
"""
Created on Sat May 30 05:57:09 2026

@author: emmallen
"""

"""Dataset preprocessing utilities for network interdiction models.

This module converts graph instances stored as JSON into PyTorch tensors,
constructs edge-level feature vectors, computes edge-to-edge attention
bias matrices, and pads variable-sized graphs into mini-batches suitable
for transformer training."""

import json
import torch
from torch.utils.data import Dataset

# Reference maximum values used for feature normalization.
COST_HIGH = 10
PENALTY_HIGH = 10
CAPACITY_HIGH = 20


def sample_to_tensors(sample):

    """Convert one graph instance into tensors used for model training.

    The returned tensors include:

        - edge feature matrix
        - edge-to-edge attention bias matrix
        - interdiction labels
        - attack budget

    Feature construction depends on the interdiction problem type so that
    each model receives only the attributes relevant to that optimization
    problem."""
    
    # number of nodes used to normalize node indices
    n = sample["n_nodes"]
    # graph density (m / n), included as a global graph-level feature
    density = sample["density"]
    # number of edges the attacker is allowed to interdict
    attack_limit = sample["attack_limit"]
    # edge specific interdiction penalties as PyTorch tensor
    penalty = torch.tensor(sample["penalty"], dtype=torch.float32)

    # edge tails and head node indices as PyTorch tensor
    u = torch.tensor(sample["u"], dtype=torch.float32)
    v = torch.tensor(sample["v"], dtype=torch.float32)

    # binary labels indicating whether each edge belongs to the optimal attack as PyTorch tensor
    y = torch.tensor(sample["attack"], dtype=torch.float32)

    # normalize numerical features to approximately [0,1] to improve optimization during neural network training
    u_norm = u / max(n - 1, 1)
    v_norm = v / max(n - 1, 1)
    penalty_norm = penalty / PENALTY_HIGH # reference max penalty value

    # source flag = 1 if edge leaves source node
    source_flag = (u == sample["source"]).float()
    # sink flag = 1 if edge enters sink node
    sink_flag = (v == sample["sink"]).float()

    # broadcast graph-level density to every edge
    density_feature = torch.full_like(u_norm, density / 10.0)
    # broadcast attack budget to every edge
    budget_feature = torch.full_like(u_norm, attack_limit / 10.0)

    # Construct problem-specific edge features
    problem_type = sample.get("problem_type", "shortest_path")


    if problem_type == "shortest_path":

        # shortest path uses distances
        dist = torch.tensor(sample["dist"], dtype=torch.float32)
        dist_norm = dist / COST_HIGH # normalize to improve optimization

         # Edge feature matrix: [tail, head, distance, source flag, sink flag, density, budget, penalty]
        edge_features = torch.stack([u_norm, v_norm, dist_norm, source_flag, sink_flag, 
                                     density_feature, budget_feature, penalty_norm], dim=1)
        
    
    elif problem_type == "max_flow":

        # max flow uses capacities
        capacity = torch.tensor(sample["capacity"], dtype=torch.float32)
        capacity_norm = capacity / CAPACITY_HIGH # normalize  to improve optimization

        # Edge feature matrix: [head, tail, capacity, source flag, sink flag, density, budget]
        edge_features = torch.stack([u_norm, v_norm, capacity_norm, source_flag, sink_flag, 
                                     density_feature, budget_feature], dim=1)
        
    
    elif problem_type == "min_cost_flow":

        # min cost flow uses distances and capacities
        dist = torch.tensor(sample["dist"], dtype=torch.float32)
        capacity = torch.tensor(sample["capacity"], dtype=torch.float32)

        # normalize both to improve optimization
        dist_norm = dist / COST_HIGH
        capacity_norm = capacity / CAPACITY_HIGH

        # Edge feature matrix: [tail, head, distance, capacity, source flag, sink flag, density, budget, penalty]
        edge_features = torch.stack([u_norm, v_norm, dist_norm, capacity_norm, source_flag, sink_flag,
                                      density_feature, budget_feature, penalty_norm], dim=1)
        
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


    # Construct an edge-to-edge connectivity matrix:
    # entry (i,j) equals 1 when the head node of edge i matches the tail node of edge j, 
    # meaning the two edges can be traversed consecutively in a path
    edge_bias = (v.unsqueeze(1) == u.unsqueeze(0)).float()

    # scale the connectivity values to create an additive attention bias.
    edge_bias = 0.5 * edge_bias

    return edge_features, edge_bias, y, attack_limit


class InterdictionDataset(Dataset):
    
    """PyTorch Dataset for network interdiction instances.

    Each dataset element corresponds to one solved graph stored in the training JSON file."""
    
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # convert the selected graph into tensors
        sample = self.data[idx]
        return sample_to_tensors(sample)
    
    
def collate_graphs(batch):
    
    """Collate variable-sized graph instances into a padded mini-batch.

    Graphs generally contain different numbers of edges, so tensors are padded to match the 
    largest graph in the batch. A Boolean mask is returned so padded entries can be ignored
    during training."""
    
    # maximum edge count among all graphs in the batch
    max_edges = max(item[0].shape[0] for item in batch)
    # number of features per edge
    input_dim = batch[0][0].shape[1]

    # initialization of lists that accumulate padded graph tensors
    edge_features_batch = []
    edge_bias_batch = []
    y_batch = []
    mask_batch = []
    attack_limits_batch = []

    # loops through each graph in the batch
    for edge_features, edge_bias, y, attack_limit in batch:
        # stores the number of valid (unpadded) edges
        m = edge_features.shape[0]

        # creates a zero matrix for padded edge features
        edge_features_padded = torch.zeros(max_edges, input_dim)
        # copies real edges into the top part of matrix
        edge_features_padded[:m] = edge_features
        
        # creates padded edge bias matrix
        edge_bias_padded = torch.zeros(max_edges, max_edges)
        # copies real edge-bias values into it
        edge_bias_padded[:m, :m] = edge_bias

        # allocated padded label vector
        y_padded = torch.zeros(max_edges)
        # copies real labels
        y_padded[:m] = y

        # boolean mask identifying valid (unpadded) edges
        mask = torch.zeros(max_edges, dtype=torch.bool)
        # padded entries = False, unpadded entries = True 
        mask[:m] = True

        # append the padded graph to the batch 
        edge_features_batch.append(edge_features_padded)
        edge_bias_batch.append(edge_bias_padded)
        y_batch.append(y_padded)
        mask_batch.append(mask)
        attack_limits_batch.append(attack_limit)


    # Returned tensors:
    # edge_features = [batch_size, max_edges, input_dim], edge_bias = [batch_size, max_edges, max_edges]
    # labels = [batch_size, max_edges], mask = [batch_size, max_edges], attack_budget = [batch_size]
    return (torch.stack(edge_features_batch),torch.stack(edge_bias_batch),torch.stack(y_batch),
        torch.stack(mask_batch),torch.tensor(attack_limits_batch, dtype=torch.long))
