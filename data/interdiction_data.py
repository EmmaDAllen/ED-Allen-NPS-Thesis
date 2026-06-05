# -*- coding: utf-8 -*-
"""
Created on Sat May 30 05:57:09 2026

@author: emmallen
"""

import json
import torch
from torch.utils.data import Dataset


def sample_to_tensors(sample):
    
    # store number of nodes
    n = sample["n_nodes"]
    # store density = # edges (m)  / # nodes (n)
    density = sample["density"]
    # store interdiction budget
    attack_limit = sample["attack_limit"]

    # converts edge heads to PyTorch tensor
    u = torch.tensor(sample["u"], dtype=torch.float32)
    # converts edge tails to PyTorch tensor
    v = torch.tensor(sample["v"], dtype=torch.float32)
    # converts edge distances to PyTorch tensor
    dist = torch.tensor(sample["dist"], dtype=torch.float32)
    # converts MIP attack labels to PyTorch tensor
    y = torch.tensor(sample["attack"], dtype=torch.float32)

    # Normalize node IDs and distances
    u_norm = u / max(n - 1, 1)
    v_norm = v / max(n - 1, 1)
    dist_norm = dist / 10.0

    # source flag = 1 if edge leaves source node
    source_flag = (u == sample["source"]).float()
    # sink flag = 1 if edge leaves sink node
    sink_flag = (v == sample["sink"]).float()

    # creates one density valye for every edge
    density_feature = torch.full_like(u_norm, density / 10.0)
    # creates one budget value for every edge
    budget_feature = torch.full_like(u_norm, attack_limit / 10.0)

    # combines edge features into one matrix
    edge_features = torch.stack([
        u_norm, v_norm, dist_norm, source_flag, sink_flag, density_feature,
        budget_feature], dim=1)

    # edge bias: edge i can flow into edge j if v_i == u_j
    # creates edge to edge connectivity matrix
    edge_bias = (v.unsqueeze(1) == u.unsqueeze(0)).float()

    # gives connected edge pairs a positive attention bias
    edge_bias = 0.5 * edge_bias

    return edge_features, edge_bias, y, attack_limit

class InterdictionDataset(Dataset):
    
    ''' class that creates a custom PyTorch dataset for the interdiction graphs'''
    
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # get graph at index idx
        sample = self.data[idx]
        return sample_to_tensors(sample)
    
    
def collate_graphs(batch):
    
    '''defines how to combine multiple graphs into one batch, graphs have different
    number of edges, so they need padding'''
    
    # finds largest number of edges in batch
    max_edges = max(item[0].shape[0] for item in batch)
    # gets # of edge features
    input_dim = batch[0][0].shape[1]

    # creates empty lists to store padded graphs
    edge_features_batch = []
    edge_bias_batch = []
    y_batch = []
    mask_batch = []
    attack_limits_batch = []

    # loops through each graph in the batch
    for edge_features, edge_bias, y, attack_limit in batch:
        # stores the number of real edges in the graph
        m = edge_features.shape[0]

        # creates a zero matrix for padded edge features
        edge_features_padded = torch.zeros(max_edges, input_dim)
        # copies real edges into the top part of matrix
        edge_features_padded[:m] = edge_features
        
        # creates padded edge bias matrix
        edge_bias_padded = torch.zeros(max_edges, max_edges)
        # copies real edge-bias values inot it
        edge_bias_padded[:m, :m] = edge_bias

        # creates padded labels
        y_padded = torch.zeros(max_edges)
        # copies real labels
        y_padded[:m] = y

        # creates a mask saying which edges are real
        mask = torch.zeros(max_edges, dtype=torch.bool)
        # marks real edges as true
        mask[:m] = True

        # adds the padded graph to the batch lists
        edge_features_batch.append(edge_features_padded)
        edge_bias_batch.append(edge_bias_padded)
        y_batch.append(y_padded)
        mask_batch.append(mask)
        attack_limits_batch.append(attack_limit)


    # stacks all graphs into batch tensors
    # edge_features: [batch_size, max_edges, 7]
    # edge_bias: [batch_size, max_edges, max_edges]
    # y:[batch_size, max_edges]
    # mask: [batch_size, max_edges]
    return (torch.stack(edge_features_batch),
        torch.stack(edge_bias_batch),
        torch.stack(y_batch),
        torch.stack(mask_batch),
        torch.tensor(attack_limits_batch, dtype=torch.long))
