# -*- coding: utf-8 -*-
"""
Created on Sat May 30 06:03:28 2026

@author: emmallen
"""

"""metrics.py

Metric and follower-problem evaluation functions for network
interdiction models.

This file contains two types of functions:

1. Prediction metrics
   compute_metrics() converts model logits into a discrete top-k
   interdiction decision and compares that decision with the MIP-optimal
   attack labels.

2. Downstream objective evaluation
   shortest_path_after_attack(), max_flow_after_attack(), and
   min_cost_flow_after_attack() rebuild the network after applying a
   model-predicted attack and solve the corresponding follower problem.

These functions are used during validation and final model evaluation."""

import torch
import networkx as nx

def compute_metrics(logits, y, mask, attack_limits):
    
    """Compare top-k model predictions with MIP-optimal attack labels.

    Each graph may contain a different number of real edges and a different interdiction
    budget. The function therefore processes each graph separately, removes padded positions
    using the mask, selects the k highest-scoring real edges, and compares that attack 
    vector with the optimal MIP attack vector.

    Parameters
    logits : torch.Tensor
        Model prediction scores with shape (batch_size, padded_num_edges).
        Larger logits indicate that an edge is more likely to be
        selected for interdiction.

    y : torch.Tensor
        Binary MIP attack labels with shape (batch_size, padded_num_edges).
        For each real edge:
            1 = edge belongs to the MIP-optimal interdiction set
            0 = edge does not belong to the MIP-optimal set.

    mask : torch.BoolTensor
        Boolean tensor with shape (batch_size, padded_num_edges).
        True indicates a real edge.
        False indicates padding added during batching.

    attack_limits : torch.Tensor
        Interdiction budget for each graph with shape (batch_size,).
        Each value gives the number of edges that must be selected for
        the corresponding graph.

        
    Returns
    tuple
        Raw totals for the complete batch:
        total_correct : int
            Number of correctly classified real edge labels.

        total_edges : int
            Number of real edge labels evaluated.

        total_tp : int
            Number of true-positive interdicted edges.

        total_fp : int
            Number of false-positive interdicted edges.

        total_fn : int
            Number of false-negative interdicted edges.

        total_hamming : int
            Total number of edge-label disagreements across graphs.

        exact_match : int
            Number of graphs whose complete predicted attack vector
            exactly matches the MIP-optimal attack vector.

        batch_size : int
            Number of graph samples in the batch.

    Notes: This function returns raw counts rather than already averaged metrics. 
    The training pipeline accumulates these totals across all validation batches 
    before calculating accuracy, precision, recall, F1, average Hamming distance,
    and exact-match rate."""      


    # number of graph samples in the current padded batch
    batch_size = logits.shape[0]

    # count correctly classified real edge labels across the batch
    total_correct = 0

    # count the total number of real edges evaluated across the batch
    total_edges = 0

    # count positive-class classification outcomes across the batch
    # true positive: model predicts interdiction and MIP label is interdiction
    # false positive: model predicts interdiction but MIP label is not interdiction
    # false negative: model does not predict interdiction but MIP label is interdiction
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # sum the number of edge-label disagreements across all graphs
    total_hamming = 0
 
    # count graphs whose complete predicted attack vector equals the MIP-optimal attack vector   
    exact_match = 0

    # graphs must be processed separately because they may have different numbers of real 
    # edges and different attack budgets
    for b in range(batch_size):
        
        # remove padded edge positions from this graph's logits
        # real_logits shape: (num_real_edges,)
        real_logits = logits[b][mask[b]]

        # remove padded positions from the MIP attack labels
        # real_y shape: (num_real_edges,)
        real_y = y[b][mask[b]]

        # retrieve the number of edges the model must select for this graph
        k = int(attack_limits[b].item())

        # count the real edges available for selection
        num_real_edges = real_logits.numel()

        # the interdiction budget must be feasible for this graph
        if not 0 <= k <= num_real_edges:
            raise ValueError(
                f"Invalid attack limit K={k} for graph {b} "
                f"with {num_real_edges} real edges."
            )        
        
        # initialize a binary attack vector in which every real edge is
        # initially predicted as not interdicted
        pred = torch.zeros_like(real_y)

        # select exactly k edges when the attack budget is positive
        # find the k highest-scoring edges
        topk_indices = torch.topk(real_logits, k=k).indices
        
        # mark k highest scoring edges (selected) as interdicted
        pred[topk_indices] = 1.0

        # count edge-level classification agreements - includes both: correctly 
        # predicted interdicted edges, and correctly predicted non-interdicted edges
        total_correct += (pred == real_y).sum().item()

        # add the number of real edge labels in this graph
        total_edges += real_y.numel()

        # Count true-positive edges: predicted attack = 1 and MIP attack label = 1
        tp = ((pred == 1) & (real_y == 1)).sum().item()

        # ount false-positive edges: predicted attack = 1 but MIP attack label = 0
        fp = ((pred == 1) & (real_y == 0)).sum().item()

        # count false-negative edges: predicted attack = 0 but MIP attack label = 1        
        fn = ((pred == 0) & (real_y == 1)).sum().item()

        # accumulate positive-class outcomes across the full batch
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # hamming distance counts every edge position where the predicted and optimal
        # binary labels differ - when both attack sets contain exactly k edges, replacing 
        # one optimal edge with one incorrect edge normally contributes two Hamming errors:
        # one missed optimal edge, and one incorrect selected edge
        total_hamming += (pred != real_y).sum().item()

        # exact match requires every edge label in the graph to agree - stricter than objective
        # equality because two different attack sets may still produce the same downstream objective
        if torch.equal(pred, real_y):
            exact_match += 1

    # return raw totals so the calling training loop can aggregate
    # counts across all validation batches before computing averages
    return (total_correct, total_edges, total_tp,total_fp,total_fn,total_hamming,exact_match,batch_size)



def shortest_path_after_attack(sample, predicted_attack):
    
    """Compute the shortest-path objective produced by a predicted attack.

    The function rebuilds the directed network from the stored sample. Each interdicted
    edge receives its specified interdiction penalty, and NetworkX then computes the resulting
    shortest source-to-sink path length.

    Parameters
    sample : dict
        Solved interdiction sample containing at least: source, sink,u, v, dist, penalty
        The lists u and v define the ordered directed edges. The dist
        and penalty entries use the same edge ordering.

    predicted_attack : sequence
        Binary attack vector aligned with the sample edge ordering.
        For each edge:
            1 = predicted as interdicted
            0 = predicted as not interdicted.


    Returns
    float
        Shortest source-to-sink path length after applying the
        predicted interdiction penalties."""          


    # create a new directed NetworkX graph for the follower shortest-path problem  
    G = nx.DiGraph()

    # retrieve the source and sink node identifiers from the sample
    s = sample["source"]
    t = sample["sink"]

    # rebuild every directed edge using the predicted attack vector - zip preserves the
    # common edge ordering across: tail node, head node, original distance, interdiction penalty,
    # predicted attack label
    for u, v, dist, penalty, attack in zip(sample["u"], sample["v"], sample["dist"], sample["penalty"],
                                  predicted_attack):
        

        # shortest-path interdiction does not remove an edge, instead, an attacked edge receives 
        # an additive cost penalty: new distance = original distance + penalty * attack
        # when attack = 0: new distance = original distance
        # When attack = 1: new distance = original distance + penalty
        new_dist = dist + penalty*attack

        # add the directed edge using its post-attack distance
        G.add_edge(u, v, dist=new_dist)

    # solve the follower shortest-path problem using the modified edge distances
    return nx.shortest_path_length(G, source=s, target=t, weight="dist")





def max_flow_after_attack(sample, predicted_attack):

    """Compute surviving maximum flow after a predicted interdiction.

    Maximum-flow interdiction is modeled by removing the capacity of every attacked arc. 
    The function reconstructs the graph with post-attack capacities and computes the 
    remaining maximum source-to-sink flow.

    Parameters
    sample : dict
        Solved interdiction sample containing at least: source, sink, u, v, capacity

    predicted_attack : sequence
        Binary attack vector aligned with the sample edge ordering.
        For each edge:
            1 = edge is interdicted and loses all capacity
            0 = edge retains its original capacity.

    Returns
    float
        Maximum source-to-sink flow remaining after the predicted
        attack."""

    # create a new directed graph for the follower maximum-flow problem
    G = nx.DiGraph()

    # retrieve the source and sink node identifiers from the sample
    s = sample["source"]
    t = sample["sink"]

    # rebuild every directed edge using the predicted attack vector - zip preserves the
    # common edge ordering across: tail node, head node, original capacity, predicted attack label
    for u, v, capacity, attack in zip(sample["u"],sample["v"],sample["capacity"],predicted_attack):

        # an attacked arc loses all physical capacity:
        # attack = 0 == remaining capacity = original capacity
        # attack = 1 == remaining capacity = 0
        remaining_capacity = capacity * (1 - attack)

        # add the arc with its surviving capacity
        G.add_edge(u,v,capacity=remaining_capacity)

    # solve the follower maximum-flow problem on the attacked network
    return nx.maximum_flow_value(G,s=s,t=t,capacity="capacity")





def min_cost_flow_after_attack(sample, predicted_attack):

    """Compute minimum feasible flow cost after a predicted interdiction.

    The function reconstructs the directed capacitated network and applies an additive
    cost penalty to every attacked edge. It then solves the follower minimum-cost-flow 
    problem for the demand stored in the sample.

    Parameters
    sample : dict
        Solved interdiction sample containing at least: n_nodes, source, sink, u, v, dist, 
                                                        capacity, penalty
        The sample may also contain: flow_demand
        If flow_demand is absent, the function defaults to one unit.

    predicted_attack : sequence
        Binary attack vector aligned with the sample edge ordering.
        For each edge:
            1 = edge receives its interdiction cost penalty
            0 = edge retains its original cost.

    Returns
    float
        Minimum cost required to send the specified demand from source
        to sink after applying the predicted attack.

    Raises
    networkx.NetworkXUnfeasible
        If the attacked network cannot satisfy the required flow demand."""

    # create a new directed graph for the follower minimum-cost-flow problem
    G = nx.DiGraph()

    # retrieve the source and sink identifiers
    s = sample["source"]
    t = sample["sink"]

    # use the stored demand selected during data generation or evaluation, default to
    # one unit for backward compatibility with samples that do not contain flow_demand
    flow_demand = sample.get("flow_demand", 1)

    # NetworkX represents node supply and demand using one "demand" attribute:
    # negative value = node supplies flow, positive value = node requires inflow,
    # zero = transshipment node - initialize every graph node as a transshipment node
    for node in range(sample["n_nodes"]):
        G.add_node(node, demand=0)

    # source supplies flow_demand units
    G.nodes[s]["demand"] = -flow_demand
    # sink requires flow_demand units
    G.nodes[t]["demand"] = flow_demand


    # rebuild every directed edge using the predicted attack vector - zip preserves the
    # common edge ordering across: tail node, head node, original distance, originalcapacity, 
    # interdictionpenalty, predicted attack label
    for u, v, dist, capacity, penalty, attack in zip(sample["u"],sample["v"],sample["dist"],sample["capacity"],
                                                     sample["penalty"],predicted_attack):

        # min-cost-flow interdiction leaves physical capacity unchanged but increases the 
        # per-unit flow cost of attacked arcs: new cost = original cost + penalty * attack       
        new_cost = dist + penalty * attack

        # add the directed arc using its post-attack cost and original capacity
        G.add_edge(u,v,weight=new_cost,capacity=capacity)

    # solve the follower minimum-cost-flow problem and return only the optimal total flow cost
    return nx.min_cost_flow_cost(G,demand="demand",capacity="capacity",weight="weight")