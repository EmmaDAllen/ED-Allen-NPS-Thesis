# -*- coding: utf-8 -*-
"""
Created on Sat May 30 06:03:28 2026

@author: emmallen
"""

import torch
import networkx as nx

def compute_metrics(logits, y, mask, attack_limits):
    
    """Computes validation metrics by converting model logits into a top-k interdiction 
    decision and comparing that decision to the optimal MIP solution."""

    # number of graphs in the batch
    batch_size = logits.shape[0]

    # running totals for accuracy calculation
    total_correct = 0
    total_edges = 0

    # running totals for precision, recall, and F1
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # running totals for graph-level metrics
    total_hamming = 0
    exact_match = 0

    # evaluate each graph in the batch separately
    for b in range(batch_size):
        
        # keep only real edges, ignore padded edges
        real_logits = logits[b][mask[b]]
        real_y = y[b][mask[b]]

        # select the top-k scoring edges as interdicted
        # k equals the interdiction budget for this graph
        k = int(attack_limits[b].item())
        
        # initialiaze prediction vector
        pred = torch.zeros_like(real_y)

        # find the k highest-scoring edges
        topk_indices = torch.topk(real_logits, k=k).indices
        
        # mark k highest scoring edges as interdicted
        pred[topk_indices] = 1.0

        # edge level classification accuracy
        total_correct += (pred == real_y).sum().item()
        total_edges += real_y.numel()

        # compute true positives, false positives, and false negatives for
        # precision and recall
        tp = ((pred == 1) & (real_y == 1)).sum().item()
        fp = ((pred == 1) & (real_y == 0)).sum().item()
        fn = ((pred == 0) & (real_y == 1)).sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Hamming distance = number of edge labels that differ
        total_hamming += (pred != real_y).sum().item()

        # exact match = predicted interdictio nset equals optimeal MIP 
        # interdiction set
        if torch.equal(pred, real_y):
            exact_match += 1

    # overall edge classification accuracy
    accuracy = total_correct / total_edges

    # overall precision, recall, and F1 score
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # average hamming distance and exact match rates
    avg_hamming = total_hamming / batch_size
    exact_match_rate = exact_match / batch_size

    return accuracy, precision, recall, f1, avg_hamming, exact_match_rate



def shortest_path_after_attack(sample, predicted_attack):
    
    """Computes the shortest-path length produced by a predicted interdiction decision."""

    # create directed graph from sample data    
    G = nx.DiGraph()

    # define source and sink nodes
    s = sample["source"]
    t = sample["sink"]

    # rebuild network using predicted interdiction decisions
    for u, v, dist, penalty, attack in zip(sample["u"], sample["v"], sample["dist"], sample["penalty"],
                                  predicted_attack):
        
        # interdicted edges receive an additional cost penalty
        new_dist = dist + penalty*attack
        G.add_edge(u, v, dist=new_dist)

    # compute resulting shortest path length
    return nx.shortest_path_length(G, source=s, target=t, weight="dist")



def max_flow_after_attack(sample, predicted_attack):

    """Computes the penalized maximum-flow objective produced by
    a fixed predicted interdiction decision."""

    nodes = range(sample["n_nodes"])
    arcs = list(zip(sample["u"], sample["v"]))

    s = sample["source"]
    t = sample["sink"]

    capacity = {arc: cap for arc, cap in zip(arcs, sample["capacity"])}

    penalty = {arc: q for arc, q in zip(arcs, sample["penalty"])}

    attack = {arc: y for arc, y in zip(arcs, predicted_attack)}

    model = pyo.ConcreteModel()

    model.N = pyo.Set(initialize=nodes)
    model.A = pyo.Set(initialize=arcs, dimen=2)

    # Flow on original arcs
    model.X = pyo.Var(model.A,within=pyo.NonNegativeReals)

    # Artificial return flow from sink to source
    model.XReturn = pyo.Var(within=pyo.NonNegativeReals)

    # Arc capacity constraints
    def capacity_rule(model, i, j):
        return model.X[i,j] <= capacity[i,j]
    model.capacity_constraints = pyo.Constraint(model.A,rule=capacity_rule)

    # Flow conservation
    def flow_balance_rule(model, node):
        inflow = sum(model.X[i,j] for i, j in model.A if j == node)

        outflow = sum(model.X[i,j] for i, j in model.A if i == node)

        if node == s:
            inflow += model.XReturn

        if node == t:
            outflow += model.XReturn

        return inflow == outflow

    model.flow_balance = pyo.Constraint(model.N, rule=flow_balance_rule)

    # Penalized maximum-flow objective
    model.objective = pyo.Objective(
        expr=model.XReturn - sum(penalty[i,j]*attack[i,j]*model.X[i,j] for i, j in model.A),
        sense=pyo.maximize)

    solver = pyo.SolverFactory("gurobi")
    solver.solve(model)

    return pyo.value(model.objective)



def min_cost_flow_after_attack(sample, predicted_attack):

    """Computes the minimum cost of sending one unit from source to sink
    after predicted interdiction."""

    G = nx.DiGraph()

    s = sample["source"]
    t = sample["sink"]
    flow_demand = sample.get("flow_demand", 1)

    # NetworkX uses negative demand for supply and positive demand
    # for required inflow.
    for node in range(sample["n_nodes"]):
        G.add_node(node, demand=0)

    G.nodes[s]["demand"] = -flow_demand
    G.nodes[t]["demand"] = flow_demand

    for u, v, dist, capacity, penalty, attack in zip(sample["u"],sample["v"],sample["dist"],sample["capacity"],
                                                     sample["penalty"],predicted_attack):
        new_cost = dist + penalty * attack

        G.add_edge(u,v,weight=new_cost,capacity=capacity)

    return nx.min_cost_flow_cost(G,demand="demand",capacity="capacity",weight="weight")