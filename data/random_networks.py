# -*- coding: utf-8 -*-
"""
Created on Sun May  3 07:20:00 2026

@author: emmallen
"""

"""Random test-network generators for network-interdiction experiments.

This module generates directed networks with designated source and sink nodes.
The primary One-In generator creates root-connected random graphs and assigns
arc cost, interdiction penalty, and capacity attributes."""

import random
import networkx as nx
import math


def generate_one_in_network(n, m, cost_low=1, cost_high=10, penalty_low=1, penalty_high=10, capacity_low=1, 
                            capacity_high=20, seed=None):
    
    '''Generate a directed random test network using the One-In method.

    Inputs:
        n: number of nodes
        m: number of directed arcs
        cost_low: minimum arc cost
        cost_high: maximum arc cost
        penalty_low: minimum arc penalty
        penalty_high: maximum arc penalty
        capacity_low: minimum arc capacity
        capacity_high: maximum arc capacity
        seed: random seed

    Returns:
        G: directed NetworkX graph
        s: source/root node
        t: sink/target node
        experimental density (arc-to-node ratio): m / n '''

    # initialize random number generator with seed
    rng = random.Random(seed)

    # initialize source and sink nodes, and compute density
    s = 0 # 1st node is always the source node
    t = n - 1 # last node is always the sink node
    density = m / n # experimental arc to node ratio

    # checks that m is at least n-1 to ensure root connectivity is possible
    if m < n - 1:
        raise ValueError("Need m >= n - 1 to make root-connectivity possible.")

    # checks that m is not too large for a directed graph without self-loops
    if m > n * (n - 1):
        raise ValueError("Too many arcs. Maximum for directed graph without self-loops is n*(n-1).")

    # True = if len(reachable_nodes) == n
    # rejection sampling loop to ensure that every node is reachable from the source node
    while True:
        
        # create directed graph
        G = nx.DiGraph()
        # add nodes, 0 to n-1 
        G.add_nodes_from(range(n))

        # for every node
        for j in range(n):
            # if node = start node - skip it
            # guarantees that every node has at least one incoming arc, except for the source node
            if j == s:
                continue
            # if nodes != start node - generate list of predecessors - nodes that could
            # point into node j, excluding node j itself
            possible_predecessors = [i for i in range(n) if i != j]
            # randomly pick a predecessor node, each candidate has equal probability of being selected
            i = rng.choice(possible_predecessors)
            # add edge from picked predecessor node i to node j
            G.add_edge(i, j)

        # list of every directed arc that can still be added 
        possible_arcs = [(i, j) for i in range(n) for j in range(n)
            if i != j and not G.has_edge(i, j)]

        # compute how many more arcs are needed to reach exactly m total arcs
        remaining_arcs = m - G.number_of_edges()
        
        # randomly sample the number of remaining arcs from possible arcs, without replacement 
        extra_arcs = rng.sample(possible_arcs, remaining_arcs)
        
        # add edges from randomly sampled extra arcs
        G.add_edges_from(extra_arcs)

        # find all nodes reachable from source node s
        reachable_nodes = nx.descendants(G, s) | {s}

        # Check whether every node is reachable from the source - if yes, accept the graph and exit 
        # the loop, if no, start the loop over and generates a new graph
        if len(reachable_nodes) == n:
            break

    # assign random arc costs/distances, penalties, and capacities between lower and 
    # upper bounds above
    for u, v in G.edges():
        G[u][v]["dist"] = rng.randint(cost_low, cost_high)
        G[u][v]["penalty"] = rng.randint(penalty_low, penalty_high)
        G[u][v]["capacity"] = rng.randint(capacity_low, capacity_high)

    # return graph, source node, sink node, and density
    return G, s, t, density
