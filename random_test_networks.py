# -*- coding: utf-8 -*-
"""
Created on Sun May  3 07:20:00 2026

@author: emmallen
"""

'''Random Test Network Generator

Generates directed, root-connected random test networks using the One-In method.'''

import random
import networkx as nx


def generate_one_in_network(n, m, cost_low=1, cost_high=10, seed=None):
    
    '''Generate a directed random test network using the One-In method.

    Inputs:
        n: number of nodes
        m: number of directed arcs
        cost_low: minimum arc cost
        cost_high: maximum arc cost
        seed: random seed

    Returns:
        G: directed NetworkX graph
        s: source/root node
        t: sink/target node
        density: m / n '''

    rng = random.Random(seed)

    s = 0
    t = n - 1
    density = m / n

    if m < n - 1:
        raise ValueError("Need m >= n - 1 to make root-connectivity possible.")

    if m > n * (n - 1):
        raise ValueError("Too many arcs. Maximum for directed graph without self-loops is n*(n-1).")

    # True = if len(reachable_nodes) == n
    while True:
        
        # create directed graph
        G = nx.DiGraph()
        # add nodes, 0 to n-1
        G.add_nodes_from(range(n))


        # for every node
        for j in range(n):
            # if node = start node - skip it
            if j == s:
                continue
            # if nodes != start node - generate list of predecessors - nodes that could
            # point into node j, excluding node j itself
            possible_predecessors = [i for i in range(n) if i != j]
            # randomly pick a predecessor node
            i = rng.choice(possible_predecessors)
            # add edge from picked predecessor node i to node j
            G.add_edge(i, j)

        # list of every directed arc that can still be added 
        possible_arcs = [(i, j) for i in range(n) for j in range(n)
            if i != j and not G.has_edge(i, j)]

        # compute how many more arcs are needed to reach exactly m total arcs
        remaining_arcs = m - G.number_of_edges()
        
        # randomly sample the number of remaining arcs from possible arcs 
        extra_arcs = rng.sample(possible_arcs, remaining_arcs)
        
        # add edges from randomly sampled extra arcs
        G.add_edges_from(extra_arcs)


        # find all nodes reachable from source node s
        reachable_nodes = nx.descendants(G, s) | {s}

        # Check whether every node is reachable from the source - if yes, 
        # accept the graph and exit the loop, if no, start the loop over and 
        # generates a new graph
        if len(reachable_nodes) == n:
            break

    # assign random arc costs/distances between 1 and 10
    for u, v in G.edges():
        G[u][v]["dist"] = rng.randint(cost_low, cost_high)

    # return graph, source node, sink node, and density
    return G, s, t, density