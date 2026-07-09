# -*- coding: utf-8 -*-
"""
Created on Sun May  3 07:20:00 2026

@author: emmallen
"""

'''Random Test Network Generator

Generates directed, root-connected random test networks using the One-In method.

Creates n nodes, forces every node except the source to have at least one 
incoming connection, randomly adds additional arcs until the graph has exactly m 
arcs, checks whether every node can be reached from the source node - if not, 
it throws the graph away and tries again, once a valid graph is created, it
assigns random costs to every arc.'''

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
        density: m / n '''

    # initialize random number generator with seed
    rng = random.Random(seed)

    # initialize source and sink nodes, and compute density
    s = 0
    t = n - 1
    density = m / n

    # checks that m is at least n-1 to ensure root connectivity is possible
    if m < n - 1:
        raise ValueError("Need m >= n - 1 to make root-connectivity possible.")

    # checks that m is not too large for a directed graph without self-loops
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

        # Check whether every node is reachable from the source - if yes, accept the graph and exit 
        # the loop, if no, start the loop over and generates a new graph
        if len(reachable_nodes) == n:
            break

    # assign random arc costs/distances between 1 and 10
    for u, v in G.edges():
        G[u][v]["dist"] = rng.randint(cost_low, cost_high)
        G[u][v]["penalty"] = rng.randint(penalty_low, penalty_high)
        G[u][v]["capacity"] = rng.randint(capacity_low, capacity_high)

    # return graph, source node, sink node, and density
    return G, s, t, density


def generate_grid_network(rows, cols, cost_low=1, cost_high=10, seed=None):
    
    """Generate a directed grid network with random arc costs.

    Nodes are arranged in a rows x cols grid.
    Arcs connect neighboring grid nodes."""

    rng = random.Random(seed)

    G = nx.DiGraph()

    def node_id(r, c):
        return r * cols + c

    # Add nodes
    for r in range(rows):
        for c in range(cols):
            G.add_node(node_id(r, c), pos=(r, c))

    # Add directed arcs between neighboring nodes
    for r in range(rows):
        for c in range(cols):
            u = node_id(r, c)

            neighbors = []

            if r + 1 < rows:
                neighbors.append(node_id(r + 1, c))
            if c + 1 < cols:
                neighbors.append(node_id(r, c + 1))
            if r - 1 >= 0:
                neighbors.append(node_id(r - 1, c))
            if c - 1 >= 0:
                neighbors.append(node_id(r, c - 1))

            for v in neighbors:
                G.add_edge(u, v, dist=rng.randint(cost_low, cost_high))

    s = node_id(0, 0)
    t = node_id(rows - 1, cols - 1)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = m / n

    return G, s, t, density


def generate_spatial_network(n, k=4, cost_scale=10, seed=None):
    
    """Generate a spatial transportation-style network.

    Nodes are placed randomly in 2D space.
    Each node connects to its k nearest neighbors."""

    rng = random.Random(seed)

    G = nx.DiGraph()
    coords = {}

    # Random 2D node locations
    for i in range(n):
        x = rng.random()
        y = rng.random()
        coords[i] = (x, y)
        G.add_node(i, pos=(x, y))

    # Connect each node to k nearest neighbors
    for i in range(n):
        xi, yi = coords[i]

        distances = []
        for j in range(n):
            if i == j:
                continue

            xj, yj = coords[j]
            distance = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            distances.append((j, distance))

        distances.sort(key=lambda item: item[1])

        for j, distance in distances[:k]:
            cost = max(1, round(cost_scale * distance))
            G.add_edge(i, j, dist=cost)

    s = 0
    t = n - 1

    # Only keep graph if source can reach sink
    if not nx.has_path(G, s, t):
        return generate_spatial_network(n, k, cost_scale, seed=rng.randint(1, 10**9))

    density = G.number_of_edges() / G.number_of_nodes()

    return G, s, t, density


def generate_hub_spoke_network(n, num_hubs=3, cost_low=1, cost_high=10, seed=None):
    
    """Generate a hub-and-spoke logistics-style network.

    A few hub nodes connect many peripheral nodes."""

    rng = random.Random(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    s = 0
    t = n - 1

    hubs = list(range(1, num_hubs + 1))
    spokes = [i for i in range(n) if i not in hubs and i not in [s, t]]

    # Source connects to hubs
    for h in hubs:
        G.add_edge(s, h, dist=rng.randint(cost_low, cost_high))

    # Hubs connect to spokes and spokes connect back to hubs
    for spoke in spokes:
        connected_hubs = rng.sample(hubs, k=min(2, len(hubs)))

        for h in connected_hubs:
            G.add_edge(h, spoke, dist=rng.randint(cost_low, cost_high))
            G.add_edge(spoke, h, dist=rng.randint(cost_low, cost_high))

    # Hubs connect to sink
    for h in hubs:
        G.add_edge(h, t, dist=rng.randint(cost_low, cost_high))

    # Add some random hub-to-hub connections
    for h1 in hubs:
        for h2 in hubs:
            if h1 != h2 and rng.random() < 0.5:
                G.add_edge(h1, h2, dist=rng.randint(cost_low, cost_high))

    density = G.number_of_edges() / G.number_of_nodes()

    return G, s, t, density