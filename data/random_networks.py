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


def _assign_arc_attributes(G, rng, cost_low=1,cost_high=10,penalty_low=1,
                           penalty_high=10,capacity_low=1,capacity_high=20,):

    """Assign random distance, penalty, and capacity values to every arc."""

    for u, v in G.edges():
        G[u][v]["dist"] = rng.randint(cost_low, cost_high)
        G[u][v]["penalty"] = rng.randint(penalty_low, penalty_high)
        G[u][v]["capacity"] = rng.randint(capacity_low, capacity_high)


def _add_random_arcs_until_m(G, m, rng):

    """Add random directed arcs until the graph contains exactly m arcs."""

    n = G.number_of_nodes()

    if G.number_of_edges() > m:
        raise ValueError(f"Base topology already has {G.number_of_edges()} arcs, "
            f"which exceeds requested m={m}.")

    possible_arcs = [(u, v) for u in range(n) for v in range(n) if u != v and not G.has_edge(u, v)]

    remaining_arcs = m - G.number_of_edges()

    if remaining_arcs > len(possible_arcs):
        raise ValueError("Not enough available arcs to reach requested m.")

    extra_arcs = rng.sample(possible_arcs, remaining_arcs)
    G.add_edges_from(extra_arcs)


def generate_grid_network(n,m,cost_low=1,cost_high=10,penalty_low=1,penalty_high=10,
                          capacity_low=1,capacity_high=20,seed=None,):

    """Generate a directed grid-like network.

    Nodes are arranged approximately on a rectangular grid. Arcs initially
    connect neighboring nodes toward the right and downward. Random shortcut
    arcs are then added until exactly m arcs are present.

    Returns:
        G: directed NetworkX graph
        s: source node
        t: sink node
        density: m / n"""

    rng = random.Random(seed)

    s = 0
    t = n - 1
    density = m / n

    if m < n - 1:
        raise ValueError("Need m >= n - 1.")

    if m > n * (n - 1):
        raise ValueError("Too many arcs.")

    # choose approximately square grid dimensions
    rows = int(math.sqrt(n))
    cols = math.ceil(n / rows)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # map grid position to node number
    def node_id(r, c):
        idx = r * cols + c
        return idx if idx < n else None

    # connect neighboring nodes to the right and downward
    for r in range(rows):
        for c in range(cols):
            u = node_id(r, c)

            if u is None:
                continue

            # right neighbor
            v = node_id(r, c + 1)
            if v is not None:
                G.add_edge(u, v)

            # downward neighbor
            v = node_id(r + 1, c)
            if v is not None:
                G.add_edge(u, v)

    # Guarantee that all nodes are reachable from source.
    # The row-major grid usually already does this, but this makes it explicit.
    for node in range(1, n):
        if not nx.has_path(G, s, node):
            G.add_edge(node - 1, node)

    _add_random_arcs_until_m(G, m, rng)

    _assign_arc_attributes(G, rng, cost_low, cost_high, penalty_low, penalty_high,
                           capacity_low, capacity_high,)

    return G, s, t, density



def generate_layered_network(n, m, n_layers=None, cost_low=1, cost_high=10, penalty_low=1,
                            penalty_high=10, capacity_low=1, capacity_high=20, seed=None,):

    """Generate a directed layered network.

    The source is placed in the first layer and sink in the last layer.
    Interior nodes are divided across intermediate layers. Each node receives
    at least one incoming arc from the previous layer, creating source-to-sink
    connectivity. Additional random forward arcs are added until m arcs exist.

    Returns:
        G: directed NetworkX graph
        s: source node
        t: sink node
        density: m / n"""

    rng = random.Random(seed)

    s = 0
    t = n - 1
    density = m / n

    if m < n - 1:
        raise ValueError("Need m >= n - 1.")

    if m > n * (n - 1):
        raise ValueError("Too many arcs.")

    if n_layers is None:
        n_layers = max(3, round(math.sqrt(n)))

    if n_layers > n:
        raise ValueError("n_layers cannot exceed number of nodes.")

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # source and sink occupy their own layers
    interior_nodes = list(range(1, n - 1))
    num_middle_layers = n_layers - 2

    layers = [[s]]

    # divide interior nodes approximately evenly across layers
    for i in range(num_middle_layers):
        layer = interior_nodes[i::num_middle_layers]
        layers.append(layer)

    layers.append([t])

    # Guarantee every node in each layer has a predecessor from previous layer
    for layer_idx in range(1, len(layers)):
        prev_layer = layers[layer_idx - 1]
        current_layer = layers[layer_idx]

        for v in current_layer:
            u = rng.choice(prev_layer)
            G.add_edge(u, v)

    # Add extra forward arcs only.
    # This preserves the layered/DAG-like structure.
    possible_arcs = []

    for i in range(len(layers) - 1):
        for j in range(i + 1, len(layers)):
            for u in layers[i]:
                for v in layers[j]:
                    if u != v and not G.has_edge(u, v):
                        possible_arcs.append((u, v))

    remaining_arcs = m - G.number_of_edges()

    if remaining_arcs > len(possible_arcs):
        raise ValueError(
            f"Cannot create {m} forward arcs with {n_layers} layers. "
            "Reduce m or use fewer layers."
        )

    extra_arcs = rng.sample(possible_arcs, remaining_arcs)
    G.add_edges_from(extra_arcs)

    _assign_arc_attributes(G,rng,cost_low,cost_high,penalty_low,penalty_high,
                           capacity_low,capacity_high,)

    return G, s, t, density


def generate_geometric_network(n,m,cost_low=1,cost_high=10,penalty_low=1,penalty_high=10,
                               capacity_low=1,capacity_high=20,seed=None,):

    """Generate a directed geometric network.

    Nodes are assigned random coordinates in the unit square. Nearby nodes
    are preferentially connected, producing local/spatial structure. A
    source-rooted spanning structure is created first, then additional
    short-distance arcs are added until exactly m arcs are present.

    Returns:
        G: directed NetworkX graph
        s: source node
        t: sink node
        density: m / n"""

    rng = random.Random(seed)

    s = 0
    t = n - 1
    density = m / n

    if m < n - 1:
        raise ValueError("Need m >= n - 1.")

    if m > n * (n - 1):
        raise ValueError("Too many arcs.")

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # assign random 2D positions
    pos = {node: (rng.random(), rng.random()) for node in range(n)}

    nx.set_node_attributes(G, pos, "pos")

    def euclidean_distance(u, v):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Build a source-rooted tree.
    # Each new node connects from the closest already-connected node.
    connected = {s}
    unconnected = set(range(n)) - {s}

    while unconnected:
        best_pair = None
        best_distance = float("inf")

        for u in connected:
            for v in unconnected:
                d = euclidean_distance(u, v)

                if d < best_distance:
                    best_distance = d
                    best_pair = (u, v)

        u, v = best_pair
        G.add_edge(u, v)

        connected.add(v)
        unconnected.remove(v)

    # Candidate arcs ranked primarily by geometric proximity
    possible_arcs = [(u, v) for u in range(n) for v in range(n) if u != v and not G.has_edge(u, v)]

    # Add a little randomness so every graph is not deterministically
    # composed of exactly the shortest remaining arcs.
    possible_arcs.sort(
        key=lambda edge: (euclidean_distance(edge[0], edge[1]) + rng.uniform(0, 0.1)))

    remaining_arcs = m - G.number_of_edges()

    extra_arcs = possible_arcs[:remaining_arcs]
    G.add_edges_from(extra_arcs)

    _assign_arc_attributes(G,rng,cost_low,cost_high,penalty_low,penalty_high,
                           capacity_low,capacity_high,)

    return G, s, t, density
