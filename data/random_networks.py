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


def generate_grid_network(n, m=None, cost_low=1, cost_high=10, penalty_low=1, penalty_high=10,
        capacity_low=1, capacity_high=20, seed=None):

    """Generate a directed Wood-style grid network.

    The network contains a separate source and sink. The source connects to all nodes in the first 
    grid column, and all nodes in the final grid column connect to the sink.

    Interior grid arcs allow vertical movement as well as forward, upper-right, and lower-right movement.

    The resulting number of arcs is determined primarily by the grid topology rather than by arbitrary
    random arc insertion."""

    rng = random.Random(seed)

    if n < 6:
        raise ValueError("Grid network requires at least 6 nodes.")

    # reserve source and sink
    s = 0
    t = n - 1

    num_grid_nodes = n - 2

    # choose approximately square grid dimensions
    rows = max(2, round(math.sqrt(num_grid_nodes)))
    cols = math.ceil(num_grid_nodes / rows)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # assign graph node IDs 1,...,n-2 to grid positions
    def node_id(r, c):
        idx = r * cols + c

        if idx >= num_grid_nodes:
            return None

        return 1 + idx

    # source -> first column
    for r in range(rows):
        v = node_id(r, 0)

        if v is not None:
            G.add_edge(s, v, dist=0, penalty=0, capacity=1, interdictable=False)

    # grid arcs
    for r in range(rows):
        for c in range(cols):

            u = node_id(r, c)

            if u is None:
                continue

            candidate_positions = []

            # vertical arcs in interior columns
            if c not in (0, cols - 1):

                if r > 0:
                    candidate_positions.append((r - 1, c))

                if r < rows - 1:
                    candidate_positions.append((r + 1, c))

            # forward / diagonal-forward arcs
            if c < cols - 1:

                candidate_positions.append((r, c + 1))

                if r > 0:
                    candidate_positions.append((r - 1, c + 1))

                if r < rows - 1:
                    candidate_positions.append((r + 1, c + 1))

            for rr, cc in candidate_positions:

                v = node_id(rr, cc)

                if v is None:
                    continue

                G.add_edge(u, v, dist=rng.randint(cost_low, cost_high),
                    penalty=rng.randint(penalty_low, penalty_high),
                    capacity=rng.randint(capacity_low, capacity_high), interdictable=True)

    # last valid node in each row -> sink
    for r in range(rows):

        row_nodes = [node_id(r, c) for c in range(cols) if node_id(r, c) is not None]

        if row_nodes:

            u = row_nodes[-1]

            G.add_edge(u, t, dist=0, penalty=0, capacity=1, interdictable=False)

    density = G.number_of_edges() / G.number_of_nodes()

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



def generate_star_mesh_network(n, m, cost_low=1, cost_high=10, penalty_low=1, penalty_high=10,
        capacity_low=1, capacity_high=20, seed=None):

    """Generate a directed star-mesh / hub-and-spoke network.

    A small number of hub nodes are created, peripheral nodes are assigned to hubs, hubs are
    interconnected, and additional arcs are added while preserving the general hub-and-spoke structure.

    This topology is intended to represent transportation and logistics networks with regional hubs,
    local connections, and limited cross-links.

    Returns:
        G: directed NetworkX graph
        s: source node
        t: sink node
        density: actual arc-to-node ratio m / n"""

    rng = random.Random(seed)

    if n < 6:
        raise ValueError("Star-mesh network requires at least 6 nodes.")

    if m < n - 1:
        raise ValueError("Need m >= n - 1.")

    if m > n * (n - 1):
        raise ValueError("Too many arcs.")

    s = 0
    t = n - 1

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Number of hubs grows slowly with network size.
    # This gives roughly 2-5 hubs for the network sizes in your training set.
    n_hubs = max(2, min(5, round(math.sqrt(n) / 2)))

    # Exclude source and sink from hub selection.
    candidate_nodes = list(range(1, n - 1))

    hubs = rng.sample(candidate_nodes, n_hubs)

    peripheral_nodes = [node for node in candidate_nodes if node not in hubs]

    # Randomly assign each peripheral node to one primary hub.
    hub_members = {hub: [] for hub in hubs}

    for node in peripheral_nodes:
        hub = rng.choice(hubs)
        hub_members[hub].append(node)



    # SOURCE CONNECTION

    # Connect the source to one or more hubs.
    # Using multiple entry hubs avoids making a single source arc an unavoidable bottleneck
    source_hubs = rng.sample(
        hubs,
        min(2, len(hubs))
    )

    for hub in source_hubs:
        G.add_edge(s, hub)




    # HUB-TO-HUB MESH

    # Ensure the hub network is connected by first creating a directed backbone
    shuffled_hubs = hubs.copy()
    rng.shuffle(shuffled_hubs)

    for i in range(len(shuffled_hubs) - 1):
        u = shuffled_hubs[i]
        v = shuffled_hubs[i + 1]

        G.add_edge(u, v)
        G.add_edge(v, u)

    # Add additional hub-to-hub arcs.
    for u in hubs:
        for v in hubs:

            if u != v and not G.has_edge(u, v):

                # Moderate probability of an additional hub connection.
                if rng.random() < 0.5:
                    G.add_edge(u, v)



    # HUB-AND-SPOKE CONNECTIONS

    for hub, members in hub_members.items():

        for node in members:

            # Primary hub -> peripheral connection
            G.add_edge(hub, node)

            # Allow return movement as well.
            G.add_edge(node, hub)



    # SINK CONNECTION

    sink_hubs = rng.sample(hubs, min(2, len(hubs)))

    for hub in sink_hubs:
        G.add_edge(hub, t)



    # GUARANTEE SOURCE-TO-SINK CONNECTIVITY

    if not nx.has_path(G, s, t):

        source_hub = source_hubs[0]
        sink_hub = sink_hubs[0]

        if source_hub != sink_hub:
            G.add_edge(source_hub, sink_hub)



    # ADD STRUCTURED CROSS-LINKS UNTIL m ARCS

    if G.number_of_edges() > m:
        raise ValueError(f"Base star-mesh topology already has "
            f"{G.number_of_edges()} arcs, which exceeds requested m={m}.")

    possible_arcs = []

    # Prefer transportation-like links:
    # 1. peripheral -> nearby/alternative hub
    # 2. peripheral -> peripheral
    # 3. hub -> peripheral

    for u in range(1, n - 1):
        for v in range(1, n - 1):

            if u == v or G.has_edge(u, v):
                continue

            u_is_hub = u in hubs
            v_is_hub = v in hubs

            # Assign priority scores so hub-related arcs are preferred.
            if u_is_hub and v_is_hub:
                priority = 0

            elif u_is_hub or v_is_hub:
                priority = 1

            else:
                priority = 2

            possible_arcs.append((priority, rng.random(), u, v))

    # Prefer hub-related arcs while randomizing within each category.
    possible_arcs.sort(key=lambda x: (x[0], x[1]))

    remaining_arcs = m - G.number_of_edges()

    if remaining_arcs > len(possible_arcs):
        raise ValueError(f"Not enough valid star-mesh arcs to reach m={m}.")

    for _, _, u, v in possible_arcs[:remaining_arcs]:
        G.add_edge(u, v)



    # ARC ATTRIBUTES

    _assign_arc_attributes(G,rng, cost_low, cost_high, penalty_low, penalty_high, 
                           capacity_low, capacity_high)

    density = G.number_of_edges() / G.number_of_nodes()

    return G, s, t, density
