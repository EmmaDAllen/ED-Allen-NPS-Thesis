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


def generate_grid_network(n,m, cost_low=1, cost_high=10, penalty_low=1,penalty_high=10,
                           capacity_low=1, capacity_high=20, seed=None,):

    """Generate a directed grid-structured network with exactly n nodes and m arcs.

    A separate source and sink are used. Interior nodes are arranged on an
    approximately square grid. A directed grid backbone guarantees that every
    interior node is reachable from the source and that at least one
    source-to-sink path exists.

    Additional arcs are then added until the requested total number of arcs m
    is reached. Candidate arcs are prioritized so that Wood-style local grid
    structure is preserved as much as possible before progressively allowing
    less-local spatial connections.

    Returns:
        G: directed NetworkX graph
        s: source node
        t: sink node
        density: actual arc-to-node ratio m / n"""

    # initialize an independent random number generator so graph generation is
    # reproducible for a given seed
    rng = random.Random(seed)

    # require enough nodes to form a meaningful grid with separate source and sink
    if n < 6:
        raise ValueError("Grid network requires at least 6 nodes.")

    # at least n - 1 arcs are required for a connected directed backbone
    if m < n - 1:
        raise ValueError("Need m >= n - 1.")

    # prevent requests exceeding the number of possible directed arcs without
    # self-loops
    if m > n * (n - 1):
        raise ValueError("Too many arcs.")

    # reserve node 0 for the source and node n - 1 for the sink
    s = 0
    t = n - 1

    # all remaining nodes are placed into the grid
    num_grid_nodes = n - 2

    # choose approximately square grid dimensions based on the number of
    # interior nodes
    rows = max(2, round(math.sqrt(num_grid_nodes)))
    cols = math.ceil(num_grid_nodes / rows)

    # initialize directed graph and explicitly add all n nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    def node_id(r, c):
        """Convert a grid row and column position to a graph node identifier."""

        # convert the two-dimensional grid location to a zero-based index
        idx = r * cols + c

        # the final row may not fill every column when n - 2 is not a perfect
        # multiple of the number of columns
        if idx >= num_grid_nodes:
            return None

        # grid nodes begin at 1 because node 0 is reserved for the source
        return 1 + idx


    # RECORD GRID POSITIONS

    # map each interior graph node to its grid coordinates; these positions are
    # later used to determine which candidate arcs are spatially local
    positions = {}

    for r in range(rows):
        for c in range(cols):

            node = node_id(r, c)

            if node is not None:
                positions[node] = (r, c)



    # CREATE GRID BACKBONE

    # connect the source to the first valid node in every grid row
    # this gives each row a direct entry point from the source
    for r in range(rows):

        first_node = node_id(r, 0)

        if first_node is not None:
            G.add_edge(s, first_node)

    # add left-to-right horizontal arcs within each row
    # this backbone guarantees that every interior node can be reached from s
    for r in range(rows):

        # collect only valid nodes because the final row may be incomplete
        row_nodes = [node_id(r, c) for c in range(cols) if node_id(r, c) is not None]

        # connect consecutive nodes in the row from left to right
        for i in range(len(row_nodes) - 1):
            G.add_edge(row_nodes[i], row_nodes[i + 1])

        # connect the final valid node in each row directly to the sink
        # this guarantees at least one complete source-to-sink route per row
        if row_nodes:
            G.add_edge(row_nodes[-1], t)

    # verify that the required backbone alone does not exceed the requested
    # total number of arcs
    if G.number_of_edges() > m:
        raise ValueError(f"Grid backbone requires {G.number_of_edges()} arcs, but requested m={m}.")



    # CONSTRUCT CANDIDATE ADDITIONAL ARCS

    # additional arcs are ranked according to how closely they preserve local
    # grid structure
    candidates = []

    # list of all interior grid nodes
    interior_nodes = list(positions.keys())

    # consider every possible directed arc between distinct interior nodes that
    # is not already part of the backbone
    for u in interior_nodes:

        ru, cu = positions[u]

        for v in interior_nodes:

            # exclude self-loops and duplicate arcs
            if u == v or G.has_edge(u, v):
                continue

            rv, cv = positions[v]

            # vertical separation between the two grid locations
            row_difference = abs(rv - ru)

            # signed column difference preserves direction:
            # positive values indicate movement toward the sink
            column_difference = cv - cu

            # Manhattan distance provides a simple measure of spatial locality
            manhattan_distance = (abs(rv - ru) + abs(cv - cu))

            # Priority 0:
            # forward, upper-right, and lower-right arcs closely match the local
            # directed structure used in the Wood-style grid networks
            if (column_difference == 1 and row_difference <= 1):
                priority = 0

            # Priority 1:
            # vertical arcs between directly adjacent nodes preserve strong
            # local grid connectivity
            elif (column_difference == 0 and row_difference == 1):
                priority = 1

            # Priority 2:
            # allow other short spatial connections within Manhattan distance 2
            elif manhattan_distance <= 2:
                priority = 2

            # Priority 3:
            # allow longer arcs that still move generally forward toward the sink
            elif column_difference > 0:
                priority = 3

            # Priority 4:
            # remaining arcs include backward or less-local spatial connections
            # and are used only when necessary to reach the requested density
            else:
                priority = 4

            # include a random tie-breaker so networks generated with different
            # seeds are not composed of exactly the same equally ranked arcs
            candidates.append((priority,manhattan_distance, rng.random(), u, v,))

    # sort candidate arcs first by structural priority, then by spatial distance,
    # then randomly among otherwise similar candidates
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))

    # determine how many additional arcs are needed to reach exactly m
    remaining_arcs = m - G.number_of_edges()

    # ensure there are enough valid candidate arcs to satisfy the requested
    # network density
    if remaining_arcs > len(candidates):
        raise ValueError(f"Not enough valid grid arcs to reach m={m}.")

    # add the highest-priority candidate arcs until the graph contains exactly m
    for _, _, _, u, v in candidates[:remaining_arcs]:
        G.add_edge(u, v)



    # ASSIGN ARC ATTRIBUTES

    # assign traversal costs, interdiction penalties, capacities, and
    # interdiction eligibility after the complete graph structure is created
    for u, v in G.edges():

        # arcs leaving the source or entering the sink are treated as connector
        # arcs and remain noninterdictable, consistent with the Wood-style
        # network construction used elsewhere in the evaluation pipeline
        if u == s or v == t:

            # connector arcs do not contribute traversal distance
            G[u][v]["dist"] = 0

            # connector arcs cannot incur additional interdiction delay
            G[u][v]["penalty"] = 0

            # retain a default capacity for compatibility with the common graph
            # representation used across interdiction problem types
            G[u][v]["capacity"] = 1

            # explicitly exclude connector arcs from interdiction
            G[u][v]["interdictable"] = False

        else:

            # assign random traversal cost from the training-data range
            G[u][v]["dist"] = rng.randint(cost_low,cost_high)

            # assign random interdiction penalty from the training-data range
            G[u][v]["penalty"] = rng.randint(penalty_low,penalty_high)

            # assign random arc capacity for compatibility with the common
            # network representation
            G[u][v]["capacity"] = rng.randint(capacity_low,capacity_high)

            # all interior grid arcs are eligible for interdiction
            G[u][v]["interdictable"] = True

    # calculate the realized arc-to-node ratio used as the density feature
    density = G.number_of_edges() / G.number_of_nodes()

    # final consistency check to guarantee that the generated graph honors the
    # requested number of arcs exactly
    if G.number_of_edges() != m:
        raise RuntimeError(f"Expected {m} arcs but generated {G.number_of_edges()}.")

    # return the completed network and metadata used by the training pipeline
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
    source_hubs = rng.sample(hubs, min(2, len(hubs)))

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

            # Primary hub-to-spoke connection.
            # Start with one directed arc so the base topology remains sparse.
            G.add_edge(hub, node)



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
