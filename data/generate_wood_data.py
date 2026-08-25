
"""generate_wood_data.py

Generate directed grid networks based on the shortest-path interdiction test problems
described by Wood.

The generated networks contain a rectangular grid of transshipment nodes with a single source 
and sink. The source is connected to every node in the first column, and every node in the 
final column is connected to the sink using zero-length, non-interdictable arcs. Directed grid
arcs connect neighboring transshipment nodes according to the Wood network structure.

For each interdictable grid arc, the script randomly generates:
1. A base traversal cost from 1 to the specified maximum cost.
2. An interdiction delay from 1 to the specified maximum delay.
3. An interdiction resource requirement from 1 to the specified maximum resource value.

The source and sink connection arcs have zero traversal cost, zero interdiction delay, and zero
 resource requirement and are excluded from interdiction.

A fixed random seed is used so each benchmark network can be reproduced exactly. The function 
returns the generated NetworkX directed graph, source node, sink node, and arc-to-node ratio for 
use in shortest-path interdiction evaluation."""

import random
import networkx as nx


def generate_wood_grid(rows, cols, cost_max, delay_max, resource_max, seed):

    """Generate one directed Wood-style shortest-path interdiction grid network.

    Parameters
    rows : int = Number of rows of transshipment nodes in the grid.

    cols : int = Number of columns of transshipment nodes in the grid.

    cost_max : int = Maximum initial traversal cost assigned to an interdictable grid arc.

    delay_max : int = Maximum interdiction delay assigned to an interdictable grid arc.

    resource_max : int
        Maximum interdiction resource requirement assigned to an interdictable grid arc.

    seed : int = Random seed used to generate reproducible arc attributes.

    Returns
    G : networkx.DiGraph = Generated directed Wood-style grid network.

    s : int = Source node identifier.

    t : int = Sink node identifier.

    density : float = Arc-to-node ratio m/n of the generated network."""


    # RANDOM NUMBER GENERATOR

    # use an independent seeded random-number generator so that all randomly
    # generated arc attributes are reproducible for a given benchmark instance
    rng = random.Random(seed)

    # INITIALIZE DIRECTED GRAPH
    G = nx.DiGraph()

    # reserve node 0 for the source   
    s = 0

    # transshipment nodes are numbered from 1 through rows * cols, so assign
    # the sink the next available node identifier
    t = rows * cols + 1

    # explicitly add the source and sink to the graph
    G.add_node(s)
    G.add_node(t)


    def node_id(r, c):

        """Convert a grid row and column position to its graph node identifier.

        r = row index from 0 to rows - 1
        c = column index from 0 to cols - 1

        Transshipment node numbering begins at 1 because node 0 is reserved for the source."""

        return 1 + r * cols + c



    # ADD TRANSSHIPMENT NODES

    # create one graph node for every position in the rows-by-columns grid
    for r in range(rows):
        for c in range(cols):
            G.add_node(node_id(r, c))


   
    # SOURCE CONNECTION ARCS

    # connect the source to every transshipment node in the first grid column
    # these arcs have zero length and cannot be interdicted in the Wood
    # benchmark construction
    for r in range(rows):

        # identify the first-column node in the current row
        v = node_id(r, 0)

        # source connection arcs contribute no distance to the path
        # interdiction produces no delay because these arcs are excluded
        # no interdiction resources are associated with excluded arcs
        # prevent the model/optimization formulation from selecting this arc
        # capacity is retained for compatibility with the shared graph format
        G.add_edge(s,v,dist=0,penalty=0,resource=0,interdictable=False,capacity=1)



    # SINK CONNECTION ARCS

    # connect every transshipment node in the final grid column to the sink
    # these arcs also have zero length and cannot be interdicted
    for r in range(rows):

        # identify the final-column node in the current row
        u = node_id(r, cols - 1)

        # source connection arcs contribute no distance to the path
        # interdiction produces no delay because these arcs are excluded
        # no interdiction resources are associated with excluded arcs
        # prevent the model/optimization formulation from selecting this arc
        # capacity is retained for compatibility with the shared graph format
        G.add_edge(u,t,dist=0,penalty=0,resource=0,interdictable=False,capacity=1)



    # GENERATE DIRECTED GRID ARCS

    # iterate through every transshipment node and determine which neighboring
    # grid positions should receive outgoing directed arcs
    for r in range(rows):

        for c in range(cols):

            # retrieve the graph identifier for the current grid position
            u = node_id(r, c)

            # store valid neighboring positions that can be reached from u
            candidate_positions = []


            # VERTICAL ARCS

            # vertical movement is included only for interior columns; the first
            # and final columns connect directly to the source and sink,
            # respectively, according to the Wood grid construction
            if c not in (0, cols - 1):

                # add the node immediately above when the current node is not
                # already in the first row
                if r > 0:
                    candidate_positions.append((r - 1, c))

                # add the node immediately below when the current node is not
                # already in the final row
                if r < rows - 1:
                    candidate_positions.append((r + 1, c))



            # FORWARD ARCS

            # nodes outside the final column may connect to nodes in the next
            # column, allowing paths to progress toward the sink
            if c < cols - 1:

                # horizontal arc to the node directly to the right
                candidate_positions.append((r, c + 1))

                # upper-right diagonal arc when an upper row exists
                if r > 0:
                    candidate_positions.append((r - 1, c + 1))

                # lower-right diagonal arc when a lower row exists
                if r < rows - 1:
                    candidate_positions.append((r + 1, c + 1))



            # ADD GRID ARCS AND ATTRIBUTES

            # create one directed arc from the current node to every valid
            # neighboring grid position identified above
            for rr, cc in candidate_positions:

                # convert the neighboring grid position to its graph node ID
                v = node_id(rr, cc)

                # generate the initial traversal cost uniformly from the
                # benchmark-specific cost range
                cost = rng.randint(1, cost_max)

                # generate the additional traversal delay imposed if the arc is interdicted                
                penalty = rng.randint(1, delay_max)

                # generate the amount of interdiction resource required to interdict this arc                
                resource = rng.randint(1, resource_max)

                # add the directed grid arc and its benchmark attributes
                # initial arc traversal cost
                # additional traversal delay produced by interdiction
                # resource expenditure required to interdict the arc
                # all generated grid arcs are eligible for interdiction
                # capacity is retained for compatibility with the common network representation 
                # used throughout the project
                G.add_edge(u,v,dist=cost,penalty=penalty,resource=resource,interdictable=True,capacity=1)



    # NETWORK DENSITY

    # calculate the same arc-to-node ratio m/n used as the density feature
    # throughout model training and evaluation
    density = G.number_of_edges() / G.number_of_nodes()


    # return the completed graph and information required by the evaluation pipeline
    return G, s, t, density