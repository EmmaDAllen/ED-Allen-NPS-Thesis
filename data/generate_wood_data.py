

import random
import networkx as nx


def generate_wood_grid(rows, cols, cost_max, delay_max, resource_max, seed):

    rng = random.Random(seed)

    G = nx.DiGraph()
    s = 0
    t = rows * cols + 1

    G.add_node(s)
    G.add_node(t)

    def node_id(r, c):
        """
        r = row index from 0 to rows-1
        c = column index from 0 to cols-1
        """
        return 1 + r * cols + c

    # Add transshipment nodes
    for r in range(rows):
        for c in range(cols):
            G.add_node(node_id(r, c))

   
    # Source -> first column
    # Wood: zero-length and NOT interdictable
    for r in range(rows):

        v = node_id(r, 0)

        G.add_edge(s,v,dist=0,penalty=0,resource=0,interdictable=False,capacity=1)


    # Last column -> sink
    # Wood: zero-length and NOT interdictable
    for r in range(rows):

        u = node_id(r, cols - 1)

        G.add_edge(u,t,dist=0,penalty=0,resource=0,interdictable=False,capacity=1)


    # Grid arcs
    for r in range(rows):

        for c in range(cols):

            u = node_id(r, c)

            candidate_positions = []

            # Vertical neighbors
            if c not in (0, cols - 1):

                if r > 0:
                    candidate_positions.append((r - 1, c))

                if r < rows - 1:
                    candidate_positions.append((r + 1, c))

            # Forward/right arcs
            if c < cols - 1:

                # horizontal
                candidate_positions.append((r, c + 1))

                # upper-right diagonal
                if r > 0:
                    candidate_positions.append((r - 1, c + 1))

                # lower-right diagonal
                if r < rows - 1:
                    candidate_positions.append((r + 1, c + 1))

            for rr, cc in candidate_positions:

                v = node_id(rr, cc)

                cost = rng.randint(1, cost_max)
                penalty = rng.randint(1, delay_max)
                resource = rng.randint(1, resource_max)

                G.add_edge(u,v,dist=cost,penalty=penalty,resource=resource,interdictable=True,capacity=1)


    density = G.number_of_edges() / G.number_of_nodes()

    return G, s, t, density