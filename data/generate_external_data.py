import pandas as pd
import networkx as nx
import numpy as np

# Must match the reference maximum used in interdiction_data.py
COST_HIGH = 10
PENALTY_LOW = 1
PENALTY_HIGH = 10



def load_external_network(node_path,arc_path,source,sink,penalty_low=1,penalty_high=10,penalty_seed=5,):

    """Load an external directed network from node and arc CSV files.

    External arc costs are uniformly rescaled so that the maximum
    arc cost equals COST_HIGH=10, matching the scale used during
    shortest-path model training.

    Interdiction penalties are generated uniformly from [1, 10]
    because the external data do not contain interdiction penalties."""

    # Load CSV files
    node_df = pd.read_csv(node_path)
    arc_df = pd.read_csv(arc_path)


    original_cost_max = arc_df["cost"].max()

    if original_cost_max <= 0:
        raise ValueError("External network must contain positive arc costs.")

    # Multiplicative scaling preserves relative path costs
    cost_scale_factor = COST_HIGH / original_cost_max

    arc_df["scaled_cost"] = (arc_df["cost"] * cost_scale_factor)

    print(
        f"Original cost range: "
        f"{arc_df['cost'].min():.4f} - "
        f"{arc_df['cost'].max():.4f}")

    print(
        f"Scaled cost range: "
        f"{arc_df['scaled_cost'].min():.4f} - "
        f"{arc_df['scaled_cost'].max():.4f}")

    print(f"Cost scale factor: {cost_scale_factor:.6f}")

    # Initialize directed graph
    G = nx.DiGraph()

 
    # ADD NODES

    node_ids = node_df["node"].astype(int).tolist()

    node_map = {original_id: new_id for new_id, original_id in enumerate(node_ids)}

    for _, row in node_df.iterrows():

        original_node = int(row["node"])
        node = node_map[original_node]

        G.add_node(int(row["node"]), lat=row["lat"], lon=row["lon"], supply=row["supply"])


    # GENERATE INTERDICTION PENALTIES

    rng = np.random.default_rng(penalty_seed)

    for _, row in arc_df.iterrows():

        original_u = int(row["from_node"])
        original_v = int(row["to_node"])

        u = int(row["from_node"])
        v = int(row["to_node"])

        # Generate penalty on the same scale used during training
        penalty = int(rng.integers(penalty_low,penalty_high + 1))

        # ADD DIRECTED ARCS

        G.add_edge(u, v,

            # Rescaled external transportation cost
            dist=float(row["scaled_cost"]),

            # Synthetic interdiction penalty
            penalty=penalty,

            # Preserve external network attributes
            capacity=float(row["capacity"]),
            transport_mode=row["transport_mode"],

            # Assume all external arcs are eligible unless
            # there is a reason to exclude particular arcs
            interdictable=True,

            # Preserve original cost for reference
            original_cost=float(row["cost"]))
        


    # VALIDATION



    if source not in G:
        raise ValueError(f"Source node {source} does not exist.")

    if sink not in G:
        raise ValueError(f"Sink node {sink} does not exist.")

    if not nx.has_path(G, source, sink):
        raise ValueError(f"No directed path exists from " f"{source} to {sink}.")

    density = (G.number_of_edges() / G.number_of_nodes())
    source_internal = node_map[source]
    sink_internal = node_map[sink]

    print(
        f"Loaded external network | "
        f"nodes={G.number_of_nodes()} | "
        f"arcs={G.number_of_edges()} | "
        f"density={density:.4f} | "
        f"source={source_internal} | "
        f"sink={sink_internal}")

    return G, source_internal, sink_internal, density