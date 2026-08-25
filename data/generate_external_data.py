
"""generate_external_data.py

Load and prepare external network data for shortest-path interdiction evaluation.

The script converts node and directed-arc data from external CSV files into the
NetworkX graph representation used by the network interdiction models.

Usage
Run from the repository root with:

    PYTHONPATH=. python data/generate_external_data.py NODE_PATH ARC_PATH SOURCE SINK

The script:
1. Loads node and directed-arc data from external CSV files.
2. Relabels external node identifiers to consecutive integer IDs required by the model.
3. Rescales external arc costs to match the cost range used during model training.
4. Generates reproducible synthetic interdiction penalties using the training penalty range.
5. Constructs a directed NetworkX graph while preserving relevant external network attributes.
6. Converts the user-specified source and sink from original external IDs to internal graph IDs.
7. Verifies that the requested source and sink exist and that a directed path connects them.
8. Computes the graph's arc-to-node ratio for use as the model's density feature.
9. Returns the prepared graph, internal source and sink IDs, and network density for evaluation."""

import pandas as pd
import networkx as nx
import numpy as np
import argparse

# EXTERNAL DATA SCALING CONSTANTS

# Maximum edge-cost value used during shortest-path model training.
# External network costs are rescaled relative to this value so their
# magnitudes are comparable to the synthetic training data.
COST_HIGH = 10

# Bounds used to generate synthetic interdiction penalties for external arcs.
# External transportation datasets do not contain interdiction penalties, so
# penalties are sampled using the same range used during model training.
PENALTY_LOW = 1
PENALTY_HIGH = 10




def load_external_network(node_path,arc_path,source,sink,penalty_low=1,penalty_high=10,penalty_seed=5,):

    """Load an external directed network from node and arc CSV files.

    External node identifiers are remapped to consecutive integer IDs because the model expects 
    graph nodes indexed from 0 through n-1. The original node identifiers are retained as node 
    attributes for reference. External arc costs are uniformly rescaled so that the maximum arc 
    cost equals COST_HIGH=10, matching the scale used during shortest-path model training. 
    Multiplicative scaling preserves the relative differences among arc costs and therefore does 
    not change which paths are shortest. Because the external data do not contain interdiction penalties, 
    synthetic penalties are generated uniformly from the same [1, 10] range used during training.

    Parameters
    node_path : str
        Path to the CSV file containing external node information.

    arc_path : str
        Path to the CSV file containing external directed arc information.

    source : int
        Original external node identifier to use as the source.

    sink : int
        Original external node identifier to use as the sink.

    penalty_low : int
        Minimum synthetic interdiction penalty.

    penalty_high : int
        Maximum synthetic interdiction penalty.

    penalty_seed : int
        Random seed used to generate reproducible interdiction penalties.

    Returns
    G : networkx.DiGraph
        Relabeled directed graph containing model-compatible edge attributes.

    source_internal : int
        Internal consecutive node ID corresponding to the requested source.

    sink_internal : int
        Internal consecutive node ID corresponding to the requested sink.

    density : float
        Arc-to-node ratio m/n for the loaded network."""


    # LOAD EXTERNAL DATA

    # read node and directed-arc information from the supplied CSV files
    node_df = pd.read_csv(node_path)
    arc_df = pd.read_csv(arc_path)


    # RESCALE EXTERNAL ARC COSTS

    # identify the largest cost in the original external network
    original_cost_max = arc_df["cost"].max()

    # the scaling operation requires at least one positive arc cost
    if original_cost_max <= 0:
        raise ValueError("External network must contain positive arc costs.")

    # scale every external cost by the same multiplicative factor so that
    # the largest cost equals COST_HIGH while preserving relative costs
    cost_scale_factor = COST_HIGH / original_cost_max

    arc_df["scaled_cost"] = (arc_df["cost"] * cost_scale_factor)

    print(f"Original cost range: "
        f"{arc_df['cost'].min():.4f} - "
        f"{arc_df['cost'].max():.4f}")

    print( f"Scaled cost range: "
        f"{arc_df['scaled_cost'].min():.4f} - "
        f"{arc_df['scaled_cost'].max():.4f}")

    print(f"Cost scale factor: {cost_scale_factor:.6f}")



    # INITIALIZE DIRECTED GRAPH

    G = nx.DiGraph()


 
    # RELABEL AND ADD NODES

    # retrieve the original external node identifiers
    node_ids = node_df["node"].astype(int).tolist()

    # map arbitrary external node identifiers to consecutive integer IDs required by the model's 
    # graph representation
    node_map = {original_id: new_id for new_id, original_id in enumerate(node_ids)}

    # add each relabeled node to the NetworkX graph while retaining its
    # original identifier and external metadata as node attributes
    for _, row in node_df.iterrows():

        original_node = int(row["node"])
        # convert the original external identifier to the model-compatible ID
        node = node_map[original_node]

        G.add_node(node,original_node=original_node,lat=row["lat"],lon=row["lon"],supply=row["supply"])


    # GENERATE SYNTHETIC INTERDICTION PENALTIES

    # initialize a deterministic random-number generator so the external
    # evaluation instance receives the same penalties each time it is loaded
    rng = np.random.default_rng(penalty_seed)

    # ADD DIRECTED ARCS
    for _, row in arc_df.iterrows():

        # retrieve the original external endpoints
        original_u = int(row["from_node"])
        original_v = int(row["to_node"])

        # convert external endpoint identifiers to consecutive internal IDs
        u = node_map[original_u]
        v = node_map[original_v]

        # external data do not contain interdiction penalties, so generate
        # one using the same integer range represented in the training data
        penalty = int(rng.integers(penalty_low,penalty_high + 1))



        # add the directed arc using attributes expected by the model and
        # preserve relevant external attributes for later analysis
        G.add_edge(u, v,

            # rescaled external transportation cost used as the shortest-path distance feature
            dist=float(row["scaled_cost"]),

            # synthetic interdiction penalty on the training-data scale
            penalty=penalty,

            # preserve available external network attributes
            capacity=float(row["capacity"]),
            transport_mode=row["transport_mode"],

            # external arcs are assumed eligible for interdiction unless the
            # dataset provides a substantive reason to exclude individual arcs
            interdictable=True,

            # preserve the unscaled transportation cost for reference
            original_cost=float(row["cost"]))
        


    # VALIDATE REQUESTED SOURCE AND SINK

    # source and sink are supplied using the ORIGINAL external node IDs.
    # Validate them against node_map before translating them into the
    # consecutive internal node IDs used by G.
    if source not in node_map:
        raise ValueError(f"Source node {source} does not exist in the external node file.")

    if sink not in node_map:
        raise ValueError(f"Sink node {sink} does not exist in the external node file.")

    # convert the user-specified external source and sink IDs to the
    # corresponding internal graph identifiers
    source_internal = node_map[source]
    sink_internal = node_map[sink]

    # verify that the directed network contains at least one feasible path
    # from the requested source to the requested sink
    if not nx.has_path(G, source_internal, sink_internal):
        raise ValueError(f"No directed path exists from " f"{source} to {sink}.")


    # COMPUTE NETWORK DENSITY

    # use the same arc-to-node ratio m/n used throughout training and evaluation
    density = (G.number_of_edges() / G.number_of_nodes())

    # report the loaded graph dimensions and the relabeled source/sink IDs
    print(
        f"Loaded external network | "
        f"nodes={G.number_of_nodes()} | "
        f"arcs={G.number_of_edges()} | "
        f"density={density:.4f} | "
        f"source={source_internal} | "
        f"sink={sink_internal}")

    # return the model-compatible NetworkX graph and its internal source/sink IDs
    return G, source_internal, sink_internal, density


if __name__ == "__main__":

    # COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser(
        description=("Load and prepare an external network for network-interdiction evaluation."))

    # path to the CSV containing node information
    parser.add_argument("node_path", help="Path to the external node CSV file.")

    # path to the CSV containing directed arc information
    parser.add_argument("arc_path",help="Path to the external arc CSV file.")

    # original external node identifier to use as the source
    parser.add_argument("source",type=int,help="Original node ID to use as the source.")

    # original external node identifier to use as the sink
    parser.add_argument("sink",type=int,help="Original node ID to use as the sink.")

    # optional seed controlling the synthetic interdiction penalties
    parser.add_argument("--penalty_seed",type=int,default=5,
        help=("Random seed for synthetic interdiction penalties (default: 5)."))

    # parse supplied command-line arguments
    args = parser.parse_args()


    # LOAD EXTERNAL NETWORK
    G, source_internal, sink_internal, density = load_external_network(node_path=args.node_path,arc_path=args.arc_path,
                                                     source=args.source,sink=args.sink,penalty_low=PENALTY_LOW,
                                                     penalty_high=PENALTY_HIGH,penalty_seed=args.penalty_seed,)