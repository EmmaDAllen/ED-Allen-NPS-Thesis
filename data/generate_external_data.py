"""generate_external_data.py

Load and prepare external network data for shortest-path interdiction evaluation.

The script converts node and directed-arc data from external CSV files into the
NetworkX graph representation used by the network interdiction models.

Usage
Run from the repository root with:

    PYTHONPATH=. python data/generate_external_data.py NODE_PATH ARC_PATH SOURCE SINK

The script:
1. Loads node and directed-arc data from external CSV files.
2. Validates required node and arc attributes.
3. Relabels external node identifiers to consecutive integer IDs required by the model.
4. Preserves the original external transportation costs.
5. Loads deterministic detour-based interdiction penalties from the arc data.
6. Constructs a directed NetworkX graph while preserving relevant external network attributes.
7. Converts the user-specified source and sink from original external IDs to internal graph IDs.
8. Verifies that the requested source and sink exist and that a directed path connects them.
9. Computes the graph's arc-to-node ratio for use as the model's density feature.
10. Returns the prepared graph, internal source and sink IDs, and network density for evaluation."""

import pandas as pd
import networkx as nx
import argparse
import os
import pickle




def load_external_network(node_path,arc_path,source,sink):

    """Load an external directed network from node and arc CSV files.

    External node identifiers are remapped to consecutive integer IDs because the
    model expects graph nodes indexed from 0 through n-1. Original node identifiers
    and relevant node metadata are retained as attributes.

    External transportation costs are preserved at their original values.
    Interdiction penalties are read directly from the arc data and represent the
    deterministic detour-based penalties constructed during external-data
    preprocessing.

    The requested source and sink are supplied using their original external node
    identifiers and converted to the corresponding internal graph identifiers."""


    # LOAD EXTERNAL DATA

    # read node and directed-arc information from the supplied CSV files
    node_df = pd.read_csv(node_path)
    arc_df = pd.read_csv(arc_path)


    # validate required columns
    required_node_columns = {"node","lat","lon","supply"}

    required_arc_columns = {"from_node","to_node","transport_mode","cost","capacity","penalty"}

    missing_node_columns = required_node_columns - set(node_df.columns)
    missing_arc_columns = required_arc_columns - set(arc_df.columns)

    if missing_node_columns:
        raise ValueError(f"Missing node columns: {sorted(missing_node_columns)}")

    if missing_arc_columns:
        raise ValueError(f"Missing arc columns: {sorted(missing_arc_columns)}")

    if arc_df["cost"].isna().any():
        raise ValueError("External arc file contains missing costs.")

    if arc_df["penalty"].isna().any():
        raise ValueError("External arc file contains missing penalties.")

    if (arc_df["cost"] < 0).any():
        raise ValueError("External arc costs must be positive.")

    if (arc_df["penalty"] <= 0).any():
        raise ValueError("External interdiction penalties must be positive.")


    # Check for duplicate directed node pairs because NetworkX DiGraph
    # can store only one arc from u to v.
    duplicate_endpoints = arc_df.duplicated(subset=["from_node", "to_node"]).sum()

    if duplicate_endpoints > 0:
        raise ValueError(f"External arc file contains {duplicate_endpoints} duplicate "
            f"directed node pairs. A DiGraph would overwrite these arcs.")


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





    # ADD DIRECTED ARCS
    for _, row in arc_df.iterrows():

        # retrieve the original external endpoints
        original_u = int(row["from_node"])
        original_v = int(row["to_node"])

        # convert external endpoint identifiers to consecutive internal IDs
        u = node_map[original_u]
        v = node_map[original_v]


        # add the directed arc using attributes expected by the model and
        # preserve relevant external attributes for later analysis
        G.add_edge(u, v,

            # rescaled external transportation cost used as the shortest-path distance feature
            dist=float(row["cost"]),

            # deterministic detour-based interdiction penalty
            penalty=float(row["penalty"]),


            # preserve available external network attributes
            capacity=float(row["capacity"]),
            transport_mode=row["transport_mode"],

            # external arcs are assumed eligible for interdiction unless the
            # dataset provides a substantive reason to exclude individual arcs
            interdictable=True)
        


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

    # parse supplied command-line arguments
    args = parser.parse_args()


    # LOAD EXTERNAL NETWORK
    G, source_internal, sink_internal, density = load_external_network(node_path=args.node_path,
                                                    arc_path=args.arc_path,source=args.source,sink=args.sink)

