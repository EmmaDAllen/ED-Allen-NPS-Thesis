"""Real-world TNTP transportation network utilities for shortest-path
network interdiction experiments.

Workflow
1. Raw TNTP networks are loaded once.
2. Free-flow travel time is used as the base edge distance.
3. A detour-based interdiction penalty is calculated once for each edge.
4. The processed NetworkX graph is saved as a pickle file.
5. Training/evaluation instances subsequently load the processed graph
   and sample different source-sink pairs.

The resulting graphs use the same primary edge attributes as the
synthetic shortest-path networks:

    dist
    penalty
    capacity
    interdictable
"""

import os
import pickle
import random

import networkx as nx
import numpy as np
import pandas as pd



# REAL-WORLD NETWORK DEFINITIONS

# Networks used during training.

REAL_WORLD_TRAINING_NETWORKS = {

    "sioux_falls": {
        "raw":"data/real_world_data/SiouxFalls/SiouxFalls_net.tntp",
        "processed":"data/real_world_data/processed/sioux_falls.pkl"},

    "anaheim": {
        "raw":"data/real_world_data/Anaheim/Anaheim_net.tntp",
        "processed":"data/real_world_data/processed/anaheim.pkl"},

    "chicago_sketch": {
        "raw":"data/real_world_data/Chicago-Sketch/ChicagoSketch_net.tntp",
        "processed":"data/real_world_data/processed/chicago_sketch.pkl"},

    "barcelona": {
        "raw":"data/real_world_data/Barcelona/Barcelona_net.tntp",
        "processed":"data/real_world_data/processed/barcelona.pkl"},

    "winnipeg": {
        "raw":"data/real_world_data/Winnipeg/Winnipeg_net.tntp",
        "processed":"data/real_world_data/processed/winnipeg.pkl"},

    "gold_coast": {
        "raw": "data/real_world_data/GoldCoast/Goldcoast_network_2016_01.tntp",
        "processed":"data/real_world_data/processed/gold_coast.pkl"}}


# Networks reserved entirely for held-out evaluation.

REAL_WORLD_EVALUATION_NETWORKS = {

    "berlin_mpf": {
        "raw":"data/real_world_data/"
            "Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center/"
            "Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center_net.tntp",
        "processed":
            "data/real_world_data/processed/berlin_mpf.pkl"},

"eastern_massachusetts": {
    "raw": "data/real_world_data/Eastern-Massachusetts/EMA_net.tntp",
    "processed": "data/real_world_data/processed/eastern_massachusetts.pkl"},
}



# LOAD RAW TNTP NETWORK

def load_tntp_network(filepath):

    """Load a directed transportation network from a TNTP network file.

    Free-flow travel time is used as the base traversal distance.

    Original TNTP attributes are retained on each edge for later
    analysis.

    Parameters
    filepath : str
        Path to the raw *_net.tntp file.

    Returns
    G : networkx.DiGraph
        Directed transportation network.

    metadata : dict
        Metadata read from the TNTP file."""


    # READ FILE

    with open(filepath, "r") as f:
        lines = f.readlines()



    # READ TNTP METADATA

    metadata = {}

    for line in lines:

        stripped = line.strip()

        if stripped.startswith("<NUMBER OF ZONES>"):

            metadata["number_of_zones"] = int(stripped.split(">")[1].strip())

        elif stripped.startswith("<NUMBER OF NODES>"):

            metadata["number_of_nodes"] = int(stripped.split(">")[1].strip())

        elif stripped.startswith("<FIRST THRU NODE>"):

            metadata["first_thru_node"] = int(stripped.split(">")[1].strip())

        elif stripped.startswith("<NUMBER OF LINKS>"):

            metadata["number_of_links"] = int(stripped.split(">")[1].strip())



    # FIND LINK TABLE HEADER

    header_idx = None

    for i, line in enumerate(lines):

        if (line.strip().startswith("~") and "init_node" in line.lower()):

            header_idx = i
            break


    if header_idx is None:

        raise ValueError(f"Could not locate TNTP link header in {filepath}.")



    # READ LINK TABLE

    link_df = pd.read_csv(
        filepath, sep=r"\s+", skiprows=header_idx + 1,
        names=["init_node","term_node","capacity","length","free_flow_time","b","power",
            "speed","toll","link_type","semicolon"],engine="python")


    # TNTP rows end with a semicolon.
    link_df = link_df.drop(columns="semicolon")


    # Remove malformed rows if present.
    link_df = link_df.dropna(subset=["init_node","term_node"])



    # CHECK FOR DUPLICATE DIRECTED EDGES

    duplicate_endpoints = link_df.duplicated(subset=["init_node","term_node"]).sum()


    if duplicate_endpoints > 0:

        raise ValueError(f"{filepath} contains {duplicate_endpoints} duplicate "
            f"directed node pairs. NetworkX DiGraph would overwrite them.")



    # CREATE DIRECTED GRAPH

    G_original = nx.DiGraph()


    for _, row in link_df.iterrows():

        u = int(row["init_node"])
        v = int(row["term_node"])

        free_flow_time = float(row["free_flow_time"])



        # BASE TRAVERSAL COST

        # Free-flow travel time is used as the shortest-path distance.

        dist = free_flow_time



        # ADD EDGE

        G_original.add_edge(u, v,

            # Attributes required by the interdiction model.
            dist=dist,

            # Calculated later using detour cost.
            penalty=None,
            capacity=float(row["capacity"]),
            interdictable=True,

            # ORIGINAL TNTP ATTRIBUTES
            original_capacity=float(row["capacity"]),
            length=float(row["length"]),
            free_flow_time=free_flow_time,
            b=float(row["b"]),
            power=float(row["power"]),
            speed=float(row["speed"]),
            toll=float(row["toll"]),
            link_type=int(row["link_type"]))



    # RELABEL NODES

    # The remainder of the thesis code assumes consecutive integer node IDs beginning at zero.

    original_nodes = sorted(G_original.nodes())


    node_map = {original_node: new_node for new_node, original_node in enumerate(original_nodes)}


    G = nx.relabel_nodes(G_original, node_map, copy=True)


    # Retain the original TNTP node number.

    for original_node, new_node in node_map.items():

        G.nodes[new_node]["original_node"] = (original_node)



    # SAVE METADATA

    metadata["filepath"] = filepath
    metadata["network_name"] = os.path.basename(filepath)
    metadata["loaded_nodes"] = (G.number_of_nodes())
    metadata["loaded_edges"] = (G.number_of_edges())


    return G, metadata



# CALCULATE DETOUR PENALTIES

def calculate_detour_penalties(G):

    """Calculate topology-derived interdiction penalties.

    For each directed edge (u, v):

        1. Record its normal traversal cost.
        2. Temporarily remove the edge.
        3. Find the shortest alternative path from u to v.
        4. Calculate the detour penalty as:

               alternative_cost - original_cost

    After all finite detour penalties are calculated, the fallback
    penalty for edges without an alternative route is defined as the
    95th percentile of the finite penalty distribution after clipping
    that distribution to a minimum value of 1.

    Returns
    G : networkx.DiGraph
        Graph containing calculated edge penalties.

    penalty_df : pandas.DataFrame
        Edge-level detour diagnostics.

    fallback_penalty : float
        Network-specific fallback penalty."""

    results = []

    edge_list = list(G.edges(data=True))



    # FIRST PASS: CALCULATE DETOUR PENALTIES WHERE ALTERNATIVES EXIST

    for edge_number, (u, v, data) in enumerate(edge_list, start=1,):

        original_cost = float(data["dist"])
        edge_attributes = data.copy()


        # TEMPORARILY REMOVE EDGE

        G.remove_edge(u, v)


        try:

            alternative_cost = nx.shortest_path_length(G, source=u, target=v, weight="dist")
            # Calculate additional travel cost caused by removing the edge.
            raw_penalty = alternative_cost - original_cost
            # Interdiction penalties must remain positive.
            penalty = max(1.0, raw_penalty)
            alternative_exists = True


        except nx.NetworkXNoPath:

            alternative_cost = np.inf
            raw_penalty = np.nan
            # Fallback is calculated after all finite penalties have been observed.
            penalty = np.nan
            alternative_exists = False


        # RESTORE ORIGINAL EDGE

        G.add_edge(u, v, **edge_attributes)



        # SAVE RESULT

        results.append({
            "from_node": u,
            "to_node": v,
            "original_cost": original_cost,
            "alternative_cost": alternative_cost,
            "alternative_exists": alternative_exists,
            "raw_penalty": raw_penalty,
            "penalty": penalty})


        if (edge_number % 500 == 0 or edge_number == len(edge_list)):

            print(f"Processed {edge_number}/{len(edge_list)} edges")



    # CREATE PENALTY DATAFRAME

    penalty_df = pd.DataFrame(results)



    # CALCULATE NETWORK-SPECIFIC FALLBACK PENALTY

    finite_penalties = penalty_df.loc[penalty_df["alternative_exists"],"penalty"]


    if len(finite_penalties) == 0:

        raise ValueError("No finite detour penalties were found. Cannot calculate fallback penalty.")


    fallback_penalty = float(finite_penalties.quantile(0.95))

    print(f"\nFallback penalty (95th percentile): {fallback_penalty:.4f}")



    # ASSIGN FALLBACK TO EDGES WITH NO ALTERNATIVE

    penalty_df.loc[~penalty_df["alternative_exists"], "penalty"] = fallback_penalty



    # WRITE FINAL PENALTIES BACK TO GRAPH

    for _, row in penalty_df.iterrows():

        u = int(row["from_node"])
        v = int(row["to_node"])
        G[u][v]["penalty"] = float(row["penalty"])
        G[u][v]["alternative_cost"] = float(row["alternative_cost"])
        G[u][v]["alternative_exists"] = bool(row["alternative_exists"])


    return (G, penalty_df, fallback_penalty)



# PREPROCESS ONE NETWORK

def preprocess_real_world_network(network_name, network_info, overwrite=False):

    """Load one raw TNTP network, calculate its detour penalties,
    and save the processed graph.

    This should be performed once per physical network.

    Parameters
    network_name : str
        Short identifier for the network.

    network_info : dict
        Dictionary containing raw and processed paths.

    disconnection_penalty : float
        Finite penalty used when no alternative path exists.

    overwrite : bool
        If False, skip preprocessing when the processed file
        already exists."""


    raw_path = network_info["raw"]
    processed_path = network_info["processed"]



    # SKIP EXISTING NETWORK

    if (os.path.exists(processed_path) and not overwrite):

        print(f"\nSkipping {network_name}: processed file already exists.")

        return



    # PRINT NETWORK INFORMATION

    print("\n" + "=" * 70)
    print(f"Preprocessing: {network_name}")
    print(f"Raw file:      {raw_path}")
    print(f"Output file:   {processed_path}")
    print("=" * 70)



    # LOAD RAW NETWORK

    G, metadata = load_tntp_network(filepath=raw_path)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = m / n


    print(f"Loaded network | "
        f"n={n} | "
        f"m={m} | "
        f"m/n={density:.4f}")



    # CALCULATE DETOUR PENALTIES

    G, penalty_df, fallback_penalty = (calculate_detour_penalties(G=G))



    # GRAPH-LEVEL METADATA

    G.graph["real_world_network"] = (network_name)
    G.graph["source_file"] = (raw_path)
    G.graph["density"] = (density)
    G.graph["tntp_metadata"] = (metadata)
    G.graph["fallback_penalty"] = (fallback_penalty)


    # CREATE OUTPUT DIRECTORY

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)



    # SAVE PROCESSED GRAPH

    with open(processed_path, "wb",) as f:

        pickle.dump(G, f)



    # SAVE EDGE-LEVEL PENALTY DATA

    penalty_output = (processed_path.replace(".pkl", "_penalties.csv"))


    penalty_df.to_csv(penalty_output,index=False)



    # PRINT SUMMARY

    print(f"\nSaved processed graph: {processed_path}")

    print(f"\nSaved penalty diagnostics:\n {penalty_output}")

    print("\nPenalty summary:")

    print(penalty_df["penalty"].describe())

    print("\nAlternative path availability:")

    print(penalty_df["alternative_exists"].value_counts())


    print(f"\nFinished preprocessing {network_name}.")



# PREPROCESS ALL NETWORKS

def preprocess_all_real_world_networks(overwrite=False):

    """Preprocess all training and held-out evaluation networks.

    Detour penalties are calculated once and the resulting graphs
    are saved for later use."""

    all_networks = {**REAL_WORLD_TRAINING_NETWORKS, **REAL_WORLD_EVALUATION_NETWORKS}


    for network_name, network_info in all_networks.items():

        preprocess_real_world_network(network_name=network_name, network_info=network_info,
                                      overwrite=overwrite)



# LOAD PROCESSED NETWORK

def load_processed_real_world_network(network_name, training=True):

    """Load a previously processed real-world network.

    Parameters
    network_name : str
        Name of network to load.

    training : bool
        True  -> use training network dictionary.
        False -> use evaluation network dictionary.

    Returns
    G : networkx.DiGraph
        Processed real-world network."""


    if training:

        networks = (REAL_WORLD_TRAINING_NETWORKS)

    else:

        networks = (REAL_WORLD_EVALUATION_NETWORKS)


    if network_name not in networks:

        raise ValueError(f"Unknown real-world network: {network_name}")


    processed_path = (networks[network_name]["processed"])


    if not os.path.exists(processed_path):

        raise FileNotFoundError(
            f"Processed network not found: {processed_path}\n"
            f"Run the preprocessing step in real_world_networks.py first.")


    with open(processed_path, "rb") as f:

        G = pickle.load(f)


    return G



# GENERATE TRAINING INSTANCE

def generate_real_world_network(network_name, min_hops, seed=None, max_attempts=10000):

    """Generate one shortest-path interdiction instance from a
    preprocessed real-world training network.

    The physical network and edge attributes remain fixed.
    Different instances are produced by selecting different
    source-sink pairs.

    The selected source-sink pair must:

        1. contain distinct nodes,
        2. have a directed source-to-sink path,
        3. satisfy the specified minimum unweighted hop count.

    Parameters
    network_name : str
        Name of one of the real-world training networks.

    min_hops : int
        Minimum number of edges in the unweighted source-to-sink
        shortest path.

    seed : int or None
        Random seed controlling source-sink selection.

    max_attempts : int
        Maximum source-sink sampling attempts.

    Returns
    G : networkx.DiGraph

    s : int
        Source node.

    t : int
        Sink node.

    density : float
        Arc-to-node ratio m/n."""

    rng = random.Random(seed)



    # LOAD PREPROCESSED NETWORK

    G = load_processed_real_world_network(network_name=network_name,training=True)


    # Give this instance its own graph object.

    G = G.copy()

    nodes = list(G.nodes())

    if len(nodes) < 2:

        raise ValueError(f"{network_name} contains fewer than two nodes.")



    # SAMPLE SOURCE-SINK PAIR

    for attempt in range(1, max_attempts + 1):

        s, t = rng.sample(nodes, 2)


        # REQUIRE DIRECTED PATH

        if not nx.has_path(G, s, t):

            continue



        # UNWEIGHTED SHORTEST-PATH HOPS

        shortest_path_hops = (nx.shortest_path_length(G,source=s,target=t))


        if (shortest_path_hops < min_hops):

            continue



        # ACCEPT INSTANCE

        density = (G.number_of_edges() / G.number_of_nodes())

        G.graph["source"] = s
        G.graph["sink"] = t
        G.graph["shortest_path_hops"] = (shortest_path_hops)
        G.graph["density"] = (density)


        print(
            f"Accepted real-world instance | "
            f"network={network_name} | "
            f"n={G.number_of_nodes()} | "
            f"m={G.number_of_edges()} | "
            f"density={density:.2f} | "
            f"s={s} | "
            f"t={t} | "
            f"hops={shortest_path_hops} | "
            f"attempts={attempt}")


        return (G, s, t, density)



    # FAILURE

    raise RuntimeError(
        f"Could not find a valid source-sink pair "
        f"for {network_name} after "
        f"{max_attempts} attempts. "
        f"Required min_hops={min_hops}.")



# PREPROCESSING ENTRY POINT

if __name__ == "__main__":

    preprocess_all_real_world_networks(overwrite=False)