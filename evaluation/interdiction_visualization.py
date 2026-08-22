"""
Created on Thu May 14 07:09:23 2026

@author: emmallen
"""

"""interdiction_visualization.py

Visualize one network-interdiction example using a trained model.

The script regenerates one test graph, solves the selected interdiction
problem exactly using the MIP, loads a trained neural-network model, and
compares the model-predicted interdiction set with the MIP-optimal set.

Three network states are displayed:

1. Original network with no interdiction
2. Network after the MIP-optimal interdiction
3. Network after the model-predicted interdiction

For shortest-path interdiction, the active shortest path is highlighted.
For maximum-flow and minimum-cost-flow interdiction, edges carrying
positive flow are highlighted and labeled with their flow values.

Usage
Run from the repository root:

    PYTHONPATH=. python visualization/interdiction_visualization.py MODEL_TYPE PROBLEM_TYPE

Optional graph arguments:

    --n      Number of nodes
    --m      Number of directed arcs
    --k      Interdiction budget
    --rep    Replication number"""


import os
from xml.parsers.expat import model
import torch
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from data.random_networks import generate_one_in_network
from data.interdiction_data import sample_to_tensors
from optimization.mip import solve_instance

from models.tropical_attention import TropicalInterdictionModel
from models.standard_transformer import StandardTransformerInterdictionModel
from models.gnn import GNNInterdictionModel
from models.edge_bias_transformer import EdgeBiasTransformerInterdictionModel
from models.tropical_attention_V2 import TropicalInterdictionModel as TropicalInterdictionModelV2


# Number of edge-level input features created by sample_to_tensors() for each network-interdiction
# problem, these dimensions must match both: 1. the feature construction in interdiction_data.py, 
# and 2. the model architecture used during training
PROBLEM_INPUT_DIMS = {
    "shortest_path": 8,
    "max_flow": 7,
    "min_cost_flow": 9}



def get_model(model_type, problem_type, device):

    """Construct the model architecture used by the saved checkpoint.

    Parameters
    model_type : str
        Neural-network architecture to construct.

    problem_type : str
        Network-interdiction problem being visualized.

    device : str or torch.device
        CPU or CUDA device on which the model will run.

    Returns
    torch.nn.Module
        Initialized model moved onto the requested device.

    Raises
    ValueError
        If the model or problem type is not recognized."""

    
    # confirm that the selected problem has a known feature dimension
    # without this check, the input projection would be constructed with
    # an undefined number of input features
    if problem_type not in PROBLEM_INPUT_DIMS:
        raise ValueError(f"Unknown problem type: {problem_type}")
 
    # retrieve the number of edge features used by the selected interdiction problem
    input_dim = PROBLEM_INPUT_DIMS[problem_type]
 
    # all Transformer-style models use the same hidden dimension, attention-head count, 
    # number of layers, and dropout probability = holding these settings constant makes 
    # the model comparison more controlled.
 
    # tropical attention model with edge bias
    if model_type == "tropical":
        # construct Tropical Attention Transformer
        return TropicalInterdictionModel(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,dropout=0.1,device=device,use_edge_bias=True
        ).to(device)
 
    # standard transformer model without edge bias
    elif model_type == "transformer":
        # construct the standard Transformer baseline
        return StandardTransformerInterdictionModel(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,dropout=0.1
        ).to(device)
 
    # gnn model 
    elif model_type == "gnn":
        # construct the edge-to-edge message-passing GNN baseline
        return GNNInterdictionModel(
            input_dim=input_dim,d_model=64,num_layers=2
        ).to(device)
 
    # edge bias transformer model
    elif model_type == "edge_transformer":
        # construct the Transformer with additive graph-structure bias
        return EdgeBiasTransformerInterdictionModel(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,dropout=0.1
        ).to(device)
 
    # version 2 tropical attention model with edge bias and slightly different architecture
    elif model_type == "tropical_v2":
        # construct Tropical Attention Transformer Version 2
        return TropicalInterdictionModelV2(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,device=device,dropout=0.1,use_edge_bias=True
        ).to(device)
 
    else:
        raise ValueError(f"Unknown model type: {model_type}")




def validate_attack_list(sample, attack_list):

    """Confirm that an attack vector aligns with the sample edge ordering.

    Parameters
    sample : dict
        Graph sample containing the ordered u and v edge lists.

    attack_list : sequence
        Binary interdiction decision aligned with those edge lists.

    Raises
    ValueError
        If the attack vector length does not equal the number of edges
        or contains values other than zero and one."""

    # The u list contains one entry per directed edge.
    num_edges = len(sample["u"])

    # Every edge must have exactly one corresponding attack label.
    if len(attack_list) != num_edges:
        raise ValueError(f"Attack vector contains {len(attack_list)} labels, "
            f"but the graph contains {num_edges} edges.")

    # Restrict the visualization to binary interdiction decisions.
    if any(value not in (0, 1) for value in attack_list):
        raise ValueError("Attack vector must contain only binary values 0 and 1.")


    

def graph_from_sample(sample, attack_list=None):

    """Reconstruct a directed graph after applying an interdiction vector.

    The effect of interdiction depends on the problem:
    shortest_path:
        Add the edge-specific penalty to every attacked edge's distance.
    max_flow:
        Set every attacked edge's capacity to zero.
    min_cost_flow:
        Add the edge-specific penalty to every attacked edge's
        per-unit flow cost.

    Parameters
    sample : dict
        Solved graph sample containing ordered edge lists and edge
        attributes.

    attack_list : sequence, optional
        Binary attack decision aligned with the stored edge ordering.
        When omitted, no edges are interdicted.

    Returns
    networkx.DiGraph
        Reconstructed graph with post-attack attributes."""

    # create a new directed graph so the original graph is not modified
    G = nx.DiGraph()

    # when no attack vector is supplied, construct an all-zero vector
    # representing the original, uninterdicted network
    if attack_list is None:
        attack_list = [0] * len(sample["u"])

    # ensure that the attack vector aligns with the sample edge lists
    validate_attack_list(sample,attack_list)

    # determine which edge attribute is modified by interdiction
    problem_type = sample["problem_type"]

    # rebuild each directed edge in the same order used by the model,
    # MIP attack labels, and feature tensors
    for edge_index, (u, v) in enumerate(zip(sample["u"], sample["v"])):

        # binary interdiction decision for the current edge
        attack = attack_list[edge_index]

        # every problem stores an edge-specific interdiction penalty
        edge_data = {"penalty": sample["penalty"][edge_index],
                     "interdictable": bool(sample["interdictable"][edge_index]),}

        # shortest-path and minimum-cost-flow samples store distance or
        # cost values under the dist field
        if "dist" in sample:
            edge_data["dist"] = sample["dist"][edge_index]

        # maximum-flow and minimum-cost-flow samples store capacities
        if "capacity" in sample:
            edge_data["capacity"] = sample["capacity"][edge_index]

        if problem_type in ["shortest_path", "min_cost_flow"]:
            # these two formulations retain the edge but add its
            # interdiction penalty to the edge cost
            edge_data["dist"] += edge_data["penalty"] * attack

        elif problem_type == "max_flow":
            # maximum-flow interdiction removes all capacity from an
            # attacked edge while leaving unattacked capacity unchanged
            edge_data["capacity"] *= (1 - attack)

        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

        # add the post-attack directed edge to the graph
        G.add_edge(u, v, **edge_data)

    return G




def path_edges_from_nodes(path_nodes):

    """Convert an ordered node path into its directed edge sequence."""

    # pair every path node with the node immediately following it
    return list(zip(path_nodes[:-1], path_nodes[1:]))





def solve_shortest_path_for_visualization(G, s, t):

    """Solve the shortest-path follower problem for one graph.

    Parameters
    G : networkx.DiGraph
        Directed graph whose dist attributes include any interdiction
        penalties.

    s : int
        Source node.

    t : int
        Sink node.

    Returns
    path_length : float
        Shortest source-to-sink path length.

    path_edges : list[tuple]
        Directed edges belonging to the selected shortest path.

    path_nodes : list
        Ordered nodes belonging to the selected shortest path."""


    # compute one shortest source-to-sink node sequence
    path_nodes = nx.shortest_path(G,source=s,target=t,weight="dist")

    # convert the node sequence into directed edge tuples so the active
    # path can be highlighted in the figure
    path_edges = path_edges_from_nodes(path_nodes)

    # compute the total distance of that shortest path
    path_length = nx.shortest_path_length(G,source=s,target=t,weight="dist")

    return path_length, path_edges, path_nodes




def solve_max_flow_for_visualization(G, s, t):

    """Solve the maximum-flow follower problem for one graph.

    Returns the maximum-flow objective, all edges carrying positive
    flow, and the flow amount assigned to each active edge."""

    # solve the maximum source-to-sink flow problem using the graph's
    # post-interdiction capacity attributes
    flow_value, flow_dict = nx.maximum_flow(G,s=s,t=t,capacity="capacity")

    # store only edges carrying positive flow so inactive edges are not
    # highlighted in the figure
    positive_flow_edges = []

    # map each positive-flow edge to the amount of flow it carries 
    flow_values = {}

    # flow_dict[u][v] gives the flow assigned to directed edge (u,v)
    for u, outgoing in flow_dict.items():
        for v, flow in outgoing.items():
            if flow > 0:
                positive_flow_edges.append((u, v))
                flow_values[(u, v)] = flow

    return flow_value, positive_flow_edges, flow_values




def solve_min_cost_flow_for_visualization(G, s, t,flow_demand):

    """Solve the minimum-cost-flow follower problem for one graph.

    Parameters
    G : networkx.DiGraph
        Directed graph containing post-interdiction dist and capacity
        attributes.

    s : int
        Source node supplying the required flow.

    t : int
        Sink node receiving the required flow.

    flow_demand : int or float
        Amount of flow that must be sent from source to sink.

    Returns
    flow_cost : float
        Minimum total cost required to satisfy the demand.

    positive_flow_edges : list[tuple]
        Edges carrying positive flow in the minimum-cost solution.

    flow_values : dict
        Flow amount assigned to each positive-flow edge."""

    # initialize every node as a transshipment node with no external
    # supply or demand
    for node in G.nodes():
        G.nodes[node]["demand"] = 0

    # NetworkX represents supply using negative demand
    G.nodes[s]["demand"] = -flow_demand

    # the sink requires the same quantity of inflow    
    G.nodes[t]["demand"] = flow_demand

    # solve the capacitated minimum-cost-flow problem
    flow_dict = nx.min_cost_flow(G,demand="demand",capacity="capacity",weight="dist")

    # calculate the total cost of the returned flow assignment
    flow_cost = nx.cost_of_flow(G,flow_dict,weight="dist")

    # store edges that carry positive flow for visualization
    positive_flow_edges = []
    flow_values = {}

    for u, outgoing in flow_dict.items():
        for v, flow in outgoing.items():
            if flow > 0:
                positive_flow_edges.append((u, v))
                flow_values[(u, v)] = flow

    return flow_cost, positive_flow_edges, flow_values




def get_attack_edges(sample, attack_list):

    """Convert a binary attack vector into directed edge tuples.

    The attack vector and graph edge lists must use the same ordering."""

    # confirm one binary attack label exists for every graph edge
    validate_attack_list(sample,attack_list)

    # reconstruct the ordered graph edge list used throughout the
    # dataset and model pipeline
    edge_list = list(zip(sample["u"], sample["v"]))

    # return only edges where the attack label = 1 (interdicted)
    return [edge_list[i] for i, val in enumerate(attack_list) if val == 1]




def solve_follower_for_visualization(G,s,t,problem_type, flow_demand=1):

    """Solve the follower problem associated with an interdiction model.

    A common dictionary format is returned so the plotting code can
    handle shortest path, maximum flow, and minimum-cost flow through
    the same interface."""

    if problem_type == "shortest_path":

        # solve the shortest-path problem and retain the selected path
        objective, active_edges, path_nodes = (solve_shortest_path_for_visualization(G,s,t))

        return {"objective": objective,"active_edges": active_edges,"flow_values": {},
            "path_nodes": path_nodes}

    elif problem_type == "max_flow":

        # solve the maximum-flow problem and retain positive-flow edges
        objective, active_edges, flow_values = (solve_max_flow_for_visualization(G,s,t))

        return {"objective": objective, "active_edges": active_edges, "flow_values": flow_values,
            "path_nodes": None}

    elif problem_type == "min_cost_flow":

        # solve minimum-cost flow for the demand stored in the sample
        objective, active_edges, flow_values = (solve_min_cost_flow_for_visualization(G,s,t,flow_demand))

        return {"objective": objective, "active_edges": active_edges, "flow_values": flow_values,
            "path_nodes": None}

    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")




def draw_panel(ax, G, pos, title, active_edges, attack_edges, flow_values=None):

    """Draw one network-interdiction comparison panel.

    Visual conventions
    Gray solid edges:
        All graph edges.

    Blue solid edges:
        Active shortest-path or positive-flow edges.

    Red dashed edges:
        Interdicted edges.

    White nodes:
        Graph nodes.

    Flow labels:
        Positive flow values for maximum-flow or minimum-cost-flow
        solutions."""
    

    # use an empty mapping when the current problem has no flow labels
    if flow_values is None:
        flow_values = {}

    # display the network state and its objective above the panel
    ax.set_title(title, fontsize=11)

    # draw graph nodes as white circles with black borders
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=180,node_color="white",edgecolors="black")

    # draw every directed edge as a light-gray background edge
    nx.draw_networkx_edges(G,pos,ax=ax,edge_color="lightgray",arrows=True,arrowsize=8,width=1)

    if active_edges:

        # shortest-path edges receive the default width of three
        # for flow problems, edges carrying more flow are drawn wider
        # cap the additional width so very large flow values do not overwhelm the figure
        active_widths = [2 + min(flow_values.get(edge, 1), 6) for edge in active_edges]

        # highlight the active shortest path or positive-flow edges
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=active_edges,edge_color="blue",
                               arrows=True,arrowsize=12,width=active_widths)
        
    if attack_edges:

        # draw interdicted edges last so they remain visible even when
        # they overlap an active path or flow edge
        nx.draw_networkx_edges(G,pos, ax=ax, edgelist=attack_edges,edge_color="red",
            arrows=True,arrowsize=14,width=4,style="dashed")


    # label graph nodes using their node identifiers
    nx.draw_networkx_labels(G,pos,ax=ax,font_size=7)

    if flow_values:
        # label only edges carrying positive flow
        nx.draw_networkx_edge_labels(G,pos,ax=ax,
            edge_labels={edge: f"{value:g}" for edge, value in flow_values.items()},
            font_size=6)

    # hide the Matplotlib coordinate axes
    ax.axis("off")



def main():

    """Generate and visualize one interdiction example.

    Workflow
    1. Read command-line settings.
    2. Regenerate one unseen evaluation graph.
    3. Solve the exact interdiction MIP.
    4. Load the trained model's best checkpoint.
    5. Predict exactly k interdicted edges.
    6. Reconstruct the original, MIP-attacked, and model-attacked graphs.
    7. Solve the appropriate follower problem on each graph.
    8. Save and display a three-panel comparison figure."""



    # COMMAND LINE ARGUMENTS

    # read command line arguments
    parser = argparse.ArgumentParser(
        description="Visualize shortest-path, maximum-flow, or "
            "minimum-cost-flow interdiction for one test graph.")

    # make the model positional argument optional so the supplied
    # default is actually used when the argument is omitted
    parser.add_argument("model_type",nargs="?",type=str,
        choices=["tropical", "transformer", "gnn", "edge_transformer","tropical_v2"],
        default="tropical")

    # make the problem positional argument optional for the same reason
    parser.add_argument("problem_type",nargs="?",type=str,
        choices=["shortest_path","max_flow","min_cost_flow"],
        default="shortest_path")

    parser.add_argument("--eval_mode",nargs="?", type=str,
        choices=["id_new", "ood_size", "wood", "external"],
        default="id_new",)

    # number of graph nodes
    parser.add_argument("--n",type=int,default=30)

    # number of directed graph edges
    parser.add_argument("--m",type=int,default=75)

    # number of edges that may be interdicted
    parser.add_argument("--k",type=int,default=1)

    # replication number used to reproduce a specific test graph
    parser.add_argument("--rep",type=int,default=0)

    args = parser.parse_args()

    # store command-line arguments under concise local names
    model_type = args.model_type
    problem_type = args.problem_type
    n = args.n
    m = args.m
    k = args.k
    rep = args.rep
    eval_mode = args.eval_mode

    # validate basic graph and attack settings before generating data
    if n < 2:
        raise ValueError("The graph must contain at least two nodes.")

    if m < 1:
        raise ValueError("The graph must contain at least one edge.")

    if k < 0:
        raise ValueError("The interdiction budget cannot be negative.")

    # use CUDA when available; otherwise use the CPU
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    print(f"Problem type: {problem_type}")
    print(f"Graph setting: n={n}, m={m}, K={k}, rep={rep}")



    # REPRODUCE ONE EVALUATION GRAPH

    # LOAD FIXED EVALUATION GRAPH

    graph_path = (f"evaluation_graphs/"
        f"{problem_type}_{eval_mode}_graphs.pkl")

    with open(graph_path, "rb") as f:
        evaluation_graphs = pickle.load(f)

    if eval_mode in ("id_new", "ood_size"):

        matching_graphs = [graph_data for graph_data in evaluation_graphs
                           if graph_data["n"] == n and graph_data["m"] == m
                           and graph_data["rep"] == rep]

        if not matching_graphs:
            raise ValueError(f"No evaluation graph found for "
                f"n={n}, m={m}, rep={rep}.")

        graph_data = matching_graphs[0]


    elif eval_mode == "wood":

        matching_graphs = [graph_data for graph_data in evaluation_graphs
                           if graph_data["wood_problem"] == rep]

        if not matching_graphs:
            raise ValueError(f"No Wood problem {rep} found.")

        graph_data = matching_graphs[0]

        # Wood uses the problem-specific stored budget
        k = graph_data["attack_budget"]

    elif eval_mode == "external":

        if len(evaluation_graphs) != 1:
            raise ValueError("Expected exactly one external evaluation graph.")

        graph_data = evaluation_graphs[0]


    G = graph_data["G"]
    s = graph_data["s"]
    t = graph_data["t"]
    density = graph_data["density"]

    n = graph_data["n"]
    m = graph_data["m"]

    seed = graph_data.get("seed", None)
    wood_problem = graph_data.get("wood_problem", None)
    network_name = graph_data.get("network_name", None)


    # the interdiction budget cannot exceed the number of generated directed edges
    if k > G.number_of_edges():
        raise ValueError(f"Attack limit K={k} exceeds the graph's "
            f"{G.number_of_edges()} edges.")


    # DETERMINE MIN COST FLOW DEMAND
    
    if problem_type == "min_cost_flow":
        # compute pre-interdiction maximum flow so the requested min-cost-flow demand is 
        # feasible on the original graph
        baseline_max_flow = nx.maximum_flow_value(G,s=s,t=t,capacity="capacity")
        # require 50% of the pre-interdiction maximum flow, with a minimum demand of one unit
        flow_demand = max(1,int(0.5 * baseline_max_flow))

    else:
        # the other problem formulations do not use flow_demand, but solve_instance 
        # accepts one shared interface
        flow_demand = 1



    # solve exact interdiction MIP
    sample = solve_instance(G=G,s=s,t=t,density=density,attack_limit=k,problem_type=problem_type,
                            flow_demand=flow_demand)

    # solve_instance returns None when it cannot produce a valid optimal
    # training-style sample
    if sample is None:
        raise RuntimeError("MIP solve failed for this example.")
    
    # ensure downstream helper functions can determine how interdiction changes the graph
    sample["problem_type"] = problem_type

   

   # LOAD TEH TRAINED MODEL CHECKPOINT

    # construct the same architecture used during training
    model = get_model(model_type, problem_type, device)

    # reproduce the checkpoint naming convention from train.py
    run_name = f"{model_type}_{problem_type}"

    # use the checkpoint with the lowest validation loss
    checkpoint_path = (f"saved_models/{run_name}_best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find model checkpoint: {checkpoint_path}")

    # load the complete checkpoint dictionary so the model state dict can be extracted
    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=False)

    # copy the saved learned parameters into the model architecture
    model.load_state_dict(checkpoint["model_state_dict"])

    # disable dropout and other training-specific behavior
    model.eval()



    # CONVERT GRAPH TO MODEL INPUT TENSORS

    # sample_to_tensors returns edge features, graph-structure bias,
    # MIP attack labels, and attack limit
    edge_features, edge_bias, _, _ = sample_to_tensors(sample)

    # add a batch dimension and move edge features to the selected
    # device using the same dtype used during training
    edge_features = (edge_features.unsqueeze(0).to(device,dtype=torch.float32,))

    # add a batch dimension to the pairwise edge-bias matrix
    edge_bias = (edge_bias.unsqueeze(0).to(device,dtype=torch.float32,))

    # this batch contains one graph and therefore requires no padding
    # mark every edge position as valid
    mask = torch.ones(1,edge_features.shape[1],dtype=torch.bool,device=device)



    # GENERATE MODEL-PREDICTED INTERDICTION

    # disable gradient tracking because visualization requires only a forward prediction
    with torch.no_grad():
        logits = model(edge_features, edge_bias=edge_bias, mask=mask)

    # remove the batch dimension to obtain one score per graph edge
    real_logits = logits[0]

    # confirm that the budget is valid for the model's edge sequence
    num_edges = real_logits.numel()

    if not 0 <= k <= num_edges:
        raise ValueError(f"Invalid attack limit K={k} for a graph "
            f"with {num_edges} edges.")

    # identify arcs that are eligible for interdiction
    interdictable_mask = torch.tensor(sample["interdictable"],dtype=torch.bool,device=device,)

    num_interdictable = int(interdictable_mask.sum().item())

    if k > num_interdictable:
        raise ValueError(f"Attack limit K={k} exceeds the number "
            f"of interdictable arcs ({num_interdictable}).")

    # initialize an all-zero binary predicted attack vector
    predicted_attack = torch.zeros_like(real_logits)

    if k > 0:

        masked_logits = real_logits.clone()

        # forbidden arcs can never be selected
        masked_logits[~interdictable_mask] = float("-inf")
        # select the k edges with the largest model logits
        topk_indices = torch.topk(masked_logits,k=k,).indices
        # mark those edges as interdicted
        predicted_attack[topk_indices] = 1.0
        # convert the prediction to ordinary integer labels
        predicted_attack_list = predicted_attack.cpu().int().tolist()


    # retrieve the exact MIP-optimal binary attack vector
    optimal_attack_list = sample["attack"]



    # CONVERT ATTACKS INTO GRAPH EDGES

    # directed edges selected by the MIP
    mip_attack_edges = get_attack_edges(sample, optimal_attack_list)
    # directed edges selected by the neural-network model
    model_attack_edges = get_attack_edges(sample, predicted_attack_list)

    # all-zero attack vector representing the original network
    no_attack = [0] * len(sample["u"])



    # RECONSTRUCT THE THREE GRAPH STATES

    original_G = graph_from_sample(sample,no_attack)
    mip_G = graph_from_sample(sample,optimal_attack_list)
    model_G = graph_from_sample(sample,predicted_attack_list)

    # use the exact flow demand stored with the solved sample
    flow_demand = sample.get("flow_demand", 1)



    # SOLVE EACH FOLLOWER PROBLEM

    # Solve the selected follower problem on each network
    original_solution = solve_follower_for_visualization(original_G,s,t,problem_type,flow_demand=flow_demand)
    mip_solution = solve_follower_for_visualization(mip_G,s,t,problem_type,flow_demand=flow_demand)
    model_solution = solve_follower_for_visualization(model_G,s,t,problem_type,flow_demand=flow_demand)



    # CREATE READALE FIGURE LABELS

    # convert attack-edge tuples into compact arrow-separated strings
    mip_attack_str = ", ".join(f"{u}→{v}" for u, v in mip_attack_edges)
    model_attack_str = ", ".join(f"{u}→{v}" for u, v in model_attack_edges)


    # display "None" when K=0 and no edge is interdicted
    if not mip_attack_str:
        mip_attack_str = "None"
    if not model_attack_str:
        model_attack_str = "None"


    # human-readable objective name for the selected problem
    objective_labels = {"shortest_path": "Shortest-path length",
        "max_flow": "Maximum flow",
        "min_cost_flow": "Minimum flow cost"}

    objective_label = objective_labels[problem_type]


    # replace internal model-name formatting for figure titles
    display_model = model_type.replace("_"," ",).title()
    display_problem = problem_type.replace("_"," ",).title()



    # DRAW THE THREE-PANEL COMPARISON FIGURE

    # generate one fixed node layout from the original graph - reusing the same
    # positions in all three panels makes structural differences easier to compare
    pos = nx.spring_layout(G, seed=42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    # original network before any interdiction
    draw_panel(ax=axes[0],G=original_G,pos=pos,title=(
            f"Original network\n"
            f"{objective_label}="
            f"{original_solution['objective']:.2f}"),
        active_edges=original_solution["active_edges"],attack_edges=[],
        flow_values=original_solution["flow_values"])


    # network after the MIP-optimal attack
    draw_panel(ax=axes[1],G=mip_G,pos=pos,title=(
            f"MIP interdiction\n"
            f"Attack: {mip_attack_str}\n"
            f"{objective_label}="
            f"{mip_solution['objective']:.2f}"),
        active_edges=mip_solution["active_edges"],attack_edges=mip_attack_edges,
        flow_values=mip_solution["flow_values"])


    # network after the model-predicted attack
    draw_panel(ax=axes[2],G=model_G,pos=pos,title=(
            f"{model_type} interdiction\n"
            f"Attack: {model_attack_str}\n"
            f"{objective_label}="
            f"{model_solution['objective']:.2f}"),
        active_edges=model_solution["active_edges"],attack_edges=model_attack_edges,
        flow_values=model_solution["flow_values"])

    if eval_mode == "wood":

        graph_label = (f"Wood problem {wood_problem}")

    elif eval_mode == "external":

        graph_label = (network_name or "External network")

    else:

        graph_label = (f"n={n}, m={m}, rep={rep}")

    # add experiment settings above the complete figure
    fig.suptitle(f"{display_problem} interdiction example: "
        f"{display_model}\n"
        f"{graph_label}, K={k}", fontsize=14)



    # SAVE AND DISPLAY FIGURE
    os.makedirs("results/figures", exist_ok=True)
    output_path = (f"results/figures/interdiction_{model_type}_{problem_type}_"
        f"n{n}_m{m}_k{k}_rep{rep}.png")
    
    # leave space above the panels for the overall figure title
    plt.tight_layout()
    plt.savefig(output_path,dpi=300,bbox_inches="tight")
    plt.show()

    # close the figure after displaying it so repeated runs do not
    # retain unnecessary Matplotlib objects in memory
    plt.close(fig)



    # PRINT NUMERICAL SUMMARY
    print(f"\nSaved figure to {output_path}")

    print(f"Original {objective_label}: " f"{original_solution['objective']}")

    print(f"MIP attack edges: " f"{mip_attack_edges}")

    print(f"MIP {objective_label}: " f"{mip_solution['objective']}")

    print(f"{display_model} attack edges: "f"{model_attack_edges}")

    print(f"{display_model} {objective_label}: "f"{model_solution['objective']}")


    # Shortest-path solutions also have an interpretable node sequence.
    if problem_type == "shortest_path":

        original_path_nodes = (original_solution["path_nodes"])

        mip_path_nodes = (mip_solution["path_nodes"])

        model_path_nodes = (model_solution["path_nodes"])

        print("Original path:"," → ".join(map(str, original_path_nodes)))
        print("MIP path:"," → ".join(map(str, mip_path_nodes)))
        print("Model path:"," → ".join(map(str, model_path_nodes)))


if __name__ == "__main__":
    main()
    