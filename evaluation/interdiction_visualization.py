
import os
import torch
import argparse
import networkx as nx
import matplotlib.pyplot as plt

from data.random_networks import generate_one_in_network
from data.interdiction_data import sample_to_tensors
from optimization.mip import solve_instance

from models.tropical_attention import TropicalInterdictionModel
from models.standard_transformer import StandardTransformerInterdictionModel
from models.gnn import GNNInterdictionModel
from models.edge_bias_transformer import EdgeBiasTransformerInterdictionModel
from models.tropical_attention_V2 import TropicalInterdictionModel as TropicalInterdictionModelV2
from evaluation.metrics import shortest_path_after_attack
from evaluation.metrics import max_flow_after_attack
from evaluation.metrics import min_cost_flow_after_attack


PROBLEM_INPUT_DIMS = {
    "shortest_path": 8,
    "max_flow": 7,
    "min_cost_flow": 9
}


def get_model(model_type, problem_type, device):

    """Creates the correct model architecture based on the model_type string."""

    if problem_type not in PROBLEM_INPUT_DIMS:
        raise ValueError(f"Unknown problem type: {problem_type}")

    input_dim = PROBLEM_INPUT_DIMS[problem_type]

    if model_type == "tropical":
        return TropicalInterdictionModel(input_dim=input_dim,d_model=64, n_heads=4, num_layers=2,
                                         device=device).to(device)

    elif model_type == "transformer":
        return StandardTransformerInterdictionModel(input_dim=input_dim, d_model=64, n_heads=4,
                                                    num_layers=2).to(device)

    elif model_type == "gnn":
        return GNNInterdictionModel(input_dim=input_dim,d_model=64,num_layers=2).to(device)

    elif model_type == "edge_transformer":
        return EdgeBiasTransformerInterdictionModel(input_dim=input_dim,d_model=64,n_heads=4,
                                                    num_layers=2).to(device)
    
    elif model_type == "tropical_v2":
        return TropicalInterdictionModelV2(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,device=device
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")



def graph_from_sample(sample, attack_list=None):

    """Reconstruct a graph and apply an optional interdiction decision."""

    G = nx.DiGraph()

    if attack_list is None:
        attack_list = [0] * len(sample["u"])

    problem_type = sample["problem_type"]

    for i, (u, v) in enumerate(zip(sample["u"], sample["v"])):

        attack = attack_list[i]
        edge_data = {"penalty": sample["penalty"][i]}

        if "dist" in sample:
            edge_data["dist"] = sample["dist"][i]

        if "capacity" in sample:
            edge_data["capacity"] = sample["capacity"][i]

        if problem_type in ["shortest_path", "min_cost_flow"]:
            edge_data["dist"] += edge_data["penalty"] * attack

        elif problem_type == "max_flow":
            edge_data["capacity"] *= (1 - attack)

        G.add_edge(u, v, **edge_data)

    return G


def path_edges_from_nodes(path_nodes):

    """Converts a shortest path from a list of nodes into a list of edges."""

    return list(zip(path_nodes[:-1], path_nodes[1:]))



def solve_shortest_path_for_visualization(G, s, t):

    """Return shortest-path objective, path edges, and node sequence."""

    path_nodes = nx.shortest_path(G,source=s,target=t,weight="dist")

    path_edges = path_edges_from_nodes(path_nodes)

    path_length = nx.shortest_path_length(G,source=s,target=t,weight="dist")

    return path_length, path_edges, path_nodes



def solve_max_flow_for_visualization(G, s, t):

    """Return maximum-flow value, positive-flow edges, and flow values."""

    flow_value, flow_dict = nx.maximum_flow(G,_s=s,_t=t,capacity="capacity")

    positive_flow_edges = []
    flow_values = {}

    for u, outgoing in flow_dict.items():
        for v, flow in outgoing.items():
            if flow > 0:
                positive_flow_edges.append((u, v))
                flow_values[(u, v)] = flow

    return flow_value, positive_flow_edges, flow_values



def solve_min_cost_flow_for_visualization(G, s, t,flow_demand):

    """Return minimum cost, positive-flow edges, and flow values."""

    for node in G.nodes():
        G.nodes[node]["demand"] = 0

    G.nodes[s]["demand"] = -flow_demand
    G.nodes[t]["demand"] = flow_demand

    flow_dict = nx.min_cost_flow(G,demand="demand",capacity="capacity",weight="dist")

    flow_cost = nx.cost_of_flow(G,flow_dict,weight="dist")

    positive_flow_edges = []
    flow_values = {}

    for u, outgoing in flow_dict.items():
        for v, flow in outgoing.items():
            if flow > 0:
                positive_flow_edges.append((u, v))
                flow_values[(u, v)] = flow

    return flow_cost, positive_flow_edges, flow_values



def get_attack_edges(sample, attack_list):

    """Converts a 0/1 attack list into actual graph edges. """

    # get graph edge in the same order used in the model
    edge_list = list(zip(sample["u"], sample["v"]))

    # keep only edges where the attack list has a 1 (interdicted)
    return [edge_list[i]
        for i, val in enumerate(attack_list)
        if val == 1]



def solve_follower_for_visualization(G,s,t,problem_type, flow_demand=1):

    """Solve the selected follower problem for visualization."""

    if problem_type == "shortest_path":

        objective, active_edges, path_nodes = (solve_shortest_path_for_visualization(G,s,t))

        return {"objective": objective,"active_edges": active_edges,"flow_values": {},
            "path_nodes": path_nodes}

    elif problem_type == "max_flow":

        objective, active_edges, flow_values = (solve_max_flow_for_visualization(G,s,t))

        return {"objective": objective, "active_edges": active_edges, "flow_values": flow_values,
            "path_nodes": None}

    elif problem_type == "min_cost_flow":

        objective, active_edges, flow_values = (solve_min_cost_flow_for_visualization(G,s,t,flow_demand))

        return {"objective": objective, "active_edges": active_edges, "flow_values": flow_values,
            "path_nodes": None}

    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")



def draw_panel(ax, G, pos, title, active_edges, attack_edges, flow_values=None):

    """Draw one network-interdiction panel.

    Each panel shows:
        - all network edges in gray
        - the active shortest path in blue
        - the interdicted edges in red"""
    

    if flow_values is None:
        flow_values = {}

    # add title to panel
    ax.set_title(title, fontsize=11)

    # draw graph nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=180,node_color="white",edgecolors="black")

    # draw graph edges
    nx.draw_networkx_edges(G,pos,ax=ax,edge_color="lightgray",arrows=True,arrowsize=8,width=1)

    if active_edges:
        active_widths = [2 + flow_values.get(edge, 1) for edge in active_edges]

        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=active_edges,edge_color="blue",
                               arrows=True,arrowsize=12,width=active_widths)
        
    if attack_edges:
        nx.draw_networkx_edges(G,pos, ax=ax, edgelist=attack_edges,edge_color="red",
            arrows=True,arrowsize=14,width=4,style="dashed")


    # draw node labels
    nx.draw_networkx_labels(G,pos,ax=ax,font_size=7)

    if flow_values:
        nx.draw_networkx_edge_labels(G,pos,ax=ax,
            edge_labels={edge: f"{value:g}" for edge, value in flow_values.items()},
            font_size=6)

    # remove axes
    ax.axis("off")


def main():

    """Main visualization workflow.

        1. Regenerates one test graph.
        2. Solves the MIP on that graph.
        3. Loads the trained model.
        4. Gets the model's predicted interdiction decision.
        5. Computes shortest paths before and after interdiction.
        6. Saves a three-panel visualization."""

    # read command line arguments
    parser = argparse.ArgumentParser(
        description="Visualize shortest-path, maximum-flow, or "
            "minimum-cost-flow interdiction for one test graph.")

    parser.add_argument("model_type",type=str,
        choices=["tropical", "transformer", "gnn", "edge_transformer","tropical_v2"],
        default="tropical")
    
    parser.add_argument("problem_type",
        choices=["shortest_path","max_flow","min_cost_flow"],
        default="shortest_path")

    parser.add_argument("--n",type=int,default=30)

    parser.add_argument("--m",type=int,default=75)

    parser.add_argument("--k",type=int,default=1)

    parser.add_argument("--rep",type=int,default=0)

    args = parser.parse_args()

    model_type = args.model_type
    problem_type = args.problem_type
    n = args.n
    m = args.m
    k = args.k
    rep = args.rep
    # use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    print(f"Problem type: {problem_type}")
    print(f"Graph setting: n={n}, m={m}, K={k}, rep={rep}")

    # use same seed in evaluate to visualize same networks
    seed = 999999 + 100000 * n + 100 * m + rep

    # generate graph
    G, s, t, density = generate_one_in_network(n=n,m=m,cost_low=1,cost_high=10,penalty_low=2,penalty_high=10,
                                               capacity_low=1,capacity_high=20,seed=seed)
    
    if problem_type == "min_cost_flow":
        baseline_max_flow = nx.maximum_flow_value(G,_s=s,_t=t,capacity="capacity")
        flow_demand = max(1,int(0.5 * baseline_max_flow))
    else:
        flow_demand = 1

    # solve exact MIP
    sample = solve_instance(G=G,s=s,t=t,density=density,attack_limit=k,problem_type=problem_type,
                            flow_demand=flow_demand)

    if sample is None:
        raise RuntimeError("MIP solve failed for this example.")
    
    # sample_to_tensors uses this value to select the correct features
    sample["problem_type"] = problem_type

    print("Sample keys:", sample.keys())

    # load trained models
    model = get_model(model_type, problem_type, device)

    run_name = f"{model_type}_{problem_type}"

    model.load_state_dict(
        torch.load(f"saved_models/{run_name}_model.pt",map_location=device))

    model.eval()

    # convert graph sample to tensors
    edge_features, edge_bias, _, attack_limit = sample_to_tensors(sample)

    edge_features = edge_features.unsqueeze(0).to(device)
    edge_bias = edge_bias.unsqueeze(0).to(device)

    # one graph with no padding = every edge is real
    mask = torch.ones(1,edge_features.shape[1],dtype=torch.bool,device=device)

    # run model prediction
    with torch.no_grad():
        logits = model(edge_features, edge_bias=edge_bias, mask=mask)

    real_logits = logits[0]

    predicted_attack = torch.zeros_like(real_logits)
    topk_indices = torch.topk(real_logits, k=k).indices
    predicted_attack[topk_indices] = 1.0

    predicted_attack_list = predicted_attack.cpu().int().tolist()

    # MIP optimal attack vector
    optimal_attack_list = sample["attack"]

    # convert attack vectors to edge tuples
    mip_attack_edges = get_attack_edges(sample, optimal_attack_list)
    model_attack_edges = get_attack_edges(sample, predicted_attack_list)

    # No-interdiction vector for the original network
    no_attack = [0] * len(sample["u"])

    # Reconstruct the three network states
    original_G = graph_from_sample(sample,no_attack)

    mip_G = graph_from_sample(sample,optimal_attack_list)

    model_G = graph_from_sample(sample,predicted_attack_list)

    flow_demand = sample.get("flow_demand", 1)

    # Solve the selected follower problem on each network
    original_solution = solve_follower_for_visualization(original_G,s,t,problem_type,flow_demand=flow_demand)

    mip_solution = solve_follower_for_visualization(mip_G,s,t,problem_type,flow_demand=flow_demand)

    model_solution = solve_follower_for_visualization(model_G,s,t,problem_type,flow_demand=flow_demand)

    # Convert attacked edges into readable strings
    mip_attack_str = ", ".join(f"{u}→{v}" for u, v in mip_attack_edges)

    model_attack_str = ", ".join(f"{u}→{v}" for u, v in model_attack_edges)

    if not mip_attack_str:
        mip_attack_str = "None"

    if not model_attack_str:
        model_attack_str = "None"

    # Problem-specific objective label
    objective_labels = {"shortest_path": "Shortest-path length",
        "max_flow": "Maximum flow",
        "min_cost_flow": "Minimum flow cost"}

    objective_label = objective_labels[problem_type]

    # draw figure
    pos = nx.spring_layout(G, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

   # Original network
    draw_panel(ax=axes[0],G=original_G,pos=pos,title=(
            f"Original network\n"
            f"{objective_label}="
            f"{original_solution['objective']:.2f}"),
        active_edges=original_solution["active_edges"],attack_edges=[],
        flow_values=original_solution["flow_values"])

    # MIP-optimal interdiction
    draw_panel(ax=axes[1],G=mip_G,pos=pos,title=(
            f"MIP interdiction\n"
            f"Attack: {mip_attack_str}\n"
            f"{objective_label}="
            f"{mip_solution['objective']:.2f}"),
        active_edges=mip_solution["active_edges"],attack_edges=mip_attack_edges,
        flow_values=mip_solution["flow_values"])

    # Model-predicted interdiction
    draw_panel(ax=axes[2],G=model_G,pos=pos,title=(
            f"{model_type} interdiction\n"
            f"Attack: {model_attack_str}\n"
            f"{objective_label}="
            f"{model_solution['objective']:.2f}"),
        active_edges=model_solution["active_edges"],attack_edges=model_attack_edges,
        flow_values=model_solution["flow_values"])
    
    display_problem = problem_type.replace("_"," ").title()

    fig.suptitle(f"{display_problem} interdiction example: "
        f"{model_type}, n={n}, m={m}, "
        f"K={k}, rep={rep}",
        fontsize=14)

    # save figures
    os.makedirs("results/figures", exist_ok=True)

    output_path = (f"results/figures/interdiction_{model_type}_{problem_type}_"
        f"n{n}_m{m}_k{k}_rep{rep}.png")
    

    plt.tight_layout()
    plt.savefig(output_path,dpi=300,bbox_inches="tight")
    plt.show()

    # Print numerical summary
    print(f"\nSaved figure to {output_path}")

    print(f"Original {objective_label}: " f"{original_solution['objective']}")

    print(f"MIP attack edges: " f"{mip_attack_edges}")

    print(f"MIP {objective_label}: " f"{mip_solution['objective']}")

    print(f"{model_type} attack edges: " f"{model_attack_edges}")

    print(f"{model_type} {objective_label}: " f"{model_solution['objective']}")

    # Optional shortest-path-specific details
    if problem_type == "shortest_path":

        original_path_nodes = (original_solution["path_nodes"])

        mip_path_nodes = (mip_solution["path_nodes"])

        model_path_nodes = (model_solution["path_nodes"])

        print("Original path:"," → ".join(map(str, original_path_nodes)))

        print("MIP path:"," → ".join(map(str, mip_path_nodes)))

        print("Model path:"," → ".join(map(str, model_path_nodes)))


if __name__ == "__main__":
    main()
    