
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
from evaluation.metrics import shortest_path_after_attack


def get_model(model_type, device):

    """Creates the correct model architecture based on the model_type string."""

    if model_type == "tropical":
        return TropicalInterdictionModel(
            input_dim=7,
            d_model=64,
            n_heads=4,
            num_layers=2,
            device=device
        ).to(device)

    elif model_type == "transformer":
        return StandardTransformerInterdictionModel(
            input_dim=7,
            d_model=64,
            n_heads=4,
            num_layers=2
        ).to(device)

    elif model_type == "gnn":
        return GNNInterdictionModel(
            input_dim=7,
            d_model=64,
            num_layers=2
        ).to(device)

    elif model_type == "edge_transformer":
        return EdgeBiasTransformerInterdictionModel(
            input_dim=7,
            d_model=64,
            n_heads=4,
            num_layers=2
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")



def path_edges_from_nodes(path_nodes):

    """Converts a shortest path from a list of nodes into a list of edges."""

    return list(zip(path_nodes[:-1], path_nodes[1:]))


def shortest_path_edges_after_attack(G, s, t, attack_edges):

    """Removes the interdicted edges from a copy of the graph,
    then computes the new shortest path from source s to target t."""

    # copy original graph
    G_temp = G.copy()

    # remove the attacked/interdicted edges
    G_temp.remove_edges_from(attack_edges)

    # compute new shortest path after edges are removed
    path_nodes = nx.shortest_path(
        G_temp,source=s,target=t,weight="dist")
    
    path_edges = path_edges_from_nodes(path_nodes)

    path_length = nx.shortest_path_length(G_temp, source=s, target=t, weight="dist")

    # convert node path into edge form 
    return path_nodes, path_edges, path_length


def get_attack_edges(sample, attack_list):

    """Converts a 0/1 attack list into actual graph edges. """

    # get graph edge in the same order used in the model
    edge_list = list(zip(sample["u"], sample["v"]))

    # keep only edges where the attack list has a 1 (interdicted)
    return [edge_list[i]
        for i, val in enumerate(attack_list)
        if val == 1]


def draw_panel(ax, G, pos, title, path_edges, attack_edges):

    """Draws one subplot/panel of the figure.

    Each panel shows:
        - all network edges in gray
        - the active shortest path in blue
        - the interdicted edges in red"""

    # add title to panel
    ax.set_title(title, fontsize=12)

    # draw graph nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=180,node_color="white",edgecolors="black")

    # draw graph edges
    nx.draw_networkx_edges(
        G,pos,ax=ax,edge_color="lightgray",arrows=True,arrowsize=8,width=1)

    # draw shortest path
    nx.draw_networkx_edges(
        G,pos,ax=ax,edgelist=path_edges,edge_color="blue",arrows=True,arrowsize=12,
        width=3)

    # draw interdicted edges
    nx.draw_networkx_edges(
        G,pos,ax=ax,edgelist=attack_edges,edge_color="red",arrows=True,arrowsize=14,
        width=4)

    # draw node labels
    nx.draw_networkx_labels(G,pos,ax=ax,font_size=7)

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
        description="Visualize shortest-path interdiction for one test graph.")

    parser.add_argument(
        "--model_type",type=str,default="tropical",
        choices=["tropical", "transformer", "gnn", "edge_transformer"],
        help="Which trained model to visualize.")

    parser.add_argument(
        "--n",type=int,default=30,help="Number of nodes in the test graph.")

    parser.add_argument(
        "--m",type=int,default=75,help="Number of edges in the test graph.")

    parser.add_argument(
        "--k",type=int,default=1,help="Interdiction budget / number of attacked edges.")

    parser.add_argument(
        "--rep",type=int,default=0,help="Replication number used to reproduce a test graph.")

    args = parser.parse_args()

    model_type = args.model_type
    n = args.n
    m = args.m
    k = args.k
    rep = args.rep

    # use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    print(f"Graph setting: n={n}, m={m}, K={k}, rep={rep}")

    # use same seed in evaluate to visualize same networks
    seed = 999999 + 100000 * n + 100 * m + 10000 * k + rep

    # generate graph
    G, s, t, density = generate_one_in_network(
        n=n,m=m,cost_low=1,cost_high=10,seed=seed)

    # solve exact MIP
    sample = solve_instance(
        G=G,s=s,t=t,density=density,attack_limit=k)

    if sample is None:
        raise RuntimeError("MIP solve failed for this example.")

    print("Sample keys:", sample.keys())

    # load trained models
    model = get_model(model_type, device)

    model.load_state_dict(
        torch.load(f"saved_models/{model_type}_model.pt",map_location=device))

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

    # compute shortest paths
    original_path_nodes = nx.shortest_path(
        G,source=s,target=t,weight="dist")

    original_path_edges = path_edges_from_nodes(original_path_nodes)

    original_length = nx.shortest_path_length(
        G,source=s,target=t,weight="dist")

    mip_path_nodes, mip_path_edges, mip_length = shortest_path_after_attack(
    G, s, t, mip_attack_edges)

    model_path_nodes, model_path_edges, model_length = shortest_path_after_attack(
    G, s, t, model_attack_edges)

    original_path_str = " → ".join(map(str, original_path_nodes))
    mip_path_str = " → ".join(map(str, mip_path_nodes))
    model_path_str = " → ".join(map(str, model_path_nodes))

    mip_attack_str = ", ".join([f"{u}→{v}" for u, v in mip_attack_edges])
    model_attack_str = ", ".join([f"{u}→{v}" for u, v in model_attack_edges])

    # draw figure
    pos = nx.spring_layout(G, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    draw_panel(axes[0],G,pos,
        f"Original shortest path\nLength={original_length}\nPath: {original_path_str}",
        original_path_edges,[])

    draw_panel(axes[1],G,pos,    
        f"MIP interdiction\nAttack: {mip_attack_str}\nLength={mip_length}\nPath: {mip_path_str}",
        mip_path_edges,mip_attack_edges)

    draw_panel(axes[2],G,pos,   
        f"{model_type} interdiction\nAttack: {model_attack_str}\nLength={model_length}\nPath: {model_path_str}",
        model_path_edges,model_attack_edges)

    fig.suptitle(
        f"Shortest-path interdiction example: {model_type}, n={n}, m={m}, K={k}, rep={rep}",
        fontsize=14)

    # save figures
    os.makedirs("results/figures", exist_ok=True)

    output_path = (f"results/figures/interdiction_{model_type}_"
        f"n{n}_m{m}_k{k}_rep{rep}.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to {output_path}")
    print("Original path edges:", original_path_edges)
    print("Original path length:", original_length)
    print(f"MIP attack edges: {mip_attack_edges}")
    print(f"MIP objective: {sample['path_length']}")
    print(f"{model_type} attack edges: {model_attack_edges}")

    # verification of objective value
    predicted_objective = shortest_path_after_attack(sample,predicted_attack_list)
    print(f"Predicted objective: {predicted_objective}")


if __name__ == "__main__":
    main()