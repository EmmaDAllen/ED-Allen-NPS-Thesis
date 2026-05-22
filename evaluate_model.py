# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:51:24 2026

@author: emmallen
"""

import torch
import networkx as nx
import pyomo.environ as pyo
import csv

from random_test_networks import generate_one_in_network
from training_set_generation import solve_instance
from train_tropical_model import collate_graphs
from tropical_attention import TropicalInterdictionModel


def sample_to_tensors(sample):
    n = sample["n_nodes"]
    density = sample["density"]
    attack_limit = sample["attack_limit"]

    u = torch.tensor(sample["u"], dtype=torch.float32)
    v = torch.tensor(sample["v"], dtype=torch.float32)
    dist = torch.tensor(sample["dist"], dtype=torch.float32)

    u_norm = u / max(n - 1, 1)
    v_norm = v / max(n - 1, 1)
    dist_norm = dist / 10.0

    source_flag = (u == sample["source"]).float()
    sink_flag = (v == sample["sink"]).float()

    density_feature = torch.full_like(u_norm, density / 10.0)
    budget_feature = torch.full_like(u_norm, attack_limit / 10.0)

    edge_features = torch.stack([
        u_norm,
        v_norm,
        dist_norm,
        source_flag,
        sink_flag,
        density_feature,
        budget_feature
    ], dim=1)

    edge_bias = (v.unsqueeze(1) == u.unsqueeze(0)).float()
    edge_bias = 0.5 * edge_bias

    return edge_features, edge_bias


def shortest_path_after_attack(sample, predicted_attack):
    G = nx.DiGraph()

    s = sample["source"]
    t = sample["sink"]

    for u, v, dist, attack in zip(
        sample["u"],
        sample["v"],
        sample["dist"],
        predicted_attack
    ):
        new_dist = dist + attack
        G.add_edge(u, v, dist=new_dist)

    return nx.shortest_path_length(G, source=s, target=t, weight="dist")


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TropicalInterdictionModel(
        input_dim=7,
        d_model=64,
        n_heads=4,
        num_layers=2,
        device=device
    ).to(device)

    model.load_state_dict(
        torch.load("tropical_interdiction_model.pt", map_location=device)
    )

    model.eval()

    test_settings = [
        (30, 75),
        (30, 120),
        (50, 125),
        (50, 200),
        (75, 188),
        (75, 300),
    ]

    reps_per_setting = 20
    max_attacks = 1
    base_seed = 999999

    total_exact = 0
    total_hamming = 0
    total_gap = 0.0
    total_instances = 0
    
    results_rows = []

    for n, m in test_settings:
        for rep in range(reps_per_setting):
            seed = base_seed + 100000 * n + 100 * m + rep

            G, s, t, density = generate_one_in_network(
                n=n,
                m=m,
                cost_low=1,
                cost_high=10,
                seed=seed
            )

            sample = solve_instance(
                G=G,
                s=s,
                t=t,
                density=density,
                max_attacks=max_attacks
            )

            if sample is None:
                continue

            edge_features, edge_bias = sample_to_tensors(sample)

            edge_features = edge_features.unsqueeze(0).to(device)
            edge_bias = edge_bias.unsqueeze(0).to(device)

            mask = torch.ones(
                1,
                edge_features.shape[1],
                dtype=torch.bool,
                device=device
            )

            with torch.no_grad():
                logits = model(
                    edge_features,
                    edge_bias=edge_bias,
                    mask=mask
                )

            real_logits = logits[0]
            k = sample["attack_limit"]

            predicted_attack = torch.zeros_like(real_logits)
            topk_indices = torch.topk(real_logits, k=k).indices
            predicted_attack[topk_indices] = 1.0

            predicted_attack_list = predicted_attack.cpu().int().tolist()
            optimal_attack_list = sample["attack"]

            exact = int(predicted_attack_list == optimal_attack_list)
            hamming = sum(
                p != o for p, o in zip(predicted_attack_list, optimal_attack_list)
            )

            optimal_objective = sample["path_length"]
            predicted_objective = shortest_path_after_attack(
                sample,
                predicted_attack_list
            )

            objective_gap = (
                optimal_objective - predicted_objective
            ) / max(abs(optimal_objective), 1e-8)
            
            
            results_rows.append({
                "n_nodes": n,
                "n_edges": m,
                "density": m / n,
                "replication": rep,

                "optimal_objective": optimal_objective,
                "predicted_objective": predicted_objective,
                "objective_gap": objective_gap,

                "hamming_distance": hamming,
                "exact_match": exact,

                "num_correct_edges": sum(
                    p == o for p, o in zip(predicted_attack_list, optimal_attack_list)),  

                "num_predicted_attacks": sum(predicted_attack_list),
                "num_optimal_attacks": sum(optimal_attack_list),
})

            total_exact += exact
            total_hamming += hamming
            total_gap += objective_gap
            total_instances += 1

            print(
                f"n={n}, m={m}, rep={rep} | "
                f"Opt={optimal_objective:.2f} | "
                f"Pred={predicted_objective:.2f} | "
                f"Gap={objective_gap:.4f} | "
                f"Hamming={hamming} | "
                f"Exact={exact}"
            )

    print("\nFINAL TEST RESULTS")
    print(f"Instances evaluated: {total_instances}")
    print(f"Exact match rate: {total_exact / total_instances:.4f}")
    print(f"Average Hamming distance: {total_hamming / total_instances:.4f}")
    print(f"Average objective gap: {total_gap / total_instances:.4f}")
    
    
    with open("evaluation_results.csv", "w", newline="") as csvfile:
        fieldnames = results_rows[0].keys()

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        writer.writeheader()
        writer.writerows(results_rows)

    print("\nSaved evaluation_results.csv")


if __name__ == "__main__":
    evaluate()