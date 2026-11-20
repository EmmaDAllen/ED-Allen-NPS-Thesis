"""Generate real-world-only SPIP supervised training data.

Run from the thesis repository root:
    PYTHONPATH=. python -u data/generate_specialized_external_training.py

Uses ONLY REAL_WORLD_TRAINING_NETWORKS (six processed TNTP networks),
never the held-out evaluation networks. Saves independently of existing data.
Can resume after interruption, checkpointing each completed OD pair.
"""
import json
import os
import tempfile
import networkx as nx

from data.real_world_networks import (
    REAL_WORLD_TRAINING_NETWORKS,
    generate_real_world_network,
)
from optimization.mip import solve_instance

ATTACK_BUDGETS = [1, 2, 3, 4, 5]
INSTANCES_PER_NETWORK = 50
BASE_SEED = 20_000_000
MIN_HOPS = 3
OUTPUT_FILE = "data/training_data_shortest_path_external_only.json"


def save_json_atomic(dataset, path):
    """Replace the checkpoint only after JSON serialization has completed."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".external_training_", suffix=".json", dir=parent)    
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(dataset, f)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            dataset = json.load(f)
        print(f"Resuming checkpoint with {len(dataset)} solved samples", flush=True)
    else:
        dataset = []

    completed = {(x["graph_seed"], int(x["attack_budget"])) for x in dataset}
    skipped = 0
    network_names = list(REAL_WORLD_TRAINING_NETWORKS)

    for network_index, network_name in enumerate(network_names):
        print(f"\nNETWORK: {network_name}", flush=True)
        for rep in range(INSTANCES_PER_NETWORK):
            seed = BASE_SEED + 10_000_000 + network_index * 100_000 + rep
            if all((seed, k) in completed for k in ATTACK_BUDGETS):
                continue

            G, s, t, density = generate_real_world_network(
                network_name=network_name, min_hops=MIN_HOPS, seed=seed
            )
            hops = nx.shortest_path_length(G, source=s, target=t)
            connectivity = nx.edge_connectivity(G, s, t)

            for attack_budget in ATTACK_BUDGETS:
                if (seed, attack_budget) in completed:
                    continue

                sample = solve_instance(
                    G=G, s=s, t=t, density=density,
                    attack_limit=attack_budget,
                    problem_type="shortest_path", flow_demand=1,
                )
                if sample is None:
                    skipped += 1
                    print(f"SKIPPED network={network_name} rep={rep} K={attack_budget}", flush=True)
                    continue

                sample.update({
                    "graph_seed": seed,
                    "replication": rep,
                    "attack_budget": attack_budget,
                    "problem_type": "shortest_path",
                    "topology": "real_world",
                    "real_world_network": network_name,
                    "shortest_path_hops": hops,
                    "edge_connectivity": connectivity,
                    "generation_attempts": 1,
                })
                dataset.append(sample)
                completed.add((seed, attack_budget))
                save_json_atomic(dataset, OUTPUT_FILE)
                print(
                    f"Solved network={network_name} rep={rep} "
                    f"n={G.number_of_nodes()} m={G.number_of_edges()} "
                    f"K={attack_budget} objective={sample['path_length']:.3f} "
                    f"mip_solve_time={sample['mip_solve_time']:.4f}s "
                    f"total_saved={len(dataset)}",
                    flush=True,
                )

    expected = len(network_names) * INSTANCES_PER_NETWORK * len(ATTACK_BUDGETS)
    print(f"\nSaved {len(dataset)}/{expected} samples to {OUTPUT_FILE}; "
          f"failed_solves_this_run={skipped}", flush=True)


if __name__ == "__main__":
    main()
