"""Generate Wood-only supervised SPIP training data.

Run from repository root:
    PYTHONPATH=. python -u data/generate_specialized_wood_training.py
"""
import json
import os
from data.generate_wood_data import generate_wood_grid
from optimization.mip import solve_instance

GRID_SIZES = [(7, 7), (8, 8), (9, 9), (12, 12)]
PENALTY_MAX_VALUES = [5, 10]
ATTACK_BUDGETS = [5, 10]
REPS_PER_SETTING = 100
BASE_SEED = 20_000_000
OUTPUT_FILE = "data/training_data_shortest_path_wood_only.json"


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    dataset = []
    skipped = 0
    for size_index, (rows, cols) in enumerate(GRID_SIZES):
        for penalty_index, penalty_max in enumerate(PENALTY_MAX_VALUES):
            for rep in range(REPS_PER_SETTING):
                # Unique seeds across all configurations; disjoint from Wood evaluation seeds.
                seed = BASE_SEED + size_index * 100_000 + penalty_index * 10_000 + rep
                G, s, t, density = generate_wood_grid(
                    rows=rows, cols=cols, cost_max=10,
                    delay_max=penalty_max, resource_max=1, seed=seed,
                )
                for attack_budget in ATTACK_BUDGETS:
                    sample = solve_instance(
                        G=G, s=s, t=t, density=density,
                        attack_limit=attack_budget,
                        problem_type="shortest_path", flow_demand=1,
                    )
                    if sample is None:
                        skipped += 1
                        print(f"SKIPPED grid={rows}x{cols} penalty_max={penalty_max} K={attack_budget} seed={seed}", flush=True)
                        continue
                    sample.update({
                        "graph_seed": seed,
                        "replication": rep,
                        "attack_budget": attack_budget,
                        "problem_type": "shortest_path",
                        "topology": "wood",
                        "rows": rows,
                        "cols": cols,
                        "penalty_max": penalty_max,
                    })
                    dataset.append(sample)
                    print(f"Solved grid={rows}x{cols} penalty_max={penalty_max} K={attack_budget} seed={seed} objective={sample['path_length']:.3f}", flush=True)
                # Checkpoint after each graph; safe to inspect if interrupted.
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(dataset, f)
    expected = len(GRID_SIZES) * len(PENALTY_MAX_VALUES) * REPS_PER_SETTING * len(ATTACK_BUDGETS)
    print(f"Saved {len(dataset)}/{expected} samples to {OUTPUT_FILE}; skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()