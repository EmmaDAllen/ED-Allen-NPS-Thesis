import argparse
import json

from data.generate_data import generate_dataset as generate_unfiltered
from data.generate_filtered_data import (
    generate_dataset as generate_filtered,
    generate_real_world_dataset,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    choices=[
        "mixed_55_15_reps100",
        "mixed_60_10_reps100",
        "mixed_55_15_filtered_reps100",
        "mixed_filtered",
    ],
)
parser.add_argument("output_file")
args = parser.parse_args()

settings = [
    (30, 60), (30, 90), (30, 120), (30, 180),
    (50, 100), (50, 150), (50, 200), (50, 300),
    (75, 150), (75, 225), (75, 300), (75, 450),
]

budgets = [1, 2, 3, 4, 5]

if args.dataset == "mixed_55_15_reps100":
    generate_unfiltered(
        settings, 100, budgets,
        problem_type="shortest_path",
        topology_mix={
            "one_in": 0.55, "grid": 0.15,
            "geometric": 0.10, "star_mesh": 0.10,
            "layered": 0.10,
        },
        base_seed=1,
        output_file=args.output_file,
    )

elif args.dataset == "mixed_60_10_reps100":
    generate_unfiltered(
        settings, 100, budgets,
        problem_type="shortest_path",
        topology_mix={
            "one_in": 0.60, "grid": 0.10,
            "geometric": 0.10, "star_mesh": 0.10,
            "layered": 0.10,
        },
        base_seed=1,
        output_file=args.output_file,
    )

elif args.dataset == "mixed_55_15_filtered_reps100":
    generate_filtered(
        settings, 100, budgets,
        problem_type="shortest_path",
        topology_mix={
            "one_in": 0.55, "grid": 0.15,
            "layered": 0.10, "geometric": 0.10,
            "star_mesh": 0.10,
        },
        base_seed=1,
        output_file=args.output_file,
        save_output=True,
    )

elif args.dataset == "mixed_filtered":
    synthetic = generate_filtered(
        settings, 85, budgets,
        problem_type="shortest_path",
        topology_mix={
            "one_in": 50 / 85, "grid": 15 / 85,
            "geometric": 10 / 85, "star_mesh": 10 / 85,
        },
        base_seed=1,
        save_output=False,
    )

    real_world = generate_real_world_dataset(
        attack_budgets=budgets,
        problem_type="shortest_path",
        instances_per_network=30,
        base_seed=1,
    )

    with open(args.output_file, "w") as f:
        json.dump(synthetic + real_world, f, indent=2)

print(f"Completed {args.dataset}: {args.output_file}")