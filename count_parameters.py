from experiments.train import get_model

device = "cpu"
problem_type = "shortest_path"

model_types = [
    "tropical",
    "tropical_v2",
    "transformer",
    "edge_transformer",
    "gnn"
]

print("\nMODEL PARAMETER COUNTS")
print("-" * 65)

for model_type in model_types:

    model = get_model(
        model_type=model_type,
        problem_type=problem_type,
        device=device
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"{model_type:<20}"
        f"Total: {total_params:>10,}   "
        f"Trainable: {trainable_params:>10,}"
    )

print("-" * 65)
