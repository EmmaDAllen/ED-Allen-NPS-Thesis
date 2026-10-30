import pickle
import random
import copy
import os

# File paths
original_path = "evaluation_graphs/shortest_path_wood_graphs.pkl"
new_path = "evaluation_graphs/shortest_path_wood_exp_graphs.pkl"

# Do not accidentally overwrite an existing experimental dataset
if os.path.exists(new_path):
    raise FileExistsError(
        f"{new_path} already exists. "
        "Remove or rename it explicitly if you intend to regenerate.")

# Load original Wood evaluation graphs
with open(original_path, "rb") as f:
    original_graphs = pickle.load(f)

new_graphs = []

# Create new graphs with resampled penalties
for graph_data in original_graphs:

    # Deep copy preserves original graph and metadata
    new_data = copy.deepcopy(graph_data)
    G = new_data["G"]

    # Deterministic independent random generator for each graph
    rng = random.Random(10_000_000 + int(new_data["seed"]))

    # Change only interdiction penalties
    for u, v, attrs in G.edges(data=True):

        # Preserve noninterdictable source/sink arcs
        if not attrs.get("interdictable", True):
            continue

        # Resample penalties uniformly from 1 to 10
        attrs["penalty"] = rng.randint(1, 10)

    # Update metadata to reflect new penalty maximum
    new_data["delay_max"] = 10

    new_graphs.append(new_data)

# Verify that only eligible penalties and metadata changed
assert len(original_graphs) == len(new_graphs)

num_changed_penalties = 0

for old, new in zip(original_graphs, new_graphs):

    assert old["seed"] == new["seed"]
    assert old["attack_budget"] == new["attack_budget"]

    G_old = old["G"]
    G_new = new["G"]

    # Verify node and edge sets
    assert set(G_old.nodes()) == set(G_new.nodes())
    assert set(G_old.edges()) == set(G_new.edges())

    for u, v in G_old.edges():

        old_attrs = G_old[u][v]
        new_attrs = G_new[u][v]

        # All attributes except penalty must be unchanged
        for key in old_attrs:
            if key != "penalty":
                assert old_attrs[key] == new_attrs[key], (
                    f"Attribute {key} changed on edge {(u, v)}"
                )

        if old_attrs.get("interdictable", True):

            assert 1 <= new_attrs["penalty"] <= 10

            if old_attrs["penalty"] != new_attrs["penalty"]:
                num_changed_penalties += 1

        else:
            # Noninterdictable penalties must remain unchanged
            assert old_attrs["penalty"] == new_attrs["penalty"]

# Save the verified new dataset
with open(new_path, "wb") as f:
    pickle.dump(new_graphs, f)


print("WOOD PENALTY SENSITIVITY DATASET")
print(f"Original file: {original_path}")
print(f"New file:      {new_path}")
print(f"Graphs copied: {len(new_graphs)}")
print(f"Penalties changed: {num_changed_penalties}")
print("Original penalty range: 1-5")
print("New penalty range:      1-10")
print("Verification passed.")
print("Topology, costs, eligibility, and budgets unchanged.")