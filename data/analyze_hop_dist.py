import pandas as pd
import networkx as nx
from data.random_networks import generate_one_in_network

records = []

network_settings = [
    (30, 60), (30, 90), (30, 120), (30, 180),
    (50, 100), (50, 150), (50, 200), (50, 300),
    (75, 150), (75, 225), (75, 300), (75, 450),
]

COST_LOW = 1
COST_HIGH = 10
PENALTY_LOW = 1
PENALTY_HIGH = 10
CAPACITY_LOW = 1
CAPACITY_HIGH = 20

N_TRIALS = 1000

for n, m in network_settings:

    for rep in range(N_TRIALS):

        seed = 999_000_000 + 100000 * n + 100 * m + rep

        G, s, t, density = generate_one_in_network(
            n=n,
            m=m,
            cost_low=COST_LOW,
            cost_high=COST_HIGH,
            penalty_low=PENALTY_LOW,
            penalty_high=PENALTY_HIGH,
            capacity_low=CAPACITY_LOW,
            capacity_high=CAPACITY_HIGH,
            seed=seed
        )

        hops = nx.shortest_path_length(
            G,
            source=s,
            target=t
        )

        records.append({
            "n": n,
            "m": m,
            "density": density,
            "shortest_path_hops": hops
        })

hop_trials = pd.DataFrame(records)


hop_summary = (
    hop_trials
    .groupby(["n", "m", "density"])["shortest_path_hops"]
    .agg(
        mean="mean",
        median="median",
        max="max",
        q90=lambda x: x.quantile(0.90),
        q95=lambda x: x.quantile(0.95),
        q99=lambda x: x.quantile(0.99),
    )
)

print(hop_summary)


thresholds = range(3, 11)

records = []

for (n, m, density), group in hop_trials.groupby(["n", "m", "density"]):

    for threshold in thresholds:

        acceptance_rate = (
            group["shortest_path_hops"] >= threshold
        ).mean()

        expected_attempts = (
            1 / acceptance_rate
            if acceptance_rate > 0
            else float("inf")
        )

        records.append({
            "n": n,
            "m": m,
            "density": density,
            "min_hops": threshold,
            "acceptance_rate": acceptance_rate,
            "expected_attempts": expected_attempts
        })

acceptance_df = pd.DataFrame(records)


print(
    acceptance_df[
        acceptance_df["acceptance_rate"] > 0
    ].round(4).to_string(index=False)
)

acceptance_table = acceptance_df.pivot_table(
    index=["n", "m", "density"],
    columns="min_hops",
    values="acceptance_rate"
)

print("\nAcceptance Rate by Minimum Hop Threshold:")
print(acceptance_table.round(3).to_string())