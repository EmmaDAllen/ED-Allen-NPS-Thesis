# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:42:40 2026

@author: emmallen
"""

'''Training set generation for shortest path interdiction thesis.

1. Imports One-In random networks from random_test_networks.py
2. Solves each generated network using the MIP formulation
3. Saves the resulting training data to JSON'''


import json
import time
import argparse

from data.random_networks import generate_one_in_network 
#from random_networks import generate_grid_network
#from random_networks import generate_spatial_network
#from random_networks import generate_hub_spoke_network

from optimization.mip import solve_instance

COST_LOW = 1
COST_HIGH = 10
PENALTY_LOW = 1
PENALTY_HIGH = 10
CAPACITY_LOW = 1
CAPACITY_HIGH = 20

def generate_dataset(network_settings,replications_per_setting, attack_budgets, problem_type="shortest_path",
                     base_seed=1,output_file="training_data.json"):
    
    '''Generates dataset across multiple network sizes and densities.

    For each (n, m):
        - generate multiple random networks
        - solve each using MIP
        - store results

    Output:
        JSON file containing training samples'''
    
    # initialize dataset and skipped counter
    dataset = []
    skipped = 0

    # iterate through all combinations of network settings, attack budgets, and replications
    for attack_budget in attack_budgets:
        for n, m in network_settings:
            for rep in range(replications_per_setting):
            
                # unique seed per instance (ensures reproducibility)
                seed = base_seed + 100000 * n + 100 * m + rep

                # generate random test network (One-In method)
                G, s, t, density = generate_one_in_network(n=n, m=m,cost_low=COST_LOW, cost_high=COST_HIGH,
                                                           penalty_low=PENALTY_LOW, penalty_high=PENALTY_HIGH,
                                                           capacity_low=CAPACITY_LOW, capacity_high=CAPACITY_HIGH,
                                                           seed=seed)

                #G, s, t, density = generate_spatial_network(n=n,k=4,seed=seed)
                #G, s, t, density = generate_grid_network(rows=10,cols=10,seed=seed)
                #G, s, t, density = generate_hub_spoke_network(n=n,num_hubs=5,seed=seed)
            
                # solve interdiction problem for respective attack budget and store sample
                sample = solve_instance(G=G,s=s,t=t,density=density, attack_limit=attack_budget, 
                                        problem_type="shortest_path")

                if sample is None:
                    skipped += 1
                    continue

                sample["graph_seed"] = seed
                sample["replication"] = rep
                sample["attack_budget"] = attack_budget
                sample["problem_type"] = problem_type
                dataset.append(sample)

                    
                print(f"Solved n={n}, m={m}, budget={attack_budget}, "
                          f"density={density:.2f}, rep={rep}, "
                          f"objective={sample['path_length']:.2f}, "
                          f"mip_solve_time={sample['mip_solve_time']:.4f}")

    print(f"\nGenerated {len(dataset)} solved training samples.")
    print(f"Skipped {skipped} instances.")

    # save dataset
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem_type",
        choices=["shortest_path", "max_flow", "min_cost_flow"],
        default="shortest_path"
    )
    parser.add_argument(
        "--output_file",
        default=None
    )

    args = parser.parse_args()

    output_file = args.output_file
    if output_file is None:
        output_file = f"training_data_{args.problem_type}.json"

    # experiment design: (n, m) pairs - subject to change based on desired network sizes and densities
    network_settings = [
        (30, 75),
        (30, 120),
        (30, 180),
        (50, 125),
        (50, 200),
        (50, 300),
        (75, 188),
        (75, 300),
        (75, 450),]
    
    # experiment design: attack budgets to test - subject to change based on desired interdiction budgets
    attack_budgets = [1, 2, 3, 4, 5]

    dataset = generate_dataset(
        network_settings=network_settings,
        replications_per_setting=100,
        attack_budgets=attack_budgets,
        problem_type=args.problem_type,
        base_seed=1,
        output_file=output_file)
