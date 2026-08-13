# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:42:40 2026

@author: emmallen
"""

"""Generate supervised training datasets for network interdiction problems.

1. Generate random directed networks using the One-In topology generator.
2. Solve each network optimally using the corresponding MIP formulation.
3. Store the optimal interdiction solution and graph features.
4. Export the complete dataset to a JSON file for model training."""


import json
import time
import argparse
import networkx as nx

from data.random_networks import generate_one_in_network 
from optimization.mip import solve_instance

# Constants for generating random networks
# edge traversal costs
COST_LOW = 1
COST_HIGH = 10
# interdiction penalties
PENALTY_LOW = 1
PENALTY_HIGH = 10
# edge capacities
CAPACITY_LOW = 1
CAPACITY_HIGH = 20

# Mapping of problem types to their corresponding objective names in the training samples
# This dictionary is used to extract the correct objective value from the training sample based on the problem type.
OBJECTIVE_NAMES = {
    "shortest_path": "path_length",
    "max_flow": "max_flow",
    "min_cost_flow": "min_cost_flow"}


def generate_dataset(network_settings,replications_per_setting, attack_budgets, problem_type="shortest_path",
                     base_seed=1,output_file="training_data.json"):
    
    '''Generates dataset across multiple network sizes and densities.

    For each (n, m):
        - generate multiple random networks
        - solve each using MIP
        - store results

    Output:
        JSON file containing training samples'''
    
    # Determine the objective name based on the problem type
    objective_name = OBJECTIVE_NAMES[problem_type]
    
    # initialize dataset - store solved samples
    dataset = []
    # counts number of instances skipped due to infeasibility or solver issues
    skipped = 0


    MIN_EDGE_CONNECTIVITY = 3
    MAX_GENERATION_ATTEMPTS = 1000

    # iterate through all combinations of network settings, attack budgets, and replications
    # enumerates every experiemntal configuration to generate a diverse dataset
    for attack_budget in attack_budgets:
        for n, m in network_settings:
            for rep in range(replications_per_setting):
            
                # unique deterministic seed per instance (ensures reproducibility)
                seed = base_seed + 100000 * n + 100 * m + rep


                if problem_type == "max_flow":

                    graph_found = False

                    for attempt in range(MAX_GENERATION_ATTEMPTS):

                        candidate_seed = seed * MAX_GENERATION_ATTEMPTS + attempt

                        G, s, t, density = generate_one_in_network(n=n,m=m,cost_low=COST_LOW,
                            cost_high=COST_HIGH,penalty_low=PENALTY_LOW,penalty_high=PENALTY_HIGH,
                            capacity_low=CAPACITY_LOW,capacity_high=CAPACITY_HIGH,seed=candidate_seed)

                        edge_connectivity = nx.edge_connectivity(G,s,t)

                        if edge_connectivity >= MIN_EDGE_CONNECTIVITY:
                            graph_found = True
                            seed = candidate_seed
                            break

                    if not graph_found:
                        skipped += 1
                        continue

                else:

                    # generate random test network (One-In method)
                    G, s, t, density = generate_one_in_network(n=n, m=m,cost_low=COST_LOW, cost_high=COST_HIGH,
                                                           penalty_low=PENALTY_LOW, penalty_high=PENALTY_HIGH,
                                                           capacity_low=CAPACITY_LOW, capacity_high=CAPACITY_HIGH,
                                                           seed=seed)
                
                # min cost flow problem requires a feasible flow demand to be specified 
                # rather than a fixed value, we compute a flow demand based on the maximum flow of the network and 
                # require only  haldf of that amount so every instance is feasible = design choice to ensure that the
                # generated instances are solvable and provide meaningful training data for the model.
                if problem_type == "min_cost_flow":
                    baseline_max_flow = nx.maximum_flow_value(G,s,t,capacity="capacity")
                    flow_demand = max(1, int(0.5 * baseline_max_flow))
                else:
                    flow_demand = 1
            
                # solve generated network interdiction problem for respective attack budget and store sample
                # using respective MIP formulation
                sample = solve_instance(G=G,s=s,t=t,density=density, attack_limit=attack_budget, 
                                        problem_type=problem_type,flow_demand=flow_demand)

                # skip instances that could not be solved to optimality
                if sample is None:
                    skipped += 1
                    continue

                # record experiment metadata for later analysis and reproducibility
                sample["graph_seed"] = seed
                sample["replication"] = rep
                sample["attack_budget"] = attack_budget
                sample["problem_type"] = problem_type

                # flow demand only applies to min cost flow problems
                if problem_type == "min_cost_flow":
                    sample["flow_demand"] = flow_demand

                # store the completed sample
                dataset.append(sample)

                # Print progress information while the dataset is being generated
                # to monitor real time generation
                print(f"Solved n={n}, m={m}, budget={attack_budget}, "
                          f"density={density:.2f}, rep={rep}, "
                          f"flow_demand={flow_demand}, "
                          f"objective={sample[objective_name]:.2f}, "
                          f"mip_solve_time={sample['mip_solve_time']:.4f}")

    # Report overall dataset statistics.
    print(f"\nGenerated {len(dataset)} solved training samples.")
    print(f"Skipped {skipped} instances.")

    # save dataset
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    return dataset


if __name__ == "__main__":

    # command line arguments
    parser = argparse.ArgumentParser()

    # select which interdiction problem to generate training data for
    parser.add_argument("problem_type",
        choices=["shortest_path", "max_flow", "min_cost_flow"],
        default="shortest_path")

    # optional output filename - default uses command line arguments
    parser.add_argument("output_file",nargs="?",default=None,help="Optional output filename.")

    # parse command line arguments
    args = parser.parse_args()

    # create respective descriptive filename
    output_file = args.output_file
    if output_file is None:
        output_file = f"training_data_{args.problem_type}.json"

    # experiment design: (n, m) pairs (nodes, edges) subject to change based on desired network sizes and densities
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
        replications_per_setting=50,
        attack_budgets=attack_budgets,
        problem_type=args.problem_type,
        base_seed=1,
        output_file=output_file)
