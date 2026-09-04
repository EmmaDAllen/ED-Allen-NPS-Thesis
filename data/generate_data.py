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
import math

from data.random_networks import (generate_one_in_network,generate_grid_network,
    generate_layered_network, generate_star_mesh_network)
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


# TOPOLOGY HELPERS

def generate_graph_by_topology(topology,n, m, cost_low, cost_high, penalty_low, penalty_high,
                                capacity_low, capacity_high, seed,):

    """Generate one network using the requested topology.

    Every topology generator must return:
        G, s, t, density

    This common interface allows the rest of the training-data generation
    pipeline to remain independent of graph topology."""

    if topology == "one_in":

        return generate_one_in_network(n=n, m=m,cost_low=cost_low,cost_high=cost_high,penalty_low=penalty_low,
                                       penalty_high=penalty_high,capacity_low=capacity_low,capacity_high=capacity_high,
                                       seed=seed)

    elif topology == "grid":

        return generate_grid_network(n=n,m=m,cost_low=cost_low,cost_high=cost_high,penalty_low=penalty_low,
                                     penalty_high=penalty_high,capacity_low=capacity_low,capacity_high=capacity_high,
                                     seed=seed)

    elif topology == "layered":

        return generate_layered_network(n=n,m=m,cost_low=cost_low,cost_high=cost_high,penalty_low=penalty_low,
                                        penalty_high=penalty_high,capacity_low=capacity_low,capacity_high=capacity_high,
                                        seed=seed)


    elif topology == "star_mesh":

        return generate_star_mesh_network(n=n,m=m,cost_low=cost_low,cost_high=cost_high,penalty_low=penalty_low,
                                          penalty_high=penalty_high,capacity_low=capacity_low,capacity_high=capacity_high,
                                          seed=seed)

    else:
        raise ValueError(f"Unknown topology: {topology}")



def build_topology_schedule(replications_per_setting, topology_mix):

    """Create a deterministic topology assignment for the replications
    associated with each (n, m) network setting.

    Example for 50 replications and:
        {"one_in": 0.70, "grid": 0.15, "layered": 0.15}

    The total number of graphs remains exactly equal to
    replications_per_setting."""

    # make sure topology proportions sum to 1
    total_weight = sum(topology_mix.values())

    if not math.isclose(total_weight, 1.0, rel_tol=1e-9):
        raise ValueError(f"Topology proportions must sum to 1.0, "
            f"but received {total_weight:.4f}.")

    topology_counts = {}
    fractional_parts = []

    # first assign the floor of the requested number of replications
    for topology, weight in topology_mix.items():

        exact_count = replications_per_setting * weight
        base_count = int(math.floor(exact_count))

        topology_counts[topology] = base_count

        fractional_parts.append((exact_count - base_count, topology))

    # distribute any remaining replications according to the largest fractional remainders
    assigned = sum(topology_counts.values())
    remaining = replications_per_setting - assigned

    fractional_parts.sort(reverse=True)

    for _, topology in fractional_parts[:remaining]:
        topology_counts[topology] += 1

    # construct the deterministic replication schedule
    topology_schedule = []

    for topology in topology_mix.keys():

        topology_schedule.extend([topology] * topology_counts[topology])

    if len(topology_schedule) != replications_per_setting:
        raise RuntimeError("Topology schedule does not contain the expected number of replications.")

    print("\nTopology schedule per (n, m) setting:")

    for topology, count in topology_counts.items():
        print(f"  {topology}: {count}")

    return topology_schedule




def generate_dataset(network_settings,replications_per_setting, attack_budgets, problem_type="shortest_path",
                     topology_mix=None, base_seed=1,output_file="training_data.json"):
    
    """Generate a training dataset across multiple network sizes,
    densities, topologies, and interdiction budgets.

    For each (n, m):
        - generate multiple random networks
        - assign each network a topology according to topology_mix
        - solve each graph for every attack budget
        - store the resulting optimal interdiction sample

    Output:
        JSON file containing training samples."""

    # default behavior preserves the original One-In-only experiment
    if topology_mix is None:
        topology_mix = {"one_in": 1.0}

    # construct a deterministic topology assignment for the replications
    topology_schedule = build_topology_schedule(
        replications_per_setting=replications_per_setting,
        topology_mix=topology_mix)
    
    # Determine the objective name based on the problem type
    objective_name = OBJECTIVE_NAMES[problem_type]
    
    # initialize dataset - store solved samples
    dataset = []
    # counts number of instances skipped due to infeasibility or solver issues
    skipped = 0

    # largest interdiction budget used in the experiment for max-flow instances, this is used to 
    # establish the minimum required source-to-sink edge connectivity of an accepted training graph
    max_attack_budget = max(attack_budgets)

    # NETWORK GENERATION LOOP

    # iterate through all network settings and replications; each accepted graph is then 
    # solved for every attack budget
    for n, m in network_settings:
        
        for rep in range(replications_per_setting):

            # select the topology assigned to this replication
            topology = topology_schedule[rep]
            
            # construct a deterministic base seed from the graph dimensions and
            # replication number so the experimental instances are reproducible
            seed = base_seed + 100000 * n + 100 * m + rep


            # MAX-FLOW GRAPH GENERATION

            # max-flow interdiction requires additional filtering because sparse One-In graphs may
            # contain small source-to-sink cuts. If the minimum s-t cut contains no more edges than 
            # the interdiction budget, the attacker can remove the entire cut and reduce the surviving
            # maximum flow to zero. To avoid a training dataset dominated by these trivial zero-flow cases,
            # repeatedly generate candidate graphs until the s-t edge connectivity is strictly greater than
            #  the largest attack budget used in the experiment.
            if problem_type == "max_flow":

                # count candidate graphs tested for the current experimental replication
                attempt = 0

                while True:

                # derive a unique deterministic seed for each candidate graph while
                # preserving reproducibility of the rejection-sampling procedure
                    candidate_seed = seed * 1000000 + attempt

                    # generate a candidate One-In directed network
                    G, s, t, density = generate_graph_by_topology(topology=topology,n=n, m=m,cost_low=COST_LOW, 
                                    cost_high=COST_HIGH,penalty_low=PENALTY_LOW, penalty_high=PENALTY_HIGH,
                                    capacity_low=CAPACITY_LOW, capacity_high=CAPACITY_HIGH,seed=seed)


                    # compute the minimum number of directed edges whose removal would
                    # disconnect the source from the sink
                    edge_connectivity = nx.edge_connectivity(G,s,t)
                
                    # accept the graph only when every tested interdiction budget is smaller than its s-t
                    # edge connectivity; therefore an attacker using K <= max_attack_budget cannot disconnect
                    # s from t solely by removing K edges
                    if edge_connectivity > max_attack_budget:

                        # save the actual accepted candidate seed rather than the original
                        # base seed so this exact graph can be regenerated later
                        seed = candidate_seed
                        break

                    # reject the current candidate and generate another deterministic graph
                    attempt += 1


            # SHORTEST-PATH AND MIN-COST-FLOW GRAPH GENERATION
            # shortest-path and min-cost-flow interdiction problems do not require additional filtering
            else:

                # these problem types do not use the max-flow edge-connectivity filter;
                # generate one One-In network directly from the deterministic graph seed
                G, s, t, density = generate_graph_by_topology(topology=topology,n=n, m=m,cost_low=COST_LOW, 
                                    cost_high=COST_HIGH,penalty_low=PENALTY_LOW, penalty_high=PENALTY_HIGH,
                                    capacity_low=CAPACITY_LOW, capacity_high=CAPACITY_HIGH,seed=seed)
                
            # min cost flow problem requires a feasible flow demand to be specified 
            # rather than a fixed value, we compute a flow demand based on the maximum flow of the network and 
            # require only  haldf of that amount so every instance is feasible = design choice to ensure that the
            # generated instances are solvable and provide meaningful training data for the model.
            if problem_type == "min_cost_flow":
                baseline_max_flow = nx.maximum_flow_value(G,s,t,capacity="capacity")
                flow_demand = max(1, int(0.5 * baseline_max_flow))

            else:
                flow_demand = 1



            # INTERDICTION BUDGET LOOP

            # solve the same accepted graph separately for every requested attack budget this produces training
            # examples for K = 1, ..., max_attack_budget while holding the underlying graph realization constant
            for attack_budget in attack_budgets:
            
                # solve generated network interdiction problem for respective attack budget and store sample
                # using respective MIP formulation
                sample = solve_instance(G=G,s=s,t=t,density=density, attack_limit=attack_budget, 
                                        problem_type=problem_type,flow_demand=flow_demand)

                # skip instances that could not be solved to optimality
                if sample is None:
                    skipped += 1
                    continue


                # EXPERIMENT METADATA

                # record experiment metadata for later analysis and reproducibility
                sample["graph_seed"] = seed
                sample["replication"] = rep
                sample["attack_budget"] = attack_budget
                sample["problem_type"] = problem_type

                # save topology for later analysis
                sample["topology"] = topology

                # flow demand is a problem-specific parameter and therefore is stored
                # only for minimum-cost-flow interdiction samples
                if problem_type == "min_cost_flow":
                    sample["flow_demand"] = flow_demand

                # store the completed sample to dataset
                dataset.append(sample)


                # PROGRESS OUTPUT

                # print instance-level progress while the dataset is being generated
                # so long-running generation jobs can be monitored in real time
                print(f"Solved topology={topology}, n={n}, m={m}, budget={attack_budget}, "
                          f"density={density:.2f}, rep={rep}, "
                          f"flow_demand={flow_demand}, "
                          f"objective={sample[objective_name]:.2f}, "
                          f"mip_solve_time={sample['mip_solve_time']:.4f}")


    # DATASET SUMMARY AND OUTPUT

    # report the total number of successfully solved and skipped instances
    print(f"\nGenerated {len(dataset)} solved training samples.")
    print(f"Skipped {skipped} instances.")


    # report topology composition of the completed dataset
    print("\nCompleted samples by topology:")

    for topology in topology_mix:

        count = sum(sample["topology"] == topology for sample in dataset)

        print(f" {topology}: {count}")


    # save the complete training dataset as formatted JSON
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
        (30, 60),
        (30, 90),
        (30, 120),
        (30, 180),

        (50, 100),
        (50, 150),
        (50, 200),
        (50, 300),

        (75, 150),
        (75, 225),
        (75, 300),
        (75, 450),]
    
    # experiment design: attack budgets to test - subject to change based on desired interdiction budgets
    attack_budgets = [1, 2, 3, 4, 5]



    # TOPOLOGY EXPERIMENT

    # EXPERIMENT 1: Clean One-In control

    # topology_mix = {"one_in": 1.0}


    # EXPERIMENT 2: Topology-augmented training
    # COMMENT OUT the One-In-only version above and uncomment this version when generating the
    # topology-augmented dataset

    topology_mix = {"one_in": 0.60, "grid": 0.10, "geometric": 0.10, "star_mesh": 0.10, "layered": 0.10}


    dataset = generate_dataset(
        network_settings=network_settings,
        replications_per_setting=50,
        attack_budgets=attack_budgets,
        problem_type=args.problem_type,
        topology_mix=topology_mix,
        base_seed=1,
        output_file=output_file)



