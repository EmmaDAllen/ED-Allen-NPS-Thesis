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

    # largest interdiction budget used in the experiment for max-flow instances, this is used to 
    # establish the minimum required source-to-sink edge connectivity of an accepted training graph
    max_attack_budget = max(attack_budgets)


    # SHORTEST-PATH STRUCTURAL FILTERING SETTINGS

    # minimum unweighted number of arcs separating source and sink
    # this prevents structurally shallow networks such as s -> v -> t
    MIN_SHORTEST_PATH_HOPS = {30: 4, 50: 4, 75: 5}

    # loose upper bounds on source-to-sink edge connectivity
    # these prevent extremely redundant networks while still allowing
    # naturally low-connectivity shortest-path instances
    MAX_EDGE_CONNECTIVITY = {2.0: 3, 3.0: 4, 4.0: 6, 6.0: 8}

    # maximum number of candidate graphs tested before failing
    MAX_GENERATION_ATTEMPTS = 100000


    # NETWORK GENERATION LOOP

    # iterate through all network settings and replications; each accepted graph is then 
    # solved for every attack budget
    for n, m in network_settings:
        for rep in range(replications_per_setting):
            
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
                    G, s, t, density = generate_one_in_network(n=n,m=m,cost_low=COST_LOW,
                            cost_high=COST_HIGH,penalty_low=PENALTY_LOW,penalty_high=PENALTY_HIGH,
                            capacity_low=CAPACITY_LOW,capacity_high=CAPACITY_HIGH,seed=candidate_seed)

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


            # SHORTEST-PATH GRAPH GENERATION

            elif problem_type == "shortest_path":

                attempt = 0

                # density implied by requested n and m
                target_density = m / n

                # minimum structural source-sink separation
                min_hops = MIN_SHORTEST_PATH_HOPS[n]

                # match current density to nearest configured connectivity limit
                density_key = min(MAX_EDGE_CONNECTIVITY, key=lambda x: abs(x - target_density))

                max_edge_connectivity = MAX_EDGE_CONNECTIVITY[density_key]

                while True:

                    # stop rather than searching forever if the structural
                    # requirements are unrealistic for a particular setting
                    if attempt >= MAX_GENERATION_ATTEMPTS:

                        raise RuntimeError(
                            f"Could not generate valid shortest-path graph after "
                            f"{MAX_GENERATION_ATTEMPTS} attempts for "
                            f"n={n}, m={m}, density={target_density:.2f}. "
                            f"Requirements: min_hops={min_hops}, "
                            f"max_edge_connectivity={max_edge_connectivity}.")

                    # deterministic candidate seed preserves reproducibility
                    candidate_seed = seed * 1000000 + attempt


                    # generate candidate One-In network
                    G, s, t, density = generate_one_in_network(n=n, m=m, cost_low=COST_LOW,
                                        cost_high=COST_HIGH, penalty_low=PENALTY_LOW,
                                        penalty_high=PENALTY_HIGH, capacity_low=CAPACITY_LOW,
                                        capacity_high=CAPACITY_HIGH, seed=candidate_seed)


                    # STRUCTURAL SCREENING


                    # unweighted shortest-path length measures the minimum
                    # structural number of arcs separating source and sink
                    shortest_path_hops = nx.shortest_path_length(G, source=s, target=t)

                    # directed source-to-sink edge connectivity measures
                    # redundancy in possible routes between source and sink
                    edge_connectivity = nx.edge_connectivity(G, s, t)

                    # reject structurally shallow graphs
                    if shortest_path_hops < min_hops:
                        attempt += 1
                        continue


                    # reject only unusually high-redundancy graphs
                    # no minimum connectivity requirement is imposed
                    if edge_connectivity > max_edge_connectivity:
                        attempt += 1
                        continue

                    # candidate passed all structural checks
                    seed = candidate_seed

                    print(
                        f"Accepted shortest-path graph | "
                        f"n={n}, m={m}, density={density:.2f}, "
                        f"rep={rep}, attempts={attempt + 1}, "
                        f"hops={shortest_path_hops}, "
                        f"edge_connectivity={edge_connectivity}")

                    break


            else:

                # these problem types do not use the max-flow edge-connectivity filter;
                # generate one One-In network directly from the deterministic graph seed
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

                # save structural statistics for shortest-path graphs
                if problem_type == "shortest_path":

                    sample["shortest_path_hops"] = shortest_path_hops
                    sample["edge_connectivity"] = edge_connectivity
                    sample["generation_attempts"] = attempt + 1

                # flow demand is a problem-specific parameter and therefore is stored
                # only for minimum-cost-flow interdiction samples
                if problem_type == "min_cost_flow":
                    sample["flow_demand"] = flow_demand

                # store the completed sample to dataset
                dataset.append(sample)


                # PROGRESS OUTPUT

                # print instance-level progress while the dataset is being generated
                # so long-running generation jobs can be monitored in real time
                print(f"Solved n={n}, m={m}, budget={attack_budget}, "
                          f"density={density:.2f}, rep={rep}, "
                          f"flow_demand={flow_demand}, "
                          f"objective={sample[objective_name]:.2f}, "
                          f"mip_solve_time={sample['mip_solve_time']:.4f}")


    # DATASET SUMMARY AND OUTPUT

    # report the total number of successfully solved and skipped instances
    print(f"\nGenerated {len(dataset)} solved training samples.")
    print(f"Skipped {skipped} instances.")

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

    dataset = generate_dataset(
        network_settings=network_settings,
        replications_per_setting=10,
        attack_budgets=attack_budgets,
        problem_type=args.problem_type,
        base_seed=1,
        output_file=output_file)
