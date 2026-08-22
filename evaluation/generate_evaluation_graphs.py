
"""generate_evaluation_graphs.py

Generate and save fixed graph sets for network interdiction model evaluation.

The script creates evaluation graphs separately from evaluate.py so that every
trained model is evaluated on exactly the same graph instances. Generated graphs
are saved incrementally, allowing interrupted generation jobs to resume without
regenerating previously accepted graphs.

Usage:
    PYTHONPATH=. python -u evaluation/generate_evaluation_graphs.py PROBLEM_TYPE EVAL_MODE"""

import os
import sys
import pickle
import networkx as nx

from data.random_networks import generate_one_in_network
from data.generate_wood_data import generate_wood_grid
from data.generate_external_data import load_external_network


def get_test_settings(eval_mode):


    """Return the network size and density settings associated with an evaluation mode.

    id_new:
        Uses the same network sizes and densities represented in training, but
        evaluation graphs are generated using new random seeds.

    ood_size:
        Uses substantially larger networks than those represented in training
        to evaluate model generalization to unseen graph sizes."""



    if eval_mode == "id_new":

        return [(30, 75),
            (30, 120),
            (50, 125),
            (50, 200),
            (75, 188),
            (75, 300),]

    elif eval_mode == "ood_size":

        return [
            # 100 nodes
            (100, 200),
            (100, 250),
            (100, 300),
            (100, 400),
            (100, 500),
            (100, 600),
            (100, 800),

            # 200 nodes
            (200, 400),
            (200, 500),
            (200, 600),
            (200, 800),
            (200, 1000),
            (200, 1200),
            (200, 1600),

            # 400 nodes
            (400, 800),
            (400, 1000),
            (400, 1200),
            (400, 1600),
            (400, 2000),
            (400, 2400),
            (400, 3200),

            # 600 nodes
            (600, 1200),
            (600, 1500),
            (600, 1800),
            (600, 2400),
            (600, 3000),
            (600, 3600),
            (600, 4800),]

    elif eval_mode == "wood":

        return [
            (13, 7, 7, 10, 5, 1, 5),
            (14, 7, 7, 10, 5, 1, 10),
            (15, 8, 8, 10, 5, 1, 5),
            (16, 8, 8, 10, 5, 1, 10),
            (17, 9, 9, 10, 5, 1, 5),
            (18, 9, 9, 10, 5, 1, 10),
            (19, 12, 12, 10, 5, 1, 5),
            (20, 12, 12, 10, 5, 1, 10),]

    elif eval_mode == "external":
        return None

    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")



def generate_evaluation_graphs(problem_type, eval_mode):

    test_settings = get_test_settings(eval_mode)

    os.makedirs("evaluation_graphs", exist_ok=True)

    output_path = (f"evaluation_graphs/"
                   f"{problem_type}_{eval_mode}_graphs.pkl")
        
    # WOOD BENCHMARKS
    
    if eval_mode == "wood":
    
        if problem_type != "shortest_path":
                raise ValueError("Wood evaluation graphs are for shortest_path only.")
    
        base_seed = 5
        evaluation_graphs = []
    
        for (problem,rows,cols,cost_max,delay_max,resource_max,resource_budget,) in test_settings:
    

            seed = base_seed + problem
    

            G, s, t, density = generate_wood_grid(rows=rows,cols=cols,cost_max=cost_max,delay_max=delay_max,
                                                      resource_max=resource_max,seed=seed,)
    

            evaluation_graphs.append({"G": G, "s": s, "t": t,"density": density,"seed": seed,
    
                        # Useful benchmark metadata
                        "wood_problem": problem, "rows": rows, "cols": cols,
    
                        # Actual graph size
                        "n": G.number_of_nodes(), "m": G.number_of_edges(),
    
                        # Wood parameters
                        "cost_max": cost_max,"delay_max": delay_max,"resource_max": resource_max,
                        "attack_budget": resource_budget,})
    
            print(f"Generated Wood problem {problem} | "
                    f"{rows}x{cols} | "
                    f"nodes={G.number_of_nodes()} | "
                    f"arcs={G.number_of_edges()} | "
                    f"budget={resource_budget}",
                    flush=True,)
    
        with open(output_path, "wb") as f:
                pickle.dump(evaluation_graphs, f)
    
        print(f"\nFinished. Saved {len(evaluation_graphs)} "
                f"Wood graphs to {output_path}",
                flush=True,)
    
        return

    # EXTERNAL NETWORK

    if eval_mode == "external":

        if problem_type != "shortest_path":
            raise ValueError("External evaluation currently supports shortest_path only.")

        G, s, t, density = load_external_network(node_path="data/external/node_data.csv",
            arc_path="data/external/arc_data.csv",source=YOUR_SOURCE,sink=YOUR_SINK,
            penalty_seed=5,)

        evaluation_graphs = [{"G": G, "s": s, "t": t, "density": density,
                              "n": G.number_of_nodes(), "m": G.number_of_edges(),
                              "network_name": "external_transportation_network", "seed": 5,}]

        with open(output_path, "wb") as f:
            pickle.dump(evaluation_graphs,f,)

        print(f"\nFinished. Saved external network to "
            f"{output_path}",flush=True,)

        return


    reps_per_setting = 20
    test_attack_limits = [1, 2, 3, 4, 5]
    max_attack_budget = max(test_attack_limits)

    # Same evaluation seed currently used by evaluate.py
    base_seed = 5

    total_expected = len(test_settings) * reps_per_setting


    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            evaluation_graphs = pickle.load(f)

        print(f"Resuming from {len(evaluation_graphs)} previously generated graphs.", flush=True,)

    else:
        evaluation_graphs = []

    existing_keys = {(g["n"], g["m"], g["rep"]) for g in evaluation_graphs}

    for n, m in test_settings:

        for rep in range(reps_per_setting):

            # Skip graphs that were already successfully generated and saved
            if (n, m, rep) in existing_keys:
                continue

            seed = base_seed + 100000 * n + 100 * m + rep

            if problem_type == "max_flow":

                attempt = 0

                while True:

                    candidate_seed = seed * 1000000 + attempt

                    G, s, t, density = generate_one_in_network(n=n,m=m,cost_low=1,cost_high=10,penalty_low=1,
                                                               penalty_high=10,capacity_low=1,capacity_high=20,
                                                               seed=candidate_seed,)

                    edge_connectivity = nx.edge_connectivity(G, s, t)

                    if edge_connectivity > max_attack_budget:

                        seed = candidate_seed

                        print(f"Accepted graph | "
                            f"n={n}, m={m}, rep={rep} | "
                            f"edge_connectivity={edge_connectivity} | "
                            f"attempts={attempt + 1}",
                            flush=True,)

                        break

                    attempt += 1

                    if attempt % 10 == 0:
                        print(f"Searching | "
                            f"n={n}, m={m}, rep={rep} | "
                            f"attempt={attempt}",
                            flush=True,)

            elif problem_type in ("shortest_path", "min_cost_flow"):

                G, s, t, density = generate_one_in_network(n=n,m=m,cost_low=1,cost_high=10,penalty_low=1,
                                                           penalty_high=10,capacity_low=1,capacity_high=20,
                                                           seed=seed,)

            else:
                raise ValueError(f"Unknown problem_type: {problem_type}")
            

            evaluation_graphs.append({"G": G,"s": s,"t": t,"density": density,"seed": seed,
                                      "n": n,"m": m,"rep": rep,})

            existing_keys.add((n, m, rep))

            with open(output_path, "wb") as f:
                pickle.dump(evaluation_graphs, f)

            print(f"\nGenerated and saved "
                  f"{len(evaluation_graphs)}/{total_expected} evaluation graphs.",flush=True,)

    print(
        f"\nFinsihed. Saved {len(evaluation_graphs)} graphs to {output_path}",flush=True,)


if __name__ == "__main__":

    problem_type = sys.argv[1] if len(sys.argv) > 1 else "max_flow"
    eval_mode = sys.argv[2] if len(sys.argv) > 2 else "ood_size"

    generate_evaluation_graphs(problem_type, eval_mode)