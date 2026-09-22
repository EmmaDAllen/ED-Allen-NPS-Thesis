
"""generate_evaluation_graphs.py

Generate and save fixed graph sets for network interdiction model evaluation.

The script creates evaluation graphs separately from evaluate.py so that all
trained models are evaluated on the same network instances. Graph collections
are saved as pickle files and later loaded by the evaluation and visualization
scripts.

The script supports four evaluation modes:

1. id_new
   Generates new One-In networks using graph sizes and densities represented
   during training.

2. ood_size
   Generates larger One-In networks to evaluate generalization to network
   sizes outside the training distribution.

3. value_ood
   Generates One-In networks using the same sizes and densities represented
   during training, but with arc costs and interdiction penalties outside
   the training range.

4. wood
   Generates selected Wood-style shortest-path interdiction benchmark
   networks using their specified grid dimensions, arc-attribute ranges,
   and interdiction budgets.

5. external
   Loads a fixed directed rail/road transportation network from node and arc
   CSV files and prepares all specified supply-to-demand OD pairs for
   shortest-path interdiction evaluation.

For synthetic One-In networks, graph generation is saved incrementally so an
interrupted generation job can resume without regenerating completed instances.
Maximum-flow networks are additionally screened for sufficient source-to-sink
edge connectivity to avoid trivial complete-disconnection cases.

Usage:
    PYTHONPATH=. python -u evaluation/generate_evaluation_graphs.py PROBLEM_TYPE EVAL_MODE"""

import os
import sys
import pickle
import networkx as nx
# import generators used for one-in, benchmark and real-world external evaluation networks
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
        to evaluate model generalization to unseen graph sizes.
        
    wood:
        Uses selected Wood shortest-path interdiction benchmark configurations,
        including each problem's grid dimensions and interdiction parameters.

    external:
        Does not require synthetic graph settings because the graph is loaded
        directly from external node and arc CSV files."""



    if eval_mode == "id_new":

        return [
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

    elif eval_mode == "value_ood":

        # Same network sizes and densities as ID evaluation.
        # Only the arc cost and interdiction-penalty distributionsare shifted outside the training range.
        return [
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

        # selected Wood shortest-path interdiction benchmark configurations
        # tuple format: (problem number, rows, columns, maximum cost, maximum delay,
        # maximum resource requirement, interdiction resource budget)
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

        # external networks are loaded directly from CSV files rather than
        # generated from a predefined (n, m) experimental setting
        return None

    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")




def generate_evaluation_graphs(problem_type, eval_mode):


    """Generate and save a fixed graph set for the selected evaluation mode.

    Depending on the evaluation mode, this function either generates synthetic One-In networks,
    generates selected Wood-style benchmark networks, or loads and prepares an external network 
    from CSV files.

    Synthetic ID and OOD graph sets contain multiple replications for each specified network size 
    and density. Previously generated graphs are loaded when available so interrupted generation 
    can resume without duplicating completed instances. Maximum-flow graphs are additionally screened
    for sufficient source-to-sink edge connectivity.

    Wood benchmark networks are generated using their specified grid dimensions, arc-attribute ranges, 
    and benchmark-specific interdiction budgets. External networks are loaded using user-specified 
    source and sink node identifiers.

    All completed graph sets are saved as pickle files for consistent reuse by
    the model evaluation and visualization scripts.

    Parameters
    problem_type : str
        Network interdiction problem to generate evaluation graphs for:
        "shortest_path", "max_flow", or "min_cost_flow".

    eval_mode : str
        Evaluation graph type: "id_new", "ood_size", "value_ood", "wood", or "external".

    source : int, optional
        Original source node identifier for external-network evaluation.
        Required when eval_mode="external".

    sink : int, optional
        Original sink node identifier for external-network evaluation.
        Required when eval_mode="external"."""


    # retrieve the graph-generation settings associated with the selected
    # evaluation mode; external evaluation does not require test settings
    test_settings = get_test_settings(eval_mode)

    # create the directory used to store fixed evaluation graph collections
    os.makedirs("evaluation_graphs", exist_ok=True)

    # construct a problem- and evaluation-specific filename so each graph set
    # is stored separately and can later be loaded by evaluate.py
    output_path = (f"evaluation_graphs/"
                   f"{problem_type}_{eval_mode}_graphs.pkl")

        
    # WOOD BENCHMARKS
    
    # generate the selected Wood-style benchmark instances separately from the
    # standard synthetic evaluation graphs because each benchmark specifies its
    # own grid dimensions, arc-attribute ranges, and interdiction budget
    if eval_mode == "wood":

        # Wood benchmark evaluation is currently implemented only for
        # shortest-path interdiction
        if problem_type != "shortest_path":
            raise ValueError("Wood evaluation graphs are for shortest_path only.")

        # Number of independent random graph realizations generated for each
        # Wood benchmark configuration
        reps_per_wood_setting = 30

        # fixed base seed makes the generated benchmark instances reproducible
        base_seed = 5

        # store all generated Wood benchmark graphs in one evaluation file
        evaluation_graphs = []

        # unpack the parameters associated with each selected Wood test problem
        for (problem,rows,cols,cost_max,delay_max,resource_max,resource_budget,) in test_settings:

            # generate multiple independent random realizations of the same
            # structural Wood configuration
            for rep in range(reps_per_wood_setting):
    
                # derive a deterministic seed unique to the benchmark problem number
                seed = base_seed + 1000 * problem + rep
    
                # generate the directed grid network using the selected Wood parameters
                G, s, t, density = generate_wood_grid(rows=rows,cols=cols,cost_max=cost_max,delay_max=delay_max,
                                                      resource_max=resource_max,seed=seed,)
    
                # store the graph together with benchmark metadata needed during evaluation
                evaluation_graphs.append({"G": G, "s": s, "t": t,"density": density,"seed": seed,
    
                        # Wood benchmark identifiers
                        "wood_problem": problem, "rows": rows, "cols": cols, "rep": rep,
    
                        # actual generated graph dimensions
                        "n": G.number_of_nodes(), "m": G.number_of_edges(),
    
                        # Wood benchmark parameters and benchmark-specific interdiction resource budget
                        "cost_max": cost_max,"delay_max": delay_max,"resource_max": resource_max,
                        "attack_budget": resource_budget,})

                # print benchmark-level progress during graph generation
                print(f"Generated Wood problem {problem} | "
                    f"rep={rep + 1}/{reps_per_wood_setting} | "
                    f"{rows}x{cols} | "
                    f"nodes={G.number_of_nodes()} | "
                    f"arcs={G.number_of_edges()} | "
                    f"budget={resource_budget}",
                    flush=True,)

        # save one fixed Wood evaluation set so every trained model is tested
        # on exactly the same graph realizations
        with open(output_path, "wb") as f:
            pickle.dump(evaluation_graphs, f)
    
        print(f"\nFinished. Saved {len(evaluation_graphs)} "
              f"Wood graphs to {output_path}",
              flush=True,)

        # Wood generation is complete; do not continue into standard graph generation
        return



    # EXTERNAL NETWORK GENERATION

    # Load and save fixed source-sink evaluation instances from the
    # real-world rail/road transportation network.
    if eval_mode == "external":

        # External evaluation is currently implemented only for
        # shortest-path interdiction.
        if problem_type != "shortest_path":
            raise ValueError("External evaluation currently supports shortest_path only.")

        # Natural supply and demand nodes defined by the external network.
        # These use the ORIGINAL external node identifiers.
        supply_nodes = [1, 2, 3, 4, 5, 10]
        demand_nodes = [6, 7, 8, 9, 1077]

        # Store one evaluation instance for every reachable supply-demand pair.
        evaluation_graphs = []

        for source in supply_nodes:

            for sink in demand_nodes:

                # Load the same fixed physical network for each OD pair.
                # load_external_network converts the original source and sink
                # identifiers to the internal consecutive node IDs used by
                # the model.
                G, s, t, density = load_external_network(
                    node_path="data/node_data.csv",
                    arc_path="data/arc_data_with_detour_penalties.csv",
                    source=source,sink=sink)

                # Store the graph and OD-pair metadata in the same general
                # format used by the other evaluation modes.
                evaluation_graphs.append({"G": G,"s": s,"t": t, "density": density,
                    "n": G.number_of_nodes(),"m": G.number_of_edges(),

                    # Preserve original external OD identifiers for analysis.
                    "source_original": source,"sink_original": sink,

                    "network_name": "rail_road_transportation_network"})

                print(f"Prepared external OD pair | "
                    f"source={source} | "
                    f"sink={sink} | "
                    f"internal_source={s} | "
                    f"internal_sink={t}",
                    flush=True)

        # Verify that the expected 6 x 5 = 30 OD pairs were created.
        expected_pairs = len(supply_nodes) * len(demand_nodes)

        if len(evaluation_graphs) != expected_pairs:
            raise ValueError(f"Expected {expected_pairs} external OD pairs, "
                f"but generated {len(evaluation_graphs)}.")

        # Save one fixed external evaluation set so every trained model
        # is evaluated on exactly the same physical network and OD pairs.
        with open(output_path, "wb") as f:
            pickle.dump(evaluation_graphs, f)

        print(f"\nFinished. Saved {len(evaluation_graphs)} external OD pairs "
            f"to {output_path}",flush=True)

        # External graph preparation is complete; do not continue into
        # synthetic graph generation.
        return



    # SYNTHETIC ONE-IN EVALUATION GRAPH GENERATION

    # generate 30 independent graph replications for each (n, m) setting and
    # evaluate the same five interdiction budgets used during model training
    reps_per_setting = 30
    test_attack_limits = [1, 2, 3, 4, 5]

    # ARC ATTRIBUTE RANGES

    # Value-OOD evaluation deliberately uses cost and penalty values outside
    # the 1-10 ranges represented during training.
    if eval_mode == "value_ood":

        cost_low = 11
        cost_high = 50
        penalty_low = 11
        penalty_high = 50

    else:

        # Standard synthetic evaluation ranges used during training
        cost_low = 1
        cost_high = 10
        penalty_low = 1
        penalty_high = 10


    # Capacity range is unchanged because the current value-OOD experiment
    # is focused on shortest-path cost and interdiction-penalty values.
    capacity_low = 1
    capacity_high = 20

    # maximum attack budget is used to establish the minimum required
    # source-to-sink edge connectivity for max-flow evaluation graphs
    max_attack_budget = max(test_attack_limits)

    # fixed evaluation seed ensures that the same graph instances can be
    # reproduced and used consistently across all evaluated models
    base_seed = 5

    # total number of graphs expected in the completed evaluation dataset
    total_expected = len(test_settings) * reps_per_setting


    # RESUME PREVIOUS GRAPH GENERATION

    # if a partially completed evaluation file already exists, load the graphs
    # that were previously generated rather than restarting from the beginning
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            evaluation_graphs = pickle.load(f)

        print(f"Resuming from {len(evaluation_graphs)} previously generated graphs.", flush=True,)

    # initialize an empty graph collection when no previous file exists
    else:
        evaluation_graphs = []

    # identify previously generated graphs by their network dimensions and
    # replication number so they are not duplicated when generation resumes
    existing_keys = {(g["n"], g["m"], g["rep"]) for g in evaluation_graphs}


    # GRAPH GENERATION LOOP

    # generate the requested number of graph replications for each network
    # size and density configuration
    for n, m in test_settings:

        for rep in range(reps_per_setting):

            # skip graphs that were already successfully generated and saved
            if (n, m, rep) in existing_keys:
                continue

            # construct a deterministic seed from the graph dimensions and
            # replication number to make each evaluation instance reproducible
            seed = base_seed + 100000 * n + 100 * m + rep


            # MAX-FLOW GRAPH GENERATION

            # max-flow graphs require additional connectivity screening to prevent
            # the tested interdiction budgets from trivially disconnecting s and t
            if problem_type == "max_flow":

                # count the number of candidate graphs tested for this replication
                attempt = 0

                while True:

                    # derive a unique deterministic seed for each candidate graph
                    # while preserving reproducibility of the search procedure
                    candidate_seed = seed * 1000000 + attempt

                    # generate a candidate One-In directed network
                    G, s, t, density = generate_one_in_network(n=n,m=m,cost_low=cost_low,cost_high=cost_high,penalty_low=penalty_low,
                                                               penalty_high=penalty_high,capacity_low=capacity_low,capacity_high=capacity_high,
                                                               seed=candidate_seed,)

                    # compute the minimum number of directed arcs whose removal
                    # would disconnect the source from the sink
                    edge_connectivity = nx.edge_connectivity(G, s, t)

                    # accept the graph only when its edge connectivity exceeds the
                    # largest tested attack budget, preventing complete disconnection
                    # using K <= max_attack_budget interdictions
                    if edge_connectivity > max_attack_budget:

                        # retain the accepted candidate seed so the exact graph can be reproduced later
                        seed = candidate_seed

                        print(f"Accepted graph | "
                            f"n={n}, m={m}, rep={rep} | "
                            f"edge_connectivity={edge_connectivity} | "
                            f"attempts={attempt + 1}",
                            flush=True,)

                        break

                    # reject the current candidate and try another graph
                    attempt += 1

                    # periodically report progress because finding sufficiently
                    # connected max-flow graphs may require many attempts
                    if attempt % 10 == 0:
                        print(f"Searching | "
                            f"n={n}, m={m}, rep={rep} | "
                            f"attempt={attempt}",
                            flush=True,)



            # SHORTEST-PATH AND MIN-COST-FLOW GRAPH GENERATION

            # these problem types do not require the max-flow connectivity filter,
            # so generate one graph directly using the deterministic evaluation seed
            elif problem_type in ("shortest_path", "min_cost_flow"):

                G, s, t, density = generate_one_in_network(n=n,m=m,cost_low=cost_low,cost_high=cost_high,penalty_low=penalty_low,
                                                           penalty_high=penalty_high,capacity_low=capacity_low,capacity_high=capacity_high,
                                                           seed=seed,)

            else:
                raise ValueError(f"Unknown problem_type: {problem_type}")
            


            # SAVE GENERATED GRAPH

            # store the graph and identifying metadata required by evaluate.py
            evaluation_graphs.append({"G": G,"s": s,"t": t,"density": density,"seed": seed,
                                      "n": n,"m": m,"rep": rep,     
                                      # attribute ranges used to generate this evaluation graph
                                      "cost_low": cost_low,"cost_high": cost_high,
                                      "penalty_low": penalty_low,"penalty_high": penalty_high,})

            # record the completed graph key so it cannot be generated twice
            existing_keys.add((n, m, rep))

            # save after every successfully generated graph so progress is preserved
            # if a long-running generation job is interrupted or reaches its time limit
            with open(output_path, "wb") as f:
                pickle.dump(evaluation_graphs, f)

            # report progress toward the complete evaluation graph collection
            print(f"\nGenerated and saved "
                  f"{len(evaluation_graphs)}/{total_expected} evaluation graphs.",flush=True,)

    # report completion and the location of the saved evaluation graph file
    print(f"\nFinished. Saved {len(evaluation_graphs)} graphs to {output_path}",flush=True,)


if __name__ == "__main__":

    # sys.argv[1] selects the interdiction problem
    problem_type = sys.argv[1] if len(sys.argv) > 1 else "shortest_path"
    # sys.argv[2] selects the evaluation graph type
    eval_mode = sys.argv[2] if len(sys.argv) > 2 else "ood_size"

    generate_evaluation_graphs(problem_type, eval_mode)