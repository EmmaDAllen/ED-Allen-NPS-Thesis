# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:51:24 2026

@author: emmallen
"""

"""evaluate.py

Evaluation pipeline for trained network-interdiction models.

Usage
Run from the repository root with:

    PYTHONPATH=. python evaluation/evaluate.py MODEL_TYPE PROBLEM_TYPE EVAL_MODE

The script:
1. Constructs the requested model architecture.
2. Loads the best-validation-loss checkpoint from training.
3. Generates previously unseen test graphs.
4. Solves each graph exactly with the appropriate interdiction MIP.
5. Uses the trained model to predict an interdiction set.
6. Compares the predicted and MIP-optimal attack sets.
7. Evaluates the downstream objective obtained by the prediction.
8. Records MIP solve time, model inference time, and prediction metrics.
9. Saves one CSV row per evaluated graph instance."""


import torch
import csv
import sys
import time
import networkx as nx
import pickle

from data.random_networks import generate_one_in_network
from optimization.mip import solve_instance
from models.tropical_attention import TropicalInterdictionModel
from models.standard_transformer import StandardTransformerInterdictionModel
from models.gnn import GNNInterdictionModel
from models.edge_bias_transformer import EdgeBiasTransformerInterdictionModel
from data.interdiction_data import sample_to_tensors
from evaluation.metrics import shortest_path_after_attack
from evaluation.metrics import max_flow_after_attack
from evaluation.metrics import min_cost_flow_after_attack
from models.tropical_attention_V2 import TropicalInterdictionModel as TropicalInterdictionModelV2



# number of edge-level input features created for each interdiction problem by 
# sample_to_tensors() - these values must exactly match the feature construction logic in 
# data/interdiction_data.py and the dimensions used during training
PROBLEM_INPUT_DIMS = {
    "shortest_path": 8,
    "max_flow": 7,
    "min_cost_flow": 9}



def get_model(model_type, problem_type, device):

    """ Construct the requested neural-network architecture.

    The architecture and hyperparameters must match the model that was used when the 
    saved checkpoint was created. Otherwise, model.load_state_dict() will fail because 
    the parameter shapes or names will not match.

    Parameters
    model_type : str
        Requested model architecture.

    problem_type : str
        Network-interdiction problem being evaluated.

    device : str or torch.device
        Device on which the model should be stored.

    Returns
    torch.nn.Module
        Initialized model moved onto the requested device.

    Raises

    ValueError
        If model_type or problem_type is not supported."""


    # confirm that the selected problem has a known input-feature dimension
    if problem_type not in PROBLEM_INPUT_DIMS:
        raise ValueError(f"Unknown problem type: {problem_type}")

    # retrieve the number of input features expected by the selected problem type
    input_dim = PROBLEM_INPUT_DIMS[problem_type]

    # use the same architecture settings as the training pipeline - holding these 
    # values constant is required to load the checkpoint

    # tropical attention model with edge bias
    if model_type == "tropical":
        # construct Tropical Attention Transformer
        return TropicalInterdictionModel(
            input_dim=input_dim,d_model=64, n_heads=4,num_layers=2,device=device,dropout=0.1,use_edge_bias=True
        ).to(device)

    # standard transformer model without edge bias
    elif model_type == "transformer":
        # construct the standard Transformer baseline
        return StandardTransformerInterdictionModel(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,dropout=0.1
        ).to(device)

    # gnn model
    elif model_type == "gnn":
        # construct the edge-to-edge message-passing GNN baseline
        return GNNInterdictionModel(
            input_dim=input_dim,d_model=64,num_layers=2
        ).to(device)

    # edge-bias transformer model
    elif model_type == "edge_transformer":
        # construct the Transformer with additive graph-structure bias
        return EdgeBiasTransformerInterdictionModel(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,dropout=0.1
        ).to(device)
    
    # version 2 tropical attention model with edge bias and slightly different architecture
    elif model_type == "tropical_v2":
        # construct Tropical Attention Transformer Version 2
        return TropicalInterdictionModelV2(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,device=device,dropout=0.1,use_edge_bias=True
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")



def evaluate():

    """Evaluate one trained model on newly generated graph instances.

    The model, interdiction problem, and evaluation mode are selected from command-line arguments. 
    Each generated graph is solved exactly using the corresponding MIP, then evaluated using the
    model's top-k predicted interdiction set."""



    # DEVICE AND COMMAND-LINE CONFIGURATION
    
    # use a CUDA GPU when one is available, otherwise, perform model inference on the CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sys.argv[0] contains the script name
    # sys.argv[1] specifies the model architecture, default: tropical
    model_type = sys.argv[1] if len(sys.argv) > 1 else "tropical"
    # sys.argv[2] specifies the interdiction problem, default: shortest_path
    problem_type = sys.argv[2] if len(sys.argv) > 2 else "shortest_path"
    # sys.argv[3] specifies whether evaluation uses newly generated in-distribution settings 
    # or larger out-of-distribution graphs, default: id_new
    eval_mode = (sys.argv[3] if len(sys.argv) > 3 else "id_new")



    # MODEL AND CHECKPOINT LOADING

    # construct an empty model having the same architecture and parameter dimensions used 
    # during training
    model = get_model(model_type, problem_type, device)

    # combine the model and problem names to reproduce the checkpoint naming convention used 
    # by train.py
    run_name = f"{model_type}_{problem_type}"
    
    # load the checkpoint corresponding to the epoch with the lowest validation loss during training
    checkpoint_path = (f"saved_models/{run_name}_best_model.pt")

    # map_location ensures that a checkpoint can be loaded even when the current evaluation device
    # differs from the training device, weights_only=False is required because the checkpoint 
    # contains metadata and optimizer information in addition to tensor weights
    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=False,)

    # copy the saved parameter tensors into the newly constructed model
    model.load_state_dict(checkpoint["model_state_dict"])

    # put model in evaluation mode
    # disables training-specific behavior such as dropout
    model.eval()

   # evaluate all five interdiction budgets used in the experiments
    test_attack_limits = [1, 2, 3, 4, 5]

    # store one dictionary for every successfully solved and evaluated graph instance - these 
    # rows will later be written to one CSV file
    results_rows = []



    # LOAD PRE-GENERATED EVALUATION GRAPHS

    graph_path = (f"evaluation_graphs/"
                  f"{problem_type}_{eval_mode}_graphs.pkl")

    with open(graph_path, "rb") as f:
        evaluation_graphs = pickle.load(f)

    print(f"\nLoaded {len(evaluation_graphs)} " 
          f"pre-generated evaluation graphs from {graph_path}.")

    
    # INTERDICTION BUDGET LOOP
    
    # loop through each interdiction budget
    for test_attack_limit in test_attack_limits:

        # reset summary accumulators at the beginning of each budget == printed averages 
        # describe only the current K value

        # number of evaluated graphs where the complete predicted interdiction set exactly
        # matches the MIP-optimal set
        total_exact = 0

        # sum of edge-label disagreements across evaluated graphs
        total_hamming = 0

        # sum of normalized downstream objective gaps
        total_gap = 0.0

        # number of graph instances successfully solved and evaluated
        total_instances = 0

        # sum of solver-reported MIP optimization times
        total_mip_solve_time = 0.0

        # sum of model forward-pass inference times
        total_inference_time = 0.0

        print(f"\nEvaluating attack limit K={test_attack_limit}")


        for graph_data in evaluation_graphs:

            # retrieve the already-generated graph
            G = graph_data["G"]
            s = graph_data["s"]
            t = graph_data["t"]
            density = graph_data["density"]
            seed = graph_data["seed"]
            n = graph_data["n"]
            m = graph_data["m"]
            rep = graph_data["rep"]



            # EXACT MIP SOLUTION

            # start an external wall-clock timer immediately before any problem-specific
            # preprocessing and the MIP solve = mip_total_time therefore includes both 
            # preprocessing and the solve_instance() function call
            mip_start = time.perf_counter()

            if problem_type == "min_cost_flow":

                # compute the maximum feasible source-to-sink flow before interdiction
                # this establishes an upper bound for a feasible minimum-cost-flow demand
                baseline_max_flow = nx.maximum_flow_value(G,s,t,capacity="capacity")

                # set the min-cost-flow demand to 50% of the graph's pre-interdiction maximum flow
                # max(1, ...) prevents zero demand on graphs with very small baseline flow
                flow_demand = max(1, int(0.5 * baseline_max_flow))

            else:
                # shortest-path and maximum-flow interdiction do not use the min-cost-flow 
                # demand argument, default value is still supplied to maintain one shared
                # solve_instance() interface
                flow_demand = 1
                
            # solve the selected network-interdiction problem exactly for the current graph and attack
            # budget - the returned sample includes: graph structure, edge attributes, optimal attack
            # labels, optimal objective, attack budget, and solver timing/status information.
            sample = solve_instance(G=G, s=s, t=t,density=density,attack_limit=test_attack_limit,
                                        problem_type=problem_type, flow_demand=flow_demand)
                
            # stop the external wall-clock timer after solve_instance returns
            mip_end = time.perf_counter()

            # total elapsed time surrounding the complete solve call                
            mip_total_time = mip_end - mip_start
    
            # solve_instance() returns None when the generated graph does not produce a valid optimal 
            # sample, such as an infeasible min-cost-flow instance or unsuccessful solver termination
            # these instances are omitted from model evaluation
            if sample is None:
                continue



            # CONVERT SOLVED SAMPLE TO MOPDEL INPUT TENSORS

            # Convert the stored graph sample into: 
            # edge_features: Shape (num_edges, input_dim)
            # edge_bias: Shape (num_edges, num_edges)
            # the attack labels and attack limit returned by sample_to_tensors() are not needed 
            # here because the original sample dictionary already contains them
            edge_features, edge_bias, _, _ = sample_to_tensors(sample)

            # add a batch dimension because all model forward functions expect:
            # (batch_size, num_edges, feature_dimension)
            # evaluation processes one graph at a time, so the batch size is 1
            edge_features = edge_features.unsqueeze(0).to(device)

            # add a batch dimension to the pairwise graph-bias matrix:(num_edges, num_edges)
            # == (1, num_edges, num_edges)
            edge_bias = edge_bias.unsqueeze(0).to(device)

            # no padding is required because this evaluation batch contains only one graph
            # mark every edge position as valid
            mask = torch.ones(1, edge_features.shape[1], dtype=torch.bool,device=device)




            # MODEL INFERENCE TIMING 

            # CUDA operations execute asynchronously - Synchronizing before the timer ensures
            # all prior GPU work has completed and is not accidentally counted as part of this 
            # model's inference time
            if device == "cuda":
                torch.cuda.synchronize()

            # record the beginning of the timed forward pass
            inference_start = time.perf_counter()

            # disable autograd because evaluation does not require gradients or parameter updates
            with torch.no_grad():

                # produce one raw interdiction score for every edge 
                # logits shape: (1, num_edges)
                logits = model(edge_features, edge_bias=edge_bias, mask=mask)

            # wait until the asynchronous CUDA forward pass has completed before stopping the timer
            if device == "cuda":
                torch.cuda.synchronize()

            # record the end of the timed model forward pass
            inference_end = time.perf_counter()

            # wall-clock time required for one model inference
            inference_time = inference_end - inference_start

            # remove the batch dimension to obtain one score per edge
            # real_logits shape: (num_edges,)
            real_logits = logits[0]

            # retrieve the interdiction budget stored in the solved sample
            # should equal test_attack_limit
            k = int(sample["attack_limit"])

            # count the number of real edges scored by the model
            num_edges = real_logits.numel()

            # the number of selected edges must be between zero and the total
            # number of graph edges
            if not 0 <= k <= num_edges:
                raise ValueError( f"Invalid attack limit K={k} for a graph "
                                f"with {num_edges} edges.")




            # CONVERT LOGITS TO A DISCRETE INTERDICTION SET

            # initialize every edge as not interdicted
            # predicted_attack shape: (num_edges,)
            predicted_attack = torch.zeros_like(real_logits)

            if k > 0:

                # identify the indices of the k edges receiving the largest model logits
                # ensures the predicted attack set satisfies the interdiction budget exactly
                topk_indices = torch.topk(real_logits,k=k,).indices

                # mark the selected edges as interdicted in the predicted attack vector
                predicted_attack[topk_indices] = 1.0

            # convert the GPU tensor into a regular Python list of integer labels for 
            # comparison and objective evaluation
            predicted_attack_list = predicted_attack.cpu().int().tolist()

            # retrieve the MIP-optimal binary attack vector - the ordering matches the 
            # stored edge ordering in the sample and the model input tensors
            optimal_attack_list = sample["attack"]




            # ATTACK-SET PREDICTION METRICS

            # Exact match equals 1 only when every predicted edge  label agrees with the 
            # MIP-optimal attack vector
            exact = int(predicted_attack_list == optimal_attack_list)

            # hamming distance counts how many edge labels differ between the predicted 
            # and optimal attack vectors - because both vectors select k edges, each incorrect
            # replacement usually contributes two mismatches: one missed optimal edge,
            # one incorrectly selected edge
            hamming = sum(
                    p != o for p, o in zip(predicted_attack_list, optimal_attack_list))




            # DOWNSTREAM OBJECTIVE EVALUATION

            if problem_type == "shortest_path":

                # the attacker seeks to maximize the shortest source-to-sink path length
                # sample["path_length"] = objective achieved by the MIP-optimal interdiction set
                optimal_objective = sample["path_length"]

                # re-solve the follower shortest-path problem after applying the model-predicted 
                # interdiction set
                predicted_objective = shortest_path_after_attack(sample, predicted_attack_list)

                # the optimal attack should produce an objective at least as large as the predicted attack
                # gap = (optimal - predicted) / |optimal| = a gap of zero means the prediction achieves the 
                # same objective value as the optimal interdiction, even when the exact attack sets differ
                objective_gap = (optimal_objective - predicted_objective) / max(abs(optimal_objective), 1e-8)

                
            elif problem_type == "max_flow":

                # the attacker seeks to minimize the surviving source-to-sink maximum flow
                optimal_objective = sample["max_flow"]

                # store the graph's maximum-flow value before any interdiction - used to normalize the gap
                baseline_objective = sample["baseline_max_flow"]

                # recompute surviving maximum flow after applying the model-predicted interdiction set
                predicted_objective = max_flow_after_attack(sample, predicted_attack_list)


                # MIP-optimal attack should leave flow no larger than the model-predicted attack
                # gap = (predicted - optimal) / |baseline| = normalizing by baseline flow avoids 
                # instability when the optimal surviving flow equals zero
                objective_gap = (predicted_objective - optimal_objective) / max(abs(baseline_objective), 1e-8)


            elif problem_type == "min_cost_flow":

                # the attacker seeks to maximize the defender's minimum feasible flow cost
                optimal_objective = sample["min_cost_flow"]

                # recompute minimum-cost flow after applying the model-predicted interdiction set
                predicted_objective = min_cost_flow_after_attack(sample, predicted_attack_list)

                # MIP-optimal attack should produce a cost at least as large as the model-predicted attack
                # gap = (optimal - predicted) / |optimal|
                objective_gap = (optimal_objective - predicted_objective) / max(abs(optimal_objective), 1e-8)

            else:
                # branch should be unreachable because problem_type was already checked in get_model()
                raise ValueError(f"Unknown problem type: {problem_type}")                




            # SAVE DETAILED INSTANCE-LEVEL METRICS

            # append one row describing the current solved graph, model prediction, objective 
            # quality, and runtime
            results_rows.append({

                    # experiment identifiers
                    "model_type": model_type,
                    "problem_type": problem_type,
                    "eval_mode": eval_mode,

                    # graph identifiers and structure
                    "graph_seed": seed,
                    "n_nodes": sample["n_nodes"],
                    "n_edges": len(sample["u"]),
                    "density": sample["density"],
                    "replication": rep,

                    # interdiction budget
                    "attack_limit": k,

                    # downstream objective values
                    "optimal_objective": optimal_objective,
                    "predicted_objective": predicted_objective,
                    "objective_gap": objective_gap,

                    # attack set comparison metrics
                    "hamming_distance": hamming,
                    "exact_match": exact,

                    # number of edges that both the model and MIP selected for interdiction
                    "num_correct_edges": sum((p == 1 and o == 1)
                                                 for p, o in zip(predicted_attack_list, optimal_attack_list)),

                    # number of model-selected interdictions
                    "num_predicted_attacks": sum(predicted_attack_list),

                    # number of MIP-selected interdictions
                    "num_optimal_attacks": sum(optimal_attack_list),

                    # internal solver time reported by solve_instance when available
                    # fall back to the externally measured total time if sample does not contain a solver timer
                    "mip_solve_time": sample.get("mip_solve_time", mip_total_time),

                    # total wall-clock time around preprocessing and solve_instance()
                    "mip_total_time": mip_total_time,

                    # timed model forward-pass duration                     
                    "inference_time": inference_time,})




            # UPDATE PER-BUDGET SUMMARY ACCUMULATORS

            # add 1 when the complete predicted attack set matches the optimal set
            total_exact += exact

            # add this graph's edge-label disagreement count
            total_hamming += hamming

            # add this graph's normalized downstream objective gap
            total_gap += objective_gap

            # record one more successfully evaluated instance
            total_instances += 1

            # add the solver's internal optimization time when available
            total_mip_solve_time += sample.get("mip_solve_time", mip_total_time)

            # add the model forward-pass time
            total_inference_time += inference_time

            # print the current instance's results so evaluation
            # progress can be monitored while the script runs
            print(f"K={test_attack_limit}, n={n}, m={m}, rep={rep} | "
                        f"Opt={optimal_objective:.2f} | "
                        f"Pred={predicted_objective:.2f} | "
                        f"Gap={objective_gap:.4f} | "
                        f"Hamming={hamming} | "
                        f"Exact={exact}")




        # PRINT SUMMART FOR CURRENT INTERDICTION BUDGET

        # print summary results for this interdiction budget
        print(f"\nFINAL TEST RESULTS FOR K={test_attack_limit}")
        print(f"Instances evaluated: {total_instances}")

        # avoid division by zero if every graph generated for this
        # budget was skipped because solve_instance returned None
        if total_instances == 0:
            print("No valid instances were evaluated for this attack limit.")
            continue

        # fraction of graphs where the full predicted attack vectorexactly equals
        # the MIP-optimal attack vector     
        print(f"Exact match rate: {total_exact / total_instances:.4f}")

        # mean number of edge-label disagreements per graph
        print(f"Average Hamming distance: {total_hamming / total_instances:.4f}")

        # mean normalized downstream objective loss relative to the MIP solution
        print(f"Average objective gap: {total_gap / total_instances:.4f}")

        # mean solver-reported MIP optimization time
        print(f"Average MIP solve time: {total_mip_solve_time / total_instances:.6f} seconds")

        # mean model forward-pass inference time
        print(f"Average model inference time: {total_inference_time / total_instances:.6f} seconds")




    # SAVE ALL INSTANCE-LEVEL RESULTS TO CSV

    # at least one graph must have been successfully solved and evaluated before field names 
    # can be taken from results_rows[0]
    if not results_rows:
        raise RuntimeError("Evaluation completed without any valid result rows.")

    # construct an output filename uniquely identifying the model problem, and evaluation mode
    results_path = ( "results/"
        f"evaluation_results_"
        f"{model_type}_"
        f"{problem_type}_"
        f"{eval_mode}.csv")

    # write one CSV row per successfully evaluated graph instance
    with open(results_path, "w", newline="") as csvfile:

        # preserve the column order used by the first result dictionary
        fieldnames = results_rows[0].keys()

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)


    print(f"\nSaved results/evaluation_results_{model_type}_{problem_type}_{eval_mode}.csv")


# run evaluation only when this file is executed directly - importing evaluate.py from another 
# Python module will not automatically begin evaluation
if __name__ == "__main__":
    evaluate()
