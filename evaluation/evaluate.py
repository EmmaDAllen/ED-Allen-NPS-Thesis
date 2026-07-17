# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:51:24 2026

@author: emmallen
"""

from random import sample

import torch
import csv
import sys
import time

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


PROBLEM_INPUT_DIMS = {
    "shortest_path": 8,
    "max_flow": 7,
    "min_cost_flow": 9
}

def get_model(model_type, problem_type, device):

    if problem_type not in PROBLEM_INPUT_DIMS:
        raise ValueError(f"Unknown problem type: {problem_type}")

    input_dim = PROBLEM_INPUT_DIMS[problem_type]


    if model_type == "tropical":
        return TropicalInterdictionModel(
            input_dim=input_dim,d_model=64, n_heads=4,num_layers=2,device=device
        ).to(device)

    elif model_type == "transformer":
        return StandardTransformerInterdictionModel(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,
        ).to(device)

    elif model_type == "gnn":
        return GNNInterdictionModel(
            input_dim=input_dim,d_model=64,num_layers=2
        ).to(device)
    
    elif model_type == "edge_transformer":
        return EdgeBiasTransformerInterdictionModel(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2
        ).to(device)
    
        
    elif model_type == "tropical_v2":
        return TropicalInterdictionModelV2(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,device=device
        ).to(device)


    else:
        raise ValueError(f"Unknown model type: {model_type}")



def evaluate():
    
    # use GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate empty tropical attention model
    model_type = sys.argv[1] if len(sys.argv) > 1 else "tropical"
    problem_type = sys.argv[2] if len(sys.argv) > 2 else "shortest_path"
    eval_mode = sys.argv[3] if len(sys.argv) > 3 else "id"

    model = get_model(model_type, problem_type, device)

    run_name = f"{model_type}_{problem_type}"
    
    # load saved moved weigts from training
    model.load_state_dict(
    torch.load(f"saved_models/{run_name}_model.pt",
               map_location=device))

    # put model in evaluation mode
    model.eval()

    if eval_mode == "id":
        test_settings = [
            (30, 75),
            (30, 120),
            (50, 125),
            (50, 200),
            (75, 188),
            (75, 300),]
        
    elif eval_mode == "ood_size":
        test_settings = [
            (100, 250),
            (100, 400),
            (150, 375),
            (150, 600),
            (200, 500),
            (200, 800),
            (300, 750),
            (300, 1200),
            (400, 1000),
            (400, 1600),]
    
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

    # number of test graphs per network setting
    reps_per_setting = 20
    
    # evaluate the model for multiple interdiction budgets
    test_attack_limits = [1, 2, 3, 4, 5]
    
    # base seed thats different from training data to create unseen test graphs
    base_seed = 999999

    # store one row of results per solved test instance
    results_rows = []
    
    # loop through each interdiction budget
    for test_attack_limit in test_attack_limits:

        # track summary metrics separately for each budget
        total_exact = 0
        total_hamming = 0
        total_gap = 0.0
        total_instances = 0
        total_mip_solve_time = 0.0
        total_inference_time = 0.0

        print(f"\nEvaluating attack limit K={test_attack_limit}")

        # loop through each network size/density setting
        for n, m in test_settings:

            # generate multiple test replications for each setting
            for rep in range(reps_per_setting):

                # unique seed for reproducibility
                seed = base_seed + 100000 * n + 100 * m + rep

                # generate test network
                G, s, t, density = generate_one_in_network(n=n, m=m, cost_low=1,cost_high=10,penalty_low=2,
                                                           penalty_high=10,capacity_low=1,capacity_high=20,seed=seed)

                # start timer for MIP solve time
                mip_start = time.perf_counter()

                if problem_type == "min_cost_flow":
                    baseline_max_flow = nx.maximum_flow_value(G,_s=s,_t=t,capacity="capacity")
                    flow_demand = max(1, int(0.5 * baseline_max_flow))
                else:
                    flow_demand = 1
                
                # solve MIP to get optimal interdiction decision for this K
                sample = solve_instance(G=G, s=s, t=t,density=density,attack_limit=test_attack_limit,
                                        problem_type=problem_type, flow_demand=flow_demand)
                
                # end timer for MIP solve time and calculate total time
                mip_end = time.perf_counter()
                mip_total_time = mip_end - mip_start
    
                # skip infeasible or non-optimal solves
                if sample is None:
                    continue

                # convert solved sample into model inputs
                edge_features, edge_bias, _, attack_limit = sample_to_tensors(sample)

                # add batch dimension and move tensors to GPU/CPU
                edge_features = edge_features.unsqueeze(0).to(device)
                edge_bias = edge_bias.unsqueeze(0).to(device)

                # since this is one graph with no padding, all edges are real
                mask = torch.ones(1, edge_features.shape[1], dtype=torch.bool,device=device)

                # synchronize GPU and start timer for model inference time
                if device == "cuda":
                    torch.cuda.synchronize()

                # run model inference and get predicted interdiction decision, 
                # while timing the inference time
                inference_start = time.perf_counter()

                # run model without computing gradients
                with torch.no_grad():
                    logits = model(edge_features, edge_bias=edge_bias, mask=mask)

                # synchronize GPU and start timer for model inference time
                if device == "cuda":
                    torch.cuda.synchronize()

                # end timer for model inference time and calculate total time
                inference_end = time.perf_counter()
                inference_time = inference_end - inference_start

                # get edge scores for the single graph
                real_logits = logits[0]

                # number of edges to interdict
                k = int(attack_limit)

                # select top-k model-scored edges
                predicted_attack = torch.zeros_like(real_logits)
                topk_indices = torch.topk(real_logits, k=k).indices
                predicted_attack[topk_indices] = 1.0

                # convert predictions and optimal labels to normal Python lists
                predicted_attack_list = predicted_attack.cpu().int().tolist()
                optimal_attack_list = sample["attack"]

                # exact match = predicted interdiction set exactly equals MIP set
                exact = int(predicted_attack_list == optimal_attack_list)

                # hamming distance = number of edge labels that differ
                hamming = sum(
                    p != o for p, o in zip(predicted_attack_list, optimal_attack_list))

                if problem_type == "shortest_path":
                    optimal_objective = sample["path_length"]
                    predicted_objective = shortest_path_after_attack(sample, predicted_attack_list)
                    objective_gap = (optimal_objective - predicted_objective) / max(abs(optimal_objective), 1e-8)
                
                elif problem_type == "max_flow":
                    optimal_objective = sample["max_flow"]
                    baseline_objective = sample["baseline_max_flow"]
                    predicted_objective = max_flow_after_attack(sample, predicted_attack_list)
                    objective_gap = (predicted_objective - optimal_objective) / max(abs(baseline_objective), 1e-8)

                elif problem_type == "min_cost_flow":
                    optimal_objective = sample["min_cost_flow"]
                    predicted_objective = min_cost_flow_after_attack(sample, predicted_attack_list)
                    objective_gap = (optimal_objective - predicted_objective) / max(abs(optimal_objective), 1e-8)


                # save detailed instance-level result
                results_rows.append({"n_nodes": n,
                        "n_edges": m,
                        "density": m / n,
                        "replication": rep,
                        "attack_limit": k,
                        "optimal_objective": optimal_objective,
                        "predicted_objective": predicted_objective,
                        "objective_gap": objective_gap,
                        "hamming_distance": hamming,
                        "exact_match": exact,
                        "num_correct_edges": sum((p == 1 and o == 1)
                                                 for p, o in zip(predicted_attack_list, optimal_attack_list)),
                        "num_predicted_attacks": sum(predicted_attack_list),
                        "num_optimal_attacks": sum(optimal_attack_list),
                        "mip_solve_time": sample.get("mip_solve_time", mip_total_time),
                        "mip_total_time": mip_total_time,
                        "inference_time": inference_time,})

                # update summary metrics for this K
                total_exact += exact
                total_hamming += hamming
                total_gap += objective_gap
                total_instances += 1
                total_mip_solve_time += sample.get("mip_solve_time", mip_total_time)
                total_inference_time += inference_time

                print(f"K={test_attack_limit}, n={n}, m={m}, rep={rep} | "
                        f"Opt={optimal_objective:.2f} | "
                        f"Pred={predicted_objective:.2f} | "
                        f"Gap={objective_gap:.4f} | "
                        f"Hamming={hamming} | "
                        f"Exact={exact}")

        # print summary results for this interdiction budget
        print(f"\nFINAL TEST RESULTS FOR K={test_attack_limit}")
        print(f"Instances evaluated: {total_instances}")
        print(f"Exact match rate: {total_exact / total_instances:.4f}")
        print(f"Average Hamming distance: {total_hamming / total_instances:.4f}")
        print(f"Average objective gap: {total_gap / total_instances:.4f}")
        print(f"Average MIP solve time: {total_mip_solve_time / total_instances:.6f} seconds")
        print(f"Average model inference time: {total_inference_time / total_instances:.6f} seconds")

    # save all detailed results across all K values
    with open(f"results/evaluation_results_{model_type}_{problem_type}_{eval_mode}.csv", "w", newline="") as csvfile:
        fieldnames = results_rows[0].keys()

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results_rows)

    print(f"\nSaved results/evaluation_results_{model_type}_{problem_type}_{eval_mode}.csv")


if __name__ == "__main__":
    evaluate()
