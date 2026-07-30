# -*- coding: utf-8 -*-
"""
Created on Thu May 14 07:09:23 2026

@author: emmallen
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import time
import csv
import random
import schedulefree

from models.tropical_attention import TropicalInterdictionModel
from models.standard_transformer import StandardTransformerInterdictionModel
from models.gnn import GNNInterdictionModel
from models.edge_bias_transformer import EdgeBiasTransformerInterdictionModel
from models.tropical_attention_V2 import TropicalInterdictionModel as TropicalInterdictionModelV2
from data.interdiction_data import InterdictionDataset
from data.interdiction_data import collate_graphs
from evaluation.metrics import compute_metrics


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
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,dropout=0.1,device=device,use_edge_bias=True
        ).to(device)

    elif model_type == "transformer":
        return StandardTransformerInterdictionModel(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,dropout=0.1
        ).to(device)

    elif model_type == "gnn":
        return GNNInterdictionModel(
            input_dim=input_dim,d_model=64,num_layers=2
        ).to(device)
    
    elif model_type == "edge_transformer":
        return EdgeBiasTransformerInterdictionModel(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,dropout=0.1
        ).to(device)
    
    elif model_type == "tropical_v2":
        return TropicalInterdictionModelV2(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,device=device,dropout=0.1,use_edge_bias=True
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")



def train():
    
    '''defines main training function'''

    seed = 1

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    
    # uses gpu if available, otherwise cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # creates tropical attention interdiction model and moves it to GPU/CPU
    model_type = sys.argv[1] if len(sys.argv) > 1 else "tropical"
    problem_type = sys.argv[2] if len(sys.argv) > 2 else "shortest_path"

    # load in solved training MIP data
    dataset_file = f"training_data_{problem_type}.json"
    dataset = InterdictionDataset(dataset_file)
    
    # Group sample indices by underlying graph seed.
    seed_to_indices = {}

    for index, sample in enumerate(dataset.data):
        graph_seed = sample["graph_seed"]

        if graph_seed not in seed_to_indices:
            seed_to_indices[graph_seed] = []

        seed_to_indices[graph_seed].append(index)


    # Shuffle graph seeds reproducibly.
    graph_seeds = list(seed_to_indices.keys())

    split_rng = random.Random(1)
    split_rng.shuffle(graph_seeds)


    # Split at the graph level: 70% of graphs for training, 15% for validation, 15% for internal testing.
    num_graphs = len(graph_seeds)

    num_train_graphs = int(0.70 * num_graphs)
    num_val_graphs = int(0.15 * num_graphs)

    train_seeds = set(graph_seeds[:num_train_graphs])

    val_seeds = set(graph_seeds[num_train_graphs: num_train_graphs + num_val_graphs])

    test_seeds = set(graph_seeds[num_train_graphs + num_val_graphs:])

    # Collect every attack-budget sample belonging to each graph.
    train_indices = sorted(index for seed in train_seeds for index in seed_to_indices[seed])

    val_indices = sorted(index for seed in val_seeds for index in seed_to_indices[seed])

    test_indices = sorted(index for seed in test_seeds for index in seed_to_indices[seed])

    split_path = f"results/data_split_{problem_type}.pt"
    
    torch.save({"split_seed": seed,
            "train_seeds": sorted(train_seeds),
            "val_seeds": sorted(val_seeds),
            "test_seeds": sorted(test_seeds),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,},split_path,)

    # Create PyTorch subset objects.
    train_dataset = Subset(dataset,train_indices)

    val_dataset = Subset(dataset,val_indices)

    # Reserved for future in-distribution evaluation.
    test_dataset = Subset(dataset,test_indices)


    # creates batches for training - 4 graphs per batch, shuffle data
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
        collate_fn=collate_graphs,generator=loader_generator,) # use padding function

    # creates batches for validation - 4 graphs per batch, do not shuffle data
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
        collate_fn=collate_graphs) # use padding function

    model_config = {"input_dim": PROBLEM_INPUT_DIMS[problem_type],"d_model": 64,
                    "n_heads": 4,"num_layers": 2,"dropout": 0.1,
                    "use_edge_bias": model_type in {"tropical","tropical_v2","edge_transformer",},
}

    model = get_model(model_type, problem_type, device)

    num_positive = 0
    num_negative = 0

    for index in train_indices:

        attack_labels = dataset.data[index]["attack"]

        sample_positive = sum(int(label) for label in attack_labels)

        sample_total = len(attack_labels)

        num_positive += sample_positive
        num_negative += (sample_total - sample_positive)

    if num_positive == 0:
        raise RuntimeError("Training set contains no positive attack labels.")

    calculated_pos_weight = (num_negative / num_positive)

    pos_weight = torch.tensor([calculated_pos_weight],dtype=torch.float32,device=device,)

    # creates binary loss function 
    # compares predicted edge logits to MIP attack labels
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    learning_rate = 1e-3
    optimizer = schedulefree.RAdamScheduleFree(model.parameters(),lr=learning_rate,weight_decay=1e-4)

    run_name = f"{model_type}_{problem_type}"

    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = None
    best_model_path = f"saved_models/{run_name}_best_model.pt"
    final_model_path = f"saved_models/{run_name}_final_model.pt"

    # number of training epochs
    epochs = 50

    # starts tracking total training time
    training_start_time = time.perf_counter()

    # initializes list to store epoch results for csv output
    epoch_rows = []

    # TRAINING LOOP
    # starts training loop
    for epoch in range(epochs):

        # starts tracking epoch time
        epoch_start_time = time.perf_counter()
        
        # puts model and optimizerin training mode
        model.train()
        optimizer.train()
        # starts tracking training loss
        total_train_loss = 0.0
        total_train_edges = 0
        total_gradient_norm = 0.0
        max_gradient_norm = 0.0
        num_gradient_updates = 0

        # loops through training batches
        for edge_features, edge_bias, y, mask, attack_limits in train_loader:
            # moves everything to GPU / CPU
            edge_features = edge_features.to(device,dtype=torch.float32)
            edge_bias = edge_bias.to(device,dtype=torch.float32)
            y = y.to(device,dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.bool)
            attack_limits = attack_limits.to(device)

            # clears old gradients
            optimizer.zero_grad(set_to_none=True)

            # gets one prediction score per edge
            logits = model(edge_features, edge_bias=edge_bias, mask=mask)

            # computes loss for every edge
            loss_matrix = loss_fn(logits, y)

            # keeps only real edges, ignores padded edges, and averages the loss 
            # loss = loss_matrix[mask].mean()

            valid_edge_losses = loss_matrix[mask]

            if valid_edge_losses.numel() == 0:
                continue 

            loss = valid_edge_losses.mean()

            # computes gradients
            loss.backward()

            # Optional but recommended protection against unstable gradients.
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           max_norm=1.0)

            gradient_norm_value = float(gradient_norm)

            total_gradient_norm += gradient_norm_value

            max_gradient_norm = max(max_gradient_norm,gradient_norm_value,)

            num_gradient_updates += 1
            
            # updates model weights
            optimizer.step()

            num_valid_edges = mask.sum().item()

            # add batch's loss to epoch total
            total_train_loss += loss.item() * num_valid_edges
            total_train_edges += num_valid_edges

        if total_train_edges == 0:
            raise RuntimeError("No valid training edges were found during this epoch.")

        if num_gradient_updates == 0:
            raise RuntimeError("No gradient updates occurred during this epoch.")

        avg_gradient_norm = (total_gradient_norm / num_gradient_updates)


        # VALIDATION LOOP
        # puts model in evaluation mode
        model.eval()
        optimizer.eval()
        # starts tracking validation loss
        total_val_loss = 0.0
        total_val_edges = 0
        
        # starts tracking metrics
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_hamming = 0.0
        total_exact_match = 0.0

        num_val_samples = 0

        # turns off gradient calculation for validation
        with torch.no_grad(): 
            # loops through validation batches
            for edge_features, edge_bias, y, mask, attack_limits in val_loader:
                # moves validation data to GPU / CPU
                edge_features = edge_features.to(device,dtype=torch.float32)
                edge_bias = edge_bias.to(device,dtype=torch.float32)
                y = y.to(device,dtype=torch.float32)
                mask = mask.to(device=device, dtype=torch.bool)
                attack_limits = attack_limits.to(device)

                # predicts edge interdiction scores
                logits = model(edge_features, edge_bias=edge_bias, mask=mask)

                # computes validation loss for every edge
                loss_matrix = loss_fn(logits, y)

                valid_edge_losses = loss_matrix[mask]

                if valid_edge_losses.numel() == 0:
                    continue

                loss = valid_edge_losses.mean()
                num_valid_edges = mask.sum().item()
               
                # ignores padded edges
                # loss = loss_matrix[mask].mean()
                
                # adds validation loss
                total_val_loss += loss.item() * num_valid_edges
                total_val_edges += num_valid_edges

                # call compute metrics function from metrics file
                accuracy, precision, recall, f1, hamming, exact_match = compute_metrics(
                    logits=logits, y=y, mask=mask, attack_limits=attack_limits)

                batch_size = edge_features.size(0)

                # increment all metrics appropriately
                total_accuracy += accuracy * batch_size
                total_precision += precision * batch_size
                total_recall += recall * batch_size
                total_f1 += f1 * batch_size
                total_hamming += hamming * batch_size
                total_exact_match += exact_match * batch_size

                num_val_samples += batch_size


        if total_val_edges == 0 or num_val_samples == 0:
            raise RuntimeError("No valid validation edges were found during this epoch.")

        # finishes tracking epoch time
        epoch_end_time = time.perf_counter()

        # calculates total epoch time in seconds
        epoch_time = epoch_end_time - epoch_start_time

        # calculates average losses and metrics for the epoch
        avg_train_loss = total_train_loss / total_train_edges 
        avg_val_loss = total_val_loss / total_val_edges 
        avg_accuracy = total_accuracy / num_val_samples
        avg_precision = total_precision / num_val_samples
        avg_recall = total_recall / num_val_samples
        avg_f1 = total_f1 / num_val_samples
        avg_hamming = total_hamming / num_val_samples
        avg_exact_match = total_exact_match / num_val_samples


        is_best_epoch = avg_val_loss < best_val_loss

        if is_best_epoch:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

            torch.save({
                "epoch": epoch + 1,
                "model_type": model_type,
                "problem_type": problem_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_f1": avg_f1,
                "val_exact_match": avg_exact_match,
                "learning_rate": learning_rate,
                "split_path": split_path,
                "seed": seed,}, best_model_path)

            print(f"Saved new best model at epoch {best_epoch} "
            f"with validation loss {best_val_loss:.6f}")

        # stores epoch results for csv output
        epoch_rows.append({
            "model_type": model_type,
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "accuracy": avg_accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "hamming": avg_hamming,
            "exact_match": avg_exact_match,
            "average_gradient_norm": avg_gradient_norm,
            "maximum_gradient_norm": max_gradient_norm,
            "best_epoch_so_far": is_best_epoch,
            "epoch_time_seconds": epoch_time})

        # prints progress
        print(f"Epoch {epoch + 1:03d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Acc: {avg_accuracy:.4f} | "
              f"Prec: {avg_precision:.4f} | "
              f"Rec: {avg_recall:.4f} | "
              f"F1: {avg_f1:.4f} | "
              f"Hamming: {avg_hamming:.2f} | "
              f"Exact: {avg_exact_match:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
    # finishes tracking total training time
    training_end_time = time.perf_counter()

    # calculates total training time in seconds
    total_training_time = training_end_time - training_start_time

    model.eval()
    optimizer.eval()

    torch.save({
        "epoch": epochs,
        "model_type": model_type,
        "problem_type": problem_type,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": avg_val_loss,
        "val_f1": avg_f1,
        "val_exact_match": avg_exact_match,
        "learning_rate": learning_rate,
        "split_path": split_path,
        "seed": seed,},final_model_path)


    print(f"\nTotal training time for {run_name}: "
           f"{total_training_time:.2f} seconds")
    print(f"Average epoch time: "
          f"{total_training_time / epochs:.2f} seconds")
    print(f"Best epoch: {best_epoch} | "
          f"Best validation loss: {best_val_loss:.6f}")

    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")

    
    run_name = f"{model_type}_{problem_type}"

    # saves epoch results to csv file for later analysis
    with open(f"results/training_log_{run_name}.csv", "w", newline="") as csvfile:
        fieldnames = epoch_rows[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)

    # saves training summary to csv file for later analysis
    with open(f"results/training_summary_{run_name}.csv", "w", newline="") as csvfile:
        fieldnames = ["model_type","problem_type","epochs","best_epoch","best_val_loss",
                      "final_val_loss","total_training_time_seconds","average_epoch_time_seconds",
                      "best_model_path","final_model_path",]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
        "model_type": model_type,
        "problem_type": problem_type,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": avg_val_loss,
        "total_training_time_seconds": total_training_time,
        "average_epoch_time_seconds": total_training_time / epochs,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    })



if __name__ == "__main__":
    train()
