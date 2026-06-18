# -*- coding: utf-8 -*-
"""
Created on Thu May 14 07:09:23 2026

@author: emmallen
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import csv

from models.tropical_attention import TropicalInterdictionModel
from models.standard_transformer import StandardTransformerInterdictionModel
from models.gnn import GNNInterdictionModel
from models.edge_bias_transformer import EdgeBiasTransformerInterdictionModel
from models.tropical_attention_V2 import TropicalInterdictionModel as TropicalInterdictionModelV2
from data.interdiction_data import InterdictionDataset
from data.interdiction_data import collate_graphs
from evaluation.metrics import compute_metrics

def get_model(model_type, device):

    if model_type == "tropical":
        return TropicalInterdictionModel(
            input_dim=8,
            d_model=64,
            n_heads=4,
            num_layers=2,
            device=device
        ).to(device)

    elif model_type == "transformer":
        return StandardTransformerInterdictionModel(
            input_dim=8,
            d_model=64,
            n_heads=4,
            num_layers=2,
        ).to(device)

    elif model_type == "gnn":
        return GNNInterdictionModel(
            input_dim=8,
            d_model=64,
            num_layers=2
        ).to(device)
    
    elif model_type == "edge_transformer":
        return EdgeBiasTransformerInterdictionModel(
            input_dim=8,
            d_model=64,
            n_heads=4,
            num_layers=2
        ).to(device)
    
    elif model_type == "tropical_v2":
        return TropicalInterdictionModelV2(
            input_dim=8,
            d_model=64,
            n_heads=4,
            num_layers=2,
            device=device
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train():
    
    '''defines main training function'''
    
    # uses gpu if available, otherwise cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load in solved training MIP data
    dataset = InterdictionDataset("training_data.json")
    
    # use 70% for training
    train_size = int(0.7 * len(dataset))
    # use 15% for validation
    val_size = int(0.15 * len(dataset))
    # use 15% for testing
    test_size = len(dataset) - train_size - val_size

    # randomly splits the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset,
    [train_size, val_size, test_size], generator=torch.Generator().manual_seed(1))


    # creates batches for training - 4 graphs per batch, shuffle data
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
        collate_fn=collate_graphs) # use padding function

    # creates batches for validation - 4 graphs per batch, do not shuffle data
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
        collate_fn=collate_graphs) # use padding function

    # creates tropical attention interdiction model and moves it to GPU/CPU
    model_type = sys.argv[1] if len(sys.argv) > 1 else "tropical"

    model = get_model(model_type, device)

    # Class imbalance correction
    # weights positive labels more heavily - most edges are not inerdicted
    pos_weight = torch.tensor([100.0], device=device)
    # creates binary loss function 
    # compares predicted edge logits to MIP attack labels
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    
    # creates optimizer that updates model weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # number of training epochs
    epochs = 50

    # starts tracking total training time
    training_start_time = time.perf_counter()

    # initializes list to store epoch results for csv output
    epoch_rows = []

    # starts training loop
    for epoch in range(epochs):

        # starts tracking epoch time
        epoch_start_time = time.perf_counter()
        
        # puts model in training mode
        model.train()
        # starts tracking training loss
        total_train_loss = 0.0

        # loops through training batches
        for edge_features, edge_bias, y, mask, attack_limits in train_loader:
            # moves everything to GPU / CPU
            edge_features = edge_features.to(device)
            edge_bias = edge_bias.to(device)
            y = y.to(device)
            mask = mask.to(device)
            attack_limits = attack_limits.to(device)

            # clears old gradients
            optimizer.zero_grad()

            # gets one prediction score per edge
            logits = model(edge_features, edge_bias=edge_bias, mask=mask)

            # computes loss for every edge
            loss_matrix = loss_fn(logits, y)
            # keeps only real edges, ignores padded edges, and averages the loss 
            loss = loss_matrix[mask].mean()
            
            # computes gradients
            loss.backward()
            # updates model weights
            optimizer.step()

            # add batch's loss to epoch total
            total_train_loss += loss.item()

        # puts model in evaluation mode
        model.eval()
        # starts tracking validation loss
        total_val_loss = 0.0
        
        # starts tracking metrics
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_hamming = 0.0
        total_exact_match = 0.0

        # turns off gradient calculation for validation
        with torch.no_grad(): 
            # loops through validation batches
            for edge_features, edge_bias, y, mask, attack_limits in val_loader:
                # moves validation data to GPU / CPU
                edge_features = edge_features.to(device)
                edge_bias = edge_bias.to(device)
                y = y.to(device)
                mask = mask.to(device)
                attack_limits = attack_limits.to(device)

                # predicts edge interdiction scores
                logits = model(edge_features, edge_bias=edge_bias, mask=mask)

                # computes validation loss for every edge
                loss_matrix = loss_fn(logits, y)
                # ignores padded edges
                loss = loss_matrix[mask].mean()
                # adds validation loss
                total_val_loss += loss.item()
                
                # call compute metrics function from metrics file
                accuracy, precision, recall, f1, hamming, exact_match = compute_metrics(
                    logits=logits, y=y, mask=mask, attack_limits=attack_limits)

                # increment all metrics appropriately
                total_accuracy += accuracy
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_hamming += hamming
                total_exact_match += exact_match

        # finishes tracking epoch time
        epoch_end_time = time.perf_counter()

        # calculates total epoch time in seconds
        epoch_time = epoch_end_time - epoch_start_time

        # calculates average losses and metrics for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        avg_precision = total_precision / len(val_loader)
        avg_recall = total_recall / len(val_loader)
        avg_f1 = total_f1 / len(val_loader)
        avg_hamming = total_hamming / len(val_loader)
        avg_exact_match = total_exact_match / len(val_loader)

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

    print(f"\nTotal training time for {model_type}: "
          f"{total_training_time:.2f} seconds")
    print(f"Average epoch time: "
          f"{total_training_time / epochs:.2f} seconds")
    
    # saves epoch results to csv file for later analysis
    with open(f"results/training_log_{model_type}.csv", "w", newline="") as csvfile:
        fieldnames = epoch_rows[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)

    # saves training summary to csv file for later analysis
    with open(f"results/training_summary_{model_type}.csv", "w", newline="") as csvfile:
        fieldnames = ["model_type","epochs", "total_training_time_seconds",
        "average_epoch_time_seconds"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "model_type": model_type,
            "epochs": epochs,
            "total_training_time_seconds": total_training_time,
            "average_epoch_time_seconds": total_training_time / epochs
    })

    # saves training model weights
    torch.save(model.state_dict(), f"saved_models/{model_type}_model.pt")


if __name__ == "__main__":
    train()
