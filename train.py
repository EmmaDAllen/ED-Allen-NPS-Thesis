# -*- coding: utf-8 -*-
"""
Created on Thu May 14 07:09:23 2026

@author: emmallen
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.tropical_attention import TropicalInterdictionModel
from data.interdiction_data import InterdictionDataset
from data.interdiction_data import collate_graphs
from training.metrics import compute_metrics


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
    model = TropicalInterdictionModel(input_dim=7, d_model=64, n_heads=4,
        num_layers=2, device=device).to(device)

    # Class imbalance correction
    # weights positive labels more heavily - most edges are not inerdicted
    pos_weight = torch.tensor([100.0], device=device)
    # creates binary loss function 
    # compares predicted edge logits to MIP attack labels
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    
    # creates optimizer that updates model weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 50

    # starts training loop
    for epoch in range(epochs):
        
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

        # prints progress
        print(f"Epoch {epoch + 1:03d} | "
              f"Train Loss: {total_train_loss / len(train_loader):.4f} | "
              f"Val Loss: {total_val_loss / len(val_loader):.4f} | "
              f"Acc: {total_accuracy / len(val_loader):.4f} | "
              f"Prec: {total_precision / len(val_loader):.4f} | "
              f"Rec: {total_recall / len(val_loader):.4f} | "
              f"F1: {total_f1 / len(val_loader):.4f} | "
              f"Hamming: {total_hamming / len(val_loader):.2f} | "
              f"Exact: {total_exact_match / len(val_loader):.4f}")

    # saves training model weights
    torch.save(model.state_dict(), "tropical_interdiction_model.pt")


if __name__ == "__main__":
    train()