# -*- coding: utf-8 -*-
"""
Created on Thu May 14 07:09:23 2026

@author: emmallen
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from Tropical_Attention import TropicalInterdictionModel

class InterdictionDataset(Dataset):
    
    ''' class that creates a custom PyTorch dataset for the interdiction graphs'''
    
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # get graph at index idx
        sample = self.data[idx]

        # store number of nodes
        n = sample["n_nodes"]
        # store density = # edges (m)  / # nodes (n)
        density = sample["density"]
        # store interdiction budget
        attack_limit = sample["attack_limit"]

        # converts edge heads to PyTorch tensor
        u = torch.tensor(sample["u"], dtype=torch.float32)
        # converts edge tails to PyTorch tensor
        v = torch.tensor(sample["v"], dtype=torch.float32)
        # converts edge distances to PyTorch tensor
        dist = torch.tensor(sample["dist"], dtype=torch.float32)
        # converts MIP attack labels to PyTorch tensor
        y = torch.tensor(sample["attack"], dtype=torch.float32)

        # Normalize node IDs and distances
        u_norm = u / max(n - 1, 1)
        v_norm = v / max(n - 1, 1)
        dist_norm = dist / 10.0

        # source flag = 1 if edge leaves source node
        source_flag = (u == sample["source"]).float()
        # sink flag = 1 if edge leaves sink node
        sink_flag = (v == sample["sink"]).float()

        # creates one density valye for every edge
        density_feature = torch.full_like(u_norm, density / 10.0)
        # creates one budget value for every edge
        budget_feature = torch.full_like(u_norm, attack_limit / 10.0)

        # combines edge features into one matrix
        edge_features = torch.stack([
            u_norm, v_norm, dist_norm, source_flag, sink_flag, density_feature,
            budget_feature], dim=1)

        # edge bias: edge i can flow into edge j if v_i == u_j
        # creates edge to edge connectivity matrix
        edge_bias = (v.unsqueeze(1) == u.unsqueeze(0)).float()

        # gives connected edge pairs a positive attention bias
        edge_bias = 0.5 * edge_bias

        return edge_features, edge_bias, y


def collate_graphs(batch):
    
    '''defines how to combine multiple graphs into one batch, graphs have different
    number of edges, so they need padding'''
    
    # finds largest number of edges in batch
    max_edges = max(item[0].shape[0] for item in batch)
    # gets # of edge features
    input_dim = batch[0][0].shape[1]

    # creates empty lists to store padded graphs
    edge_features_batch = []
    edge_bias_batch = []
    y_batch = []
    mask_batch = []

    # loops through each graph in the batch
    for edge_features, edge_bias, y in batch:
        # stores the number of real edges in the graph
        m = edge_features.shape[0]

        # creates a zero matrix for padded edge features
        edge_features_padded = torch.zeros(max_edges, input_dim)
        # copies real edges into the top part of matrix
        edge_features_padded[:m] = edge_features
        
        # creates padded edge bias matrix
        edge_bias_padded = torch.zeros(max_edges, max_edges)
        # copies real edge-bias values inot it
        edge_bias_padded[:m, :m] = edge_bias

        # creates padded labels
        y_padded = torch.zeros(max_edges)
        # copies real labels
        y_padded[:m] = y

        # creates a mask saying which edges are real
        mask = torch.zeros(max_edges, dtype=torch.bool)
        # marks real edges as true
        mask[:m] = True

        # adds the padded graph to the batch lists
        edge_features_batch.append(edge_features_padded)
        edge_bias_batch.append(edge_bias_padded)
        y_batch.append(y_padded)
        mask_batch.append(mask)


    # stacks all graphs into batch tensors
    # edge_features: [batch_size, max_edges, 7]
    # edge_bias: [batch_size, max_edges, max_edges]
    # y:[batch_size, max_edges]
    # mask: [batch_size, max_edges]
    return (torch.stack(edge_features_batch),
        torch.stack(edge_bias_batch),
        torch.stack(y_batch),
        torch.stack(mask_batch))

def compute_metrics(logits, y, mask, attack_limit=1):
    """
    Computes edge-level and graph-level prediction metrics.
    """

    batch_size = logits.shape[0]

    total_correct = 0
    total_edges = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_hamming = 0
    exact_match = 0

    for b in range(batch_size):
        real_logits = logits[b][mask[b]]
        real_y = y[b][mask[b]]

        # Pick top-k edges as interdicted
        k = attack_limit
        pred = torch.zeros_like(real_y)

        topk_indices = torch.topk(real_logits, k=k).indices
        pred[topk_indices] = 1.0

        # Edge accuracy
        total_correct += (pred == real_y).sum().item()
        total_edges += real_y.numel()

        # Precision / recall / F1 pieces
        tp = ((pred == 1) & (real_y == 1)).sum().item()
        fp = ((pred == 1) & (real_y == 0)).sum().item()
        fn = ((pred == 0) & (real_y == 1)).sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Hamming distance
        total_hamming += (pred != real_y).sum().item()

        # Exact interdiction set match
        if torch.equal(pred, real_y):
            exact_match += 1

    accuracy = total_correct / total_edges

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    avg_hamming = total_hamming / batch_size
    exact_match_rate = exact_match / batch_size

    return accuracy, precision, recall, f1, avg_hamming, exact_match_rate




def train():
    
    '''defines main training function'''
    
    # uses gpu if available, otherwise cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load in solved training MIP data
    dataset = InterdictionDataset("training_data.json")

    # use 80% for training
    train_size = int(0.8 * len(dataset))
    # use remaining 20% for validation
    val_size = len(dataset) - train_size

    # randomly splits the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # creates batches for training - 4 graphs per batch, shuffle data
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
        collate_fn=collate_graphs) # use padding function

    # creates batches for validation - 4 graphs per batch, do not shuffle data
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
        collate_fn=collate_graphs) # use padding function

    # creates tropical attentino interdiction model and moves it to GPU/CPU
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
        for edge_features, edge_bias, y, mask in train_loader:
            # moves everything to GPU / CPU
            edge_features = edge_features.to(device)
            edge_bias = edge_bias.to(device)
            y = y.to(device)
            mask = mask.to(device)

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
        
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_hamming = 0.0
        total_exact_match = 0.0

        # turns off gradient calculation for validation
        with torch.no_grad(): 
            # loops through validation batches
            for edge_features, edge_bias, y, mask in val_loader:
                # moves validation data to GPU / CPU
                edge_features = edge_features.to(device)
                edge_bias = edge_bias.to(device)
                y = y.to(device)
                mask = mask.to(device)

                # predicts edge interdiction scores
                logits = model(edge_features, edge_bias=edge_bias, mask=mask)

                # computes validation loss for every edge
                loss_matrix = loss_fn(logits, y)
                # ignores padded edges
                loss = loss_matrix[mask].mean()
                # adds validation loss
                total_val_loss += loss.item()
                
                
                accuracy, precision, recall, f1, hamming, exact_match = compute_metrics(
                    logits=logits, y=y, mask=mask, attack_limit=1)

                total_accuracy += accuracy
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_hamming += hamming
                total_exact_match += exact_match

        # prints progresss
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