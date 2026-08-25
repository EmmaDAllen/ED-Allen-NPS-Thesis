# -*- coding: utf-8 -*-
"""
Created on Thu May 14 07:09:23 2026

@author: emmallen
"""

"""train.py

Training pipeline for all network interdiction models.

This script trains the Tropical Attention Transformer, Tropical Attention Transformer V2,
Standard Transformer, Edge-Bias Transformer, and GNN baseline on shortest-path, maximum-flow, 
and minimum-cost-flow interdiction datasets.

Usage
Run from the repository root with:

    PYTHONPATH=. python experiments/train.py MODEL_TYPE PROBLEM_TYPE

    
The script:
1. Selects the requested model and interdiction problem.
2. Loads the corresponding MIP-labeled dataset.
3. Splits the dataset by graph seed so all attack-budget versions of the same graph remain
 in the same partition.
4. Creates training and validation DataLoaders.
5. Calculates a positive-class weight from the training labels.
6. Trains the selected model for a fixed number of epochs.
7. Evaluates validation loss and prediction metrics after every epoch.
8. Saves the best-validation-loss model and the final model.
9. Saves per-epoch and run-level training summaries."""

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

# number of edge-level input features produced for each problem type.
# shortest_path: uses 8 edge/global features (cost, penalties, no capacities)
# max_flow: Uses 7 edge/global features (capacities, no cost or penalties)
# min_cost_flow: Uses 9 edge/global features (cost, penalties, and capacities)
# these values must match the feature vectors created by the dataset preprocessing code
PROBLEM_INPUT_DIMS = {
    "shortest_path": 8,
    "max_flow": 7,
    "min_cost_flow": 9}


def get_model(model_type, problem_type, device):

    """Construct the requested neural network architecture.

    Parameters:
    model_type : str
        Model architecture requested on the command line. 

    problem_type : str
        Network-interdiction problem being modeled.

    device : str
        Device on which the model parameters should be stored. (CPU or CUDA)

    Returns
    torch.nn.Module
        Initialized model on the requested device."""

    # confirm that the selected problem has a known feature dimension
    # without this check, the input projection would be constructed with
    # an undefined number of input features
    if problem_type not in PROBLEM_INPUT_DIMS:
        raise ValueError(f"Unknown problem type: {problem_type}")

    # retrieve the number of edge features used by the selected interdiction problem
    input_dim = PROBLEM_INPUT_DIMS[problem_type]

    # all Transformer-style models use the same hidden dimension, attention-head count, 
    # number of layers, and dropout probability = holding these settings constant makes 
    # the model comparison more controlled.

    # tropical attention model with edge bias
    if model_type == "tropical":
        # construct Tropical Attention Transformer
        return TropicalInterdictionModel(
            input_dim=input_dim,d_model=64,n_heads=4,num_layers=2,dropout=0.1,device=device,use_edge_bias=True
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

    # edge bias transformer model
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




def train():
    
    """Train one interdiction model.

    The function performs:

    1. Random seed initialization
    2. Dataset loading
    3. Graph-level train/validation/test split
    4. DataLoader construction
    5. Model initialization
    6. Model training
    7. Validation
    8. Checkpointing
    9. Metric logging
    
    The model and problem are selected from command-line arguments.
    When no arguments are supplied, the script defaults to the Version 1
    tropical model and shortest-path interdiction."""



    # REPRODUCIBILITY

    # use one seed for Python randomness, PyTorch initialization,
    # CUDA operations, DataLoader shuffling, and the graph split
    seed = 1

    # control Python's built-in random-number generator
    random.seed(seed)
    # control PyTorch random operations performed on the CPU,
    # including initial model parameter values
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # seed random operations on the current CUDA device
        torch.cuda.manual_seed(seed)
        # seed every visible CUDA device = matters when the system
        # has multiple GPUs, even if this run uses only one
        torch.cuda.manual_seed_all(seed)

    # create a dedicated random-number generator for the training
    # DataLoader = keeps batch shuffling reproducible
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)



    # DEVICE AND COMMAND-LINE CONFIGURATION
 
    # uses cude gpu if available, otherwise cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sys.argv[0] contains the script name
    # sys.argv[1], when supplied, is the requested model type, default = tropical
    model_type = sys.argv[1] if len(sys.argv) > 1 else "tropical"
    # sys.argv[2], when supplied, is the interdiction problem type, deafult = shortest path
    problem_type = sys.argv[2] if len(sys.argv) > 2 else "shortest_path"
    # sys.argv[3] optionally specifies an experiment tag used to distinguish retrained
    # models from previous runs and prevent checkpoints/results from being overwritten
    experiment_tag = sys.argv[3] if len(sys.argv) > 3 else None



    # DATASET LOADING 

    # choose the solved training-data file corresponding to the requested problem
    dataset_file = f"training_data_{problem_type}.json"

    # load the JSON data through the custom PyTorch Dataset class - each item represents one 
    # graph under one attack budget and includes edge features, edge bias, attack labels, and a mask
    dataset = InterdictionDataset(dataset_file)



    # GRAPH-LEVEL TRAIN/VALIDATION/TEST SPLIT
    
    # The same underlying graph appears multiple times in the dataset because it is solved under 
    # multiple attack budgets - grouping indices by graph_seed ensures all budget variants of one 
    # graph remain in the same partition, otherwise, the model could train on one budget for a graph 
    # and validate on another budget for that exact same graph, causing information leakage
    seed_to_indices = {}

    # enumerate every sample stored in the dataset
    for index, sample in enumerate(dataset.data):

        # graph_seed identifies the underlying randomly generated graph
        graph_seed = sample["graph_seed"]

        # create an empty list the first time this graph is encountered
        if graph_seed not in seed_to_indices:
            seed_to_indices[graph_seed] = []

        # record this dataset row as one sample belonging to this graph
        seed_to_indices[graph_seed].append(index)


    # extract one entry per unique underlying graph
    graph_seeds = list(seed_to_indices.keys())

    # use a separate Random object so the graph split is reproducible
    # and does not depend on random operations performed elsewhere
    split_rng = random.Random(1)

    # randomize the order of unique graph identifiers before splitting
    split_rng.shuffle(graph_seeds)

    # count the number of unique underlying graphs, not the number of attack-budget samples
    num_graphs = len(graph_seeds)

    # assign 70% of unique graphs to training - int() rounds down, so any leftover graphs
    # are ultimately assigned to the test set
    num_train_graphs = int(0.70 * num_graphs)
    # assign 15% of unique graphs to validation
    num_val_graphs = int(0.15 * num_graphs)

    # select the first 70% of shuffled graph seeds for training
    train_seeds = set(graph_seeds[:num_train_graphs])

    # select the next 15% of graph seeds for validation
    val_seeds = set(graph_seeds[num_train_graphs: num_train_graphs + num_val_graphs])

    # assign all remaining graph seeds to the internal test set
    test_seeds = set(graph_seeds[num_train_graphs + num_val_graphs:])

    # collect every dataset index associated with a training graph = includes all 
    # attack-budget versions of every selected graph
    train_indices = sorted(index for seed in train_seeds for index in seed_to_indices[seed])

    # collect every dataset index associated with a validation graph
    val_indices = sorted(index for seed in val_seeds for index in seed_to_indices[seed])

    # collect every dataset index associated with a test graph
    test_indices = sorted(index for seed in test_seeds for index in seed_to_indices[seed])

    # include the optional experiment tag in the run name so evaluation loads the
    # checkpoint associated with the correct training run
    if experiment_tag:
        split_path = f"results/data_split_{problem_type}_{experiment_tag}.pt"
    else:
        split_path = f"results/data_split_{problem_type}.pt"

    # ensure the results directory exists before attempting to save the split file
    os.makedirs("results", exist_ok=True)

    # save both the graph seeds and the resulting dataset indices - reusing this file allows 
    # all model architectures to use exactly the same graph partitions
    torch.save({"split_seed": seed,
            "train_seeds": sorted(train_seeds),
            "val_seeds": sorted(val_seeds),
            "test_seeds": sorted(test_seeds),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,},split_path,)

    # create a Dataset view containing only training samples = Subset does not copy the original 
    # dataset; it stores the selected indices and retrieves those rows from dataset when requested.
    train_dataset = Subset(dataset,train_indices)

    # create a Dataset view containing only validation samples
    val_dataset = Subset(dataset,val_indices)


    # reserve the held-out in-distribution test set
    # it is not used during training or validation, but keeping it separate
    # prevents the test results from influencing model selection
    test_dataset = Subset(dataset,test_indices)



    # DATALOADER CONSTRUCTION

    # Construct training mini-batches containing four graph samples
    # shuffle=True: Changes the sample order at the beginning of every epoch
    # collate_graphs: Pads variable-size graphs to the largest edge count in the batch
    # and returns the corresponding validity mask
    # generator: Makes the shuffle order reproducible
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
        collate_fn=collate_graphs,generator=loader_generator,) # use padding function

    # Construct validation batches.
    # shuffle=False preserves the same validation order each epoch
    # shuffling is unnecessary because no gradient updates occur
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
        collate_fn=collate_graphs) # use padding function



    # MODEL CONSTRUCTION

    # record the architecture settings used for this run
    model_config = {"input_dim": PROBLEM_INPUT_DIMS[problem_type],"d_model": 64,
                    "n_heads": 4,"num_layers": 2,"dropout": 0.1,
                    "use_edge_bias": model_type in {"tropical","tropical_v2","edge_transformer",},}

    # initialize the selected model and move all model parameters onto the selected CPU or GPU device
    model = get_model(model_type, problem_type, device)



    # POSITIVE-CLASS WEIGHTING

    # interdicted edges are rare relative to non-interdicted edges = count both classes in the 
    # training partition so the loss can give greater weight to positive attack labels
    num_positive = 0
    num_negative = 0

    # examine only samples assigned to the training split
    for index in train_indices:

        # retrieve the binary MIP attack labels for this graph sample
        # each entry is 1 if the MIP selected that edge for interdiction and 0 otherwise
        attack_labels = dataset.data[index]["attack"]

        # count attacked edges in this sample
        sample_positive = sum(int(label) for label in attack_labels)

        # count all real edges in this graph sample
        sample_total = len(attack_labels)

        # add this graph's positive labels to the training-set total
        num_positive += sample_positive

        # every non-positive label is a negative class example
        num_negative += (sample_total - sample_positive)

    # BCEWithLogitsLoss cannot construct a meaningful positive-class
    # weight when no training label is positive
    if num_positive == 0:
        raise RuntimeError("Training set contains no positive attack labels.")

    # PyTorch's pos_weight multiplies the loss contribution of positive labels
    # using negatives / positives compensates for the observed training-set class imbalance
    calculated_pos_weight = (num_negative / num_positive)

    # BCEWithLogitsLoss expects pos_weight to be a tensor = place it on the same device as 
    # the logits and labels
    pos_weight = torch.tensor([calculated_pos_weight],dtype=torch.float32,device=device,)




    # LOSS AND OPTIMIZER

    # BCEWithLogitsLoss combines sigmoid(logit) and binary cross-entropy in one numerically 
    # stable operation, reduction="none" returns one loss value per edge = required
    # because padded edge positions must be removed before averaging
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    # initial optimizer learning rate
    learning_rate = 1e-3

    # Schedule-Free RAdam performs adaptive gradient updates without a separate learning-rate
    # scheduler, weight_decay=1e-4 applies L2-style parameter regularization
    optimizer = schedulefree.RAdamScheduleFree(model.parameters(),lr=learning_rate,weight_decay=1e-4)


    # include the optional experiment tag in the run name so evaluation loads the
    # checkpoint associated with the correct training run
    if experiment_tag:
        run_name = f"{model_type}_{problem_type}_{experiment_tag}"
    else:
        run_name = f"{model_type}_{problem_type}"


    # create directories for metrics and model checkpoints
    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    # start with an infinite best validation loss so the first valid epoch is automatically 
    # considered an improvement
    best_val_loss = float("inf")

    # store the one-indexed epoch number of the best checkpoint
    best_epoch = None

    # path for the checkpoint with the lowest observed validation loss
    best_model_path = f"saved_models/{run_name}_best_model.pt"

    # path for the checkpoint after the final training epoch
    final_model_path = f"saved_models/{run_name}_final_model.pt"

    # train for a fixed number of complete passes through the training dataset
    epochs = 50

    # record wall-clock time immediately before training begins
    training_start_time = time.perf_counter()

    # store one dictionary of metrics for every epoch = rows will later be written to a CSV file
    epoch_rows = []



    # TRAINING/EPOCH LOOP

    # starts training loop
    for epoch in range(epochs):

        # record the beginning of this epoch so training and validation
        # time can be measured together
        epoch_start_time = time.perf_counter()
        
        # Enable training behavior in the model = 
        # activates dropout and causes any other training-specific layers to use their training behavior
        model.train()

        # Schedule-Free optimizers maintain separate train/evaluation parameter representations
        # optimizer.train() switches to the parameter state used for optimization
        optimizer.train()

        # sum of batch losses weighted by the number of real edges - later be divided by 
        # total_train_edges to compute the average training loss per valid edge
        total_train_loss = 0.0
        # number of non-padded edges processed during the epoch
        total_train_edges = 0
        # sum of gradient norms across optimizer updates
        total_gradient_norm = 0.0
        # largest pre-clipping gradient norm observed during the epoch
        max_gradient_norm = 0.0
        # number of batches that produced a valid optimizer update
        num_gradient_updates = 0



        # TRAINING BATCH LOOP

        # loops through training batches
        for edge_features, edge_bias, y, mask, attack_limits in train_loader:
            
            # edge_features shape: (batch_size, padded_num_edges, input_dim)
            # convert features to float32 and move them to the model's CPU or GPU device
            edge_features = edge_features.to(device,dtype=torch.float32)
            # edge_bias shape: (batch_size, padded_num_edges, padded_num_edges)
            # tensor stores pairwise graph structure used by the edge-bias, tropical, and GNN models
            edge_bias = edge_bias.to(device,dtype=torch.float32)
            # y shape: (batch_size, padded_num_edges)
            # each real edge has a binary MIP attack label
            y = y.to(device,dtype=torch.float32)
            # mask shape: (batch_size, padded_num_edges)
            # True indicates a real edge, False indicates padding added by collate_graphs
            mask = mask.to(device=device, dtype=torch.bool)
            # attack_limits shape:(batch_size,)
            # each value gives the permitted number of interdicted edges for the corresponding graph 
            # sample - training loss does not directly use this tensor, but it is included in the batch 
            # format and is used for top-k prediction metrics during validation
            attack_limits = attack_limits.to(device)

            # remove gradients left over from the previous batch set_to_none=True is generally more
            # memory-efficient than overwriting existing gradient tensors with zeros
            optimizer.zero_grad(set_to_none=True)

            # run one forward pass = logits shape: (batch_size, padded_num_edges) - each logit is an 
            # unnormalized prediction score indicating  whether the corresponding edge should be interdicted
            logits = model(edge_features, edge_bias=edge_bias, mask=mask)

            # compute one binary cross-entropy loss for every real or padded edge position
            # loss_matrix shape matches logits and y: (batch_size, padded_num_edges)
            loss_matrix = loss_fn(logits, y)

            # retain only losses corresponding to actual graph edges - losses at padded positions 
            # must not influence training
            valid_edge_losses = loss_matrix[mask]

            # a properly formed batch should contain valid edges - guard prevents mean() from being 
            # called on an empty tensor if an invalid batch is encountered
            if valid_edge_losses.numel() == 0:
                continue 

            # average binary cross-entropy over all real edges in the current batch
            loss = valid_edge_losses.mean()

            # backpropagate the batch loss and populate the .grad attribute of each trainable
            #  model parameter
            loss.backward()

            # compute the total gradient norm and clip gradients whose combined norm exceeds 1.0
            # clip_grad_norm_ returns the total norm before clipping
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           max_norm=1.0)

            # convert the scalar tensor to a Python number for logging
            gradient_norm_value = gradient_norm.item()

            # accumulate gradient norms so an epoch mean can be reported
            total_gradient_norm += gradient_norm_value

            # record the largest gradient norm seen during this epoch
            max_gradient_norm = max(max_gradient_norm,gradient_norm_value,)

            # record that this batch produced gradients and will result in one optimizer update
            num_gradient_updates += 1
            
            # update model parameters using the clipped gradients
            optimizer.step()

            # count number of read edges in this batch
            num_valid_edges = mask.sum().item()

            # loss is the mean loss for this batch - multiplying it by the number of valid edges
            # converts it back to the summed loss contribution for the batch
            # this allows the final epoch average to weight batches by edge count instead of treating
            # a small graph batch and a large graph batch equally
            total_train_loss += loss.item() * num_valid_edges
            # accumulate the number of real edges used in training
            total_train_edges += num_valid_edges

        # the epoch cannot produce a meaningful loss if no valid edges were processed
        if total_train_edges == 0:
            raise RuntimeError("No valid training edges were found during this epoch.")

        # ensure at least one batch actually completed backpropagation and an optimizer update
        if num_gradient_updates == 0:
            raise RuntimeError("No gradient updates occurred during this epoch.")

        # compute the mean pre-clipping gradient norm across all optimizer updates in this epoch
        avg_gradient_norm = (total_gradient_norm / num_gradient_updates)



        # VALIDATION LOOP

        # disable training-specific model behavior, including dropout
        model.eval()
        # Schedule-Free requires optimizer.eval() before evaluating the model = switches to the
        # optimizer's evaluation parameter representation
        optimizer.eval()

        # sum of validation batch losses weighted by valid-edge count
        total_val_loss = 0.0

        # number of real validation edges used to compute validation loss
        total_val_edges = 0

        # number of correctly classified edge labels across the complete validation set
        total_correct = 0

        # number of real validation edges used for accuracy
        total_edges = 0

        # total true positives across all validation graphs
        total_tp = 0

        # total false positives across all validation graphs      
        total_fp = 0

        # total false negatives across all validation graphs
        total_fn = 0

        # sum of graph-level Hamming distances across validation samples
        total_hamming = 0

        # number of validation graphs for which the predicted attack set exactly equals
        #  the MIP-optimal attack set
        total_exact_match = 0

        # total number of graph samples evaluated during validation
        num_val_samples = 0
        

        # disable automatic-gradient tracking because validation does
        # not call backward() or update model parameters
        with torch.no_grad(): 
            # loops through validation batches
            for edge_features, edge_bias, y, mask, attack_limits in val_loader:

                # move validation edge features to the selected device
                edge_features = edge_features.to(device,dtype=torch.float32)
                # move pairwise graph-structure tensors to the device
                edge_bias = edge_bias.to(device,dtype=torch.float32)
                # move binary MIP labels to the device
                y = y.to(device,dtype=torch.float32)
                # preserve the validity mask as Boolean values
                mask = mask.to(device=device, dtype=torch.bool)
                # move each graph's interdiction budget to the device
                attack_limits = attack_limits.to(device)

                # predict one interdiction logit for every real or padded edge position
                logits = model(edge_features, edge_bias=edge_bias, mask=mask)

                # compute unreduced edge-level validation losses
                loss_matrix = loss_fn(logits, y)

                # remove losses associated with padded edge positions
                valid_edge_losses = loss_matrix[mask]

                # skip an invalid empty batch rather than computing the
                # mean of an empty tensor
                if valid_edge_losses.numel() == 0:
                    continue

                # compute mean validation loss for real edges in this batch
                loss = valid_edge_losses.mean()

                # count real edges in this validation batch
                num_valid_edges = mask.sum().item()
               
                # convert the batch mean loss back to a summed loss so the final epoch
                #  average is weighted by edge count
                total_val_loss += loss.item() * num_valid_edges

                # accumulate real validation-edge count
                total_val_edges += num_valid_edges

                # convert logits into top-k attack predictions using each graph's attack limit, 
                # compare predictions with the MIP labels, and return raw batch totals
                (batch_correct, batch_edges, batch_tp, batch_fp, batch_fn, batch_hamming,
                batch_exact_match, batch_size,) = compute_metrics(logits=logits,y=y,mask=mask,
                                                                  attack_limits=attack_limits)

                # add correctly predicted edge labels
                total_correct += batch_correct

                # add total real edges used for classification metrics
                total_edges += batch_edges

                # add positive-class classification counts
                total_tp += batch_tp
                total_fp += batch_fp
                total_fn += batch_fn

                # add graph-level attack-set disagreement totals
                total_hamming += batch_hamming

                # add number of exactly correct graph attack sets
                total_exact_match += batch_exact_match

                # add number of graphs evaluated in this batch
                num_val_samples += batch_size


        # validation metrics are undefined if no real edges or graph samples were processed
        if total_val_edges == 0 or num_val_samples == 0:
            raise RuntimeError("No valid validation edges were found during this epoch.")

        # record the end of both training and validation for this epoch
        epoch_end_time = time.perf_counter()

        # total wall-clock seconds required for the epoch
        epoch_time = epoch_end_time - epoch_start_time

        # compute mean training loss per real edge across the full training partition
        avg_train_loss = total_train_loss / total_train_edges

        # compute mean validation loss per real edge across the full validation partition  
        avg_val_loss = total_val_loss / total_val_edges 

        # edge-level accuracy = correctly classified real edges / all real validation edges
        avg_accuracy = total_correct / max(total_edges, 1)

        # micro-averaged precision and recall across all validation edges (full set)
        avg_precision = total_tp / max(total_tp + total_fp, 1)
        avg_recall = total_tp / max(total_tp + total_fn, 1)


        # Compute the harmonic mean of global precision and recall = F1
        # zero check avoids division by zero when neither metric has a positive value
        # F1 calculated from global precision and recall
        if avg_precision + avg_recall > 0:
            avg_f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall))
        else:
            avg_f1 = 0.0

        # mean number of mismatched attack labels per validation graph
        avg_hamming = total_hamming / max(num_val_samples, 1)

        # fraction of validation graphs for which the complete predicted
        # attack set exactly matches the MIP-optimal attack set
        avg_exact_match = total_exact_match / max(num_val_samples, 1)

        # determine whether this epoch achieved a lower validation loss than every prior epoch
        is_best_epoch = avg_val_loss < best_val_loss

        if is_best_epoch:
            # update the best validation loss observed so far
            best_val_loss = avg_val_loss
            # convert the zero-indexed loop value to a human-readable one-indexed epoch number            
            best_epoch = epoch + 1

            # save a checkpoint containing the model state, optimizer
            # state, experiment identifiers, and validation statistics
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

        # store all epoch-level metrics in a dictionary for later CSV output
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

        # print a compact summary of the current epoch
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



    # FINAL CHECKPOINT AND RUN SUMMARY
    
    # record the wall-clock time after all epochs are complete
    training_end_time = time.perf_counter()

    # calculate total training and validation time across every epoch
    total_training_time = training_end_time - training_start_time

    # put the model and Schedule-Free optimizer into evaluation mode before saving the final checkpoint
    model.eval()
    optimizer.eval()

    # save the state reached after the final epoch, even when that state
    # is not the best-validation-loss state
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

    # print total runtime and checkpoint locations
    print(f"\nTotal training time for {run_name}: "
           f"{total_training_time:.2f} seconds")
    
    print(f"Average epoch time: "
          f"{total_training_time / epochs:.2f} seconds")
    
    print(f"Best epoch: {best_epoch} | "
          f"Best validation loss: {best_val_loss:.6f}")

    print(f"Best model saved to: {best_model_path}")

    print(f"Final model saved to: {final_model_path}")



    # SAVE EPOCH-LEVEL TRAINING LOG
    # include the optional experiment tag in the run name so evaluation loads the
    # checkpoint associated with the correct training run
    if experiment_tag:
        run_name = f"{model_type}_{problem_type}_{experiment_tag}"
    else:
        run_name = f"{model_type}_{problem_type}"


    # save one row per epoch for plotting learning curves and comparing
    # training behavior across model architectures
    with open(f"results/training_log_{run_name}.csv", "w", newline="") as csvfile:
        # use the keys from the first epoch dictionary as CSV columns
        fieldnames = epoch_rows[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)



    # SAVE RUN-LEVEL SUMMARY

    # save one compact row describing the completed training run
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
        "final_model_path": final_model_path,})


# execute train() only when this file is run directly = importing train.py from another 
# module will not automatically begin model training
if __name__ == "__main__":
    train()
