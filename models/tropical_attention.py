# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:21:22 2026

@author: emmallen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TropicalLinear(nn.Module):
    
    '''Class that defines a custom PyTorch layer'''
    
    def __init__(self, input_dim, output_dim):
        
        # initializes parent PyTorch class
        super(TropicalLinear, self).__init__()
        # stores input/output sizes
        self.input_dim = input_dim
        self.output_dim = output_dim
        # creates tropical weights
        self.W = nn.Parameter(torch.randn(output_dim, input_dim))
    
    def forward(self, x):
        
        '''defines what happens when data passes through this layer:
            normal linear layer: y = Wx
            tropical linear layer: y_j = max_i(x_i + W_ji)'''
        
        # adds a dimension so x can be compared against every output weight row
        x_expanded = x.unsqueeze(-2)
        # adds a batch dimension to weights
        W_expanded = self.W.unsqueeze(0)
        # tropical multiplication = instead of matrix multiplication = adds inputs and weights
        Wx = x_expanded + W_expanded 
        # tropical addition = takes max across input features
        y, _ = torch.max(Wx, dim=-1)
        return y
    
class TropicalAttention(nn.Module):
    
    '''Class that defines one Tropical Attnetion layer'''
    
    def __init__(self, d_model, n_heads, device, tropical_proj=True, tropical_norm=False,
                 symmetric=True):
        
        # initializes parent PyTorch class
        super(TropicalAttention, self).__init__()
        
        # make sure hidden dimension can be evenly split across attention heads
        assert d_model % n_heads == 0
        
        # sets up dimension per attention head
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.tropical_proj = tropical_proj
        self.tropical_norm = tropical_norm
        self.symmetric = symmetric
        
        # final normal linear layers without bias
        self.out = nn.Linear(d_model, d_model, bias=False)

        # if tropical projection is "turned on"
        if self.tropical_proj:
            
            # creates tropical transformations for queries, keys, and values
            # these replace usual learned Q, K, V linear projections from standard projection
            self.query_trop = TropicalLinear(self.d_k, self.d_k)
            self.key_trop = TropicalLinear(self.d_k, self.d_k)
            self.value_trop = TropicalLinear(self.d_k, self.d_k)

        # if tropical normalization is turned on
        if self.tropical_norm:
            # creates learnable value that gets subtracted from features
            self.lambda_param = nn.Parameter(torch.ones(1, 1, d_model, device=device))

    def normalize_tropical(self, x):
        
        '''subtracts learned tropical normalization parameter'''
        
        return x - self.lambda_param


    def forward(self, x, edge_bias=None, mask=None):
        
        '''runs the attention layer'''
        
        # extracts batch size and number of edges
        batch_size, seq_len, _ = x.size()
        
        # this stage = q, k, v are all based on same x
        
        # if normalization is enabled
        if self.tropical_norm:
            # Apply ReLU and log1p in a single pass before linear transformation
            # 1. ReLU removes negative values
            # 2. log1p applies log(1 + x)
            # 3. normalize_tropical subtracts lambda
            q = self.normalize_tropical(torch.log1p(F.relu(x)))
            k = self.normalize_tropical(torch.log1p(F.relu(x)))
            v = self.normalize_tropical(torch.log1p(F.relu(x)))
            
        # if normalization is not enabled
        else:
            # do the same thing but without subtracting lambda
            q = torch.log1p(F.relu(x))
            k = torch.log1p(F.relu(x))
            v = torch.log1p(F.relu(x))
        
        # reshape and permute queries for multi-head attention
        # shape changes from [B,S,d_model] to [B,H,S,d_k]
        # B = batch size, H = number of heads, S = number of edges, D = dimension per head
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, H, S, D]
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        
        # Merge batch and heads for parallel computation
        B = batch_size * self.n_heads
        
        # each attention head is treated like its own batch item
        # shape = [B*H,S,D]
        q = q.reshape(B, seq_len, self.d_k)  # [B, S, D]
        k = k.reshape(B, seq_len, self.d_k)
        v = v.reshape(B, seq_len, self.d_k)

        # Tropical linear map - if enabled
        if self.tropical_proj:
            # applies tropical linear transformations to query, key and value
            q = self.query_trop(q)
            k = self.key_trop(k)
            v = self.value_trop(v)

        # Compute Hilbert Projective Metric
        # uses symmetric tropical distance
        if self.symmetric:
            # computes pairwise query-key differences
            diff = q.unsqueeze(2) - k.unsqueeze(1)  # [BH, S, S, D]
            
            # Calculate tropical distance
            # gets max and min feature differences for each edge pair
            max_diff, _ = diff.max(dim=-1)  # [B, S, S]
            min_diff, _ = diff.min(dim=-1)  # [B, S, S]
            
            # computes tropical/Hilbert projective distance
            d_trop = max_diff - min_diff    # [B, S, S]
            
            # makes smaller distances become larger attention scores
            # closer edge representations receive stronger attention
            attn_scores = - d_trop  
        
        # Add edge bias = if edge bias matrix is passed in
        # encourages model to apy more attention to connected, adjacent, or structurally
        # important edges
        if edge_bias is not None:
            # changes shape from [B,S,S] to [B,1,S,S]
            edge_bias = edge_bias.unsqueeze(1)  # [B, 1, S, S]
            # compies same bias across all heads
            edge_bias = edge_bias.repeat(1, self.n_heads, 1, 1)
            # matches reshaped attention score shapes
            edge_bias = edge_bias.reshape(B, seq_len, seq_len)
            # adds graph structure information directly into attention
            attn_scores = attn_scores + edge_bias

        # mask padded edges = if some edge positions are padded 
        if mask is not None:
            # copies mask across attention heads
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1)
            # matches combined batch-head shape
            mask = mask.reshape(B, seq_len)
            # creates pairwise mask so attention is only allowed between valid edges
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            # sets valid attention scores to a huge negative number (ignored)
            attn_scores = attn_scores.masked_fill(~pair_mask, -1e9)
        

        # combines attention scores and value vectors using tropical multiplication
        # instead of attention_weight * value == attention_score + value
        sum_sv = attn_scores.unsqueeze(-1) + v.unsqueeze(1)  # [B, S, S, D]
        # aggregates across attended edges using tropical addition = max
        context = sum_sv.max(dim=2).values  # [B, S, D]
        
        # reshape context back to [batch_size, seq_len, d_model]
        context = context.reshape(batch_size, self.n_heads, seq_len, self.d_k).permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # reverses earlier log transformation
        context = torch.expm1(context)
        # applies final output projection
        output = self.out(context)
        
        return output, attn_scores
    
    
class TropicalInterdictionModel(nn.Module):
    
    '''defines full model for interdiction problem'''
    
    # defines default setup
    def __init__(self, input_dim=7, d_model=64, n_heads=4, num_layers=2, device="cpu"):
        
        # initializes PyTorch parent class
        super(TropicalInterdictionModel, self).__init__()

        # projects raw edge features into hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # creates list of tropical attention layers
        self.layers = nn.ModuleList([
            TropicalAttention(d_model=d_model, n_heads=n_heads, device=device,
                tropical_proj=True, tropical_norm=False, symmetric=True)
            for _ in range(num_layers)]) # repeats for every layer defined

        # creates final prediction head
        # maps hidden features to hidden features
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model),
            # adds nonlinearity, outputs one score per edge
            # score = model's raw belief that edge should be interdicted
            nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, edge_features, edge_bias=None, mask=None):
        
        # turns raw edge features into embeddings
        x = self.input_proj(edge_features)

        # loops through each tropical attention layer
        for layer in self.layers:
            # saves original input to layer
            residual = x
            # applies tropical attention
            x, _ = layer(x, edge_bias=edge_bias, mask=mask)
            # adds residual connection
            x = x + residual

        # applies classifier to each edge
        logits = self.classifier(x).squeeze(-1)

        # invalid padded edges get very large negative score = 
        if mask is not None:
            # prevents model from selecting fake / padded edges
            logits = logits.masked_fill(~mask, -1e9)

        # returns one interdiction score per edge
        return logits
