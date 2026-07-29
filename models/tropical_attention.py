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
                 symmetric=True,use_edge_bias=True):
        
        # initializes parent PyTorch class
        super(TropicalAttention, self).__init__()
        
        # make sure hidden dimension can be evenly split across attention heads
        assert d_model % n_heads == 0
        
        # sets up dimension per attention head
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.tropical_proj = tropical_proj
        self.tropical_norm = tropical_norm
        self.symmetric = symmetric
        self.use_edge_bias = use_edge_bias

        if not symmetric:
            raise ValueError("This implementation currently requires symmetric=True.")

        # Ordinary learned Q, K, and V projections. These match the projection stage used in standard attention.
        self.query_linear = nn.Linear(d_model,d_model,bias=False,)
        self.key_linear = nn.Linear(d_model,d_model,bias=False,)
        self.value_linear = nn.Linear(d_model,d_model,bias=False,)
    
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

        # Apply separate learned Q, K, and V projections.
        q = self.query_linear(x)
        k = self.key_linear(x)
        v = self.value_linear(x)

        # Move projected values into the nonnegative log domain.
        q = torch.log1p(F.relu(q))
        k = torch.log1p(F.relu(k))
        v = torch.log1p(F.relu(v))
        
        # if normalization is enabled
        if self.tropical_norm:
            # Apply ReLU and log1p in a single pass before linear transformation
            # 1. ReLU removes negative values
            # 2. log1p applies log(1 + x)
            # 3. normalize_tropical subtracts lambda
            q = self.normalize_tropical(q)
            k = self.normalize_tropical(k)
            v = self.normalize_tropical(v)
        
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
            max_diff = diff.max(dim=-1).values  # [B, S, S]
            min_diff = diff.min(dim=-1).values  # [B, S, S]
            
            # computes tropical/Hilbert projective distance
            d_trop = max_diff - min_diff    # [B, S, S]
            
            # makes smaller distances become larger attention scores
            # closer edge representations receive stronger attention
            attn_scores = - d_trop  
        
        # Add edge bias = if edge bias matrix is passed in
        # encourages model to apy more attention to connected, adjacent, or structurally
        # important edges
        # Add structural graph bias only when explicitly enabled.
        if self.use_edge_bias and edge_bias is not None:

            expanded_bias = edge_bias.unsqueeze(1).expand(-1,self.n_heads,-1,-1,)

            expanded_bias = expanded_bias.reshape(B,seq_len,seq_len,)

            attn_scores = attn_scores + expanded_bias

           

        # mask padded edges = if some edge positions are padded 
        if mask is not None:
            # copies mask across attention heads
            head_mask = mask.unsqueeze(1).expand(-1,self.n_heads,-1,)

            head_mask = head_mask.reshape(B,seq_len,)

            pair_mask = (head_mask.unsqueeze(1) & head_mask.unsqueeze(2))
            mask_value = torch.finfo(attn_scores.dtype).min
            # sets valid attention scores to a huge negative number (ignored)
            attn_scores = attn_scores.masked_fill(~pair_mask, mask_value)
        
        
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




class TropicalTransformerBlock(nn.Module):

    """Complete Transformer encoder block using tropical attention."""

    def __init__(self,d_model,n_heads,device="cpu",dropout=0.1,use_edge_bias=True):

        super(TropicalTransformerBlock, self).__init__()

        self.attn = TropicalAttention(d_model=d_model,n_heads=n_heads,device=device,tropical_proj=True,
                                      tropical_norm=False,symmetric=True,use_edge_bias=use_edge_bias,)

        # Same feed-forward structure used by the other Transformers.
        self.ff = nn.Sequential(
            nn.Linear(d_model,4 * d_model,),
            nn.ReLU(),
            nn.Linear(4 * d_model,d_model,),)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_bias=None, mask=None):

        # Tropical-attention residual block.
        residual = x

        attn_out, attn_scores = self.attn(x, edge_bias=edge_bias,mask=mask,)

        x = self.norm1(residual + self.dropout(attn_out))

        # Feed-forward residual block.
        residual = x

        ff_out = self.ff(x)

        x = self.norm2(residual + self.dropout(ff_out))

        return x, attn_scores
    


    
class TropicalInterdictionModel(nn.Module):
    
    '''defines full model for interdiction problem'''
    
    # defines default setup
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2, dropout=0.1, 
                 device="cpu", use_edge_bias=True):
        
        # initializes PyTorch parent class
        super(TropicalInterdictionModel, self).__init__()

        # projects raw edge features into hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # creates list of tropical attention layers
        self.layers = nn.ModuleList([
            TropicalTransformerBlock(d_model=d_model, n_heads=n_heads, device=device,
                dropout=dropout, use_edge_bias=use_edge_bias)
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

            # applies tropical attention
            x, _ = layer(x, edge_bias=edge_bias, mask=mask)


        # applies classifier to each edge
        logits = self.classifier(x).squeeze(-1)

        # invalid padded edges get very large negative score = 
        if mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            # prevents model from selecting fake / padded edges
            logits = logits.masked_fill(~mask, mask_value)

        # returns one interdiction score per edge
        return logits
