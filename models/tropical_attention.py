# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:21:22 2026

@author: emmallen
"""

"""tropical_attention.py

Version 1 of the Tropical Attention Transformer for network interdiction.

This model treats graph edges as sequence tokens and predicts one interdiction logit 
for every edge. It combines ordinary learned query, key, and value projections with 
tropical linear transformations, Hilbert projective distance, and max-plus aggregation.

The model may also incorporate an additive graph-structure bias between pairs of edge 
tokens. It serves as the primary tropical-attention model for comparison against the 
standard Transformer, edge-bias Transformer,and other baseline architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TropicalLinear(nn.Module):
    
    """Max-plus tropical linear transformation.

    For an input vector x and learnable weight matrix W, the layer computes

        y_j = max_i(x_i + W[j, i])

    rather than the ordinary linear transformation y = Wx.

    The addition of x and W corresponds to tropical multiplication, while the maximum 
    over input dimensions corresponds to tropical addition."""
    
    def __init__(self, input_dim, output_dim):
        
        # initializes parent PyTorch class
        super(TropicalLinear, self).__init__()

        # stores input/output dimensions for the layer
        self.input_dim = input_dim
        self.output_dim = output_dim

        # learnable max-plus weight matrix with shape (output_dim, input_dim)
        self.W = nn.Parameter(torch.randn(output_dim, input_dim))
    
    def forward(self, x):
        
        """Apply the max-plus tropical linear map.

        Parameters:
        x : torch.Tensor
            Input tensor with shape (..., input_dim).

        Returns:
        torch.Tensor
            Output tensor with shape (..., output_dim)."""
        
        # insert an output-dimension axis so each input vector can be combined with
        # every row of the tropical weight matrix
        x_expanded = x.unsqueeze(-2)

        # add a leading singleton dimension so the weight matrix broadcasts across all
        # preceding input dimensions
        W_expanded = self.W.unsqueeze(0)

        # tropical multiplication (max-plus multiplication instead of matrix multiplication)
        # replaces ordinary multiplication with addition
        Wx = x_expanded + W_expanded 

        # tropical addition = (max-plus addition instead of matrix addition)
        # replaces ordinary summation with a maximum over hte input-feature dimension
        y = Wx.max(dim=-1).values

        return y


    
class TropicalAttention(nn.Module):
    
    """Multi-head attention using tropical distance and max-plus aggregation.

    The layer first creates ordinary learned query, key, and value projections. These 
    representations are mapped into a nonnegative log domain and may then pass through
    tropical linear projections.

    Pairwise query-key similarity is based on the negative Hilbert projective distance. 
    Smaller tropical distances therefore produce larger attention scores. Unlike standard
    attention, the scores are not normalized with softmax. Context vectors are formed using
    max-plus aggregation."""

    
    def __init__(self, d_model, n_heads, device, tropical_proj=True, tropical_norm=False,
                 symmetric=True,use_edge_bias=True):
        
        # initializes parent PyTorch class
        super(TropicalAttention, self).__init__()
        
        # ensures the hidden simarion can be divided evenly among all attention heads
        assert d_model % n_heads == 0
        
        # store the model dimension, number of heads, and dimension processed by each 
        # individual attention head
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # store configuration options controlling the tropical projections, learned 
        # tropical normalization, distance symmetry, and edge bias
        self.tropical_proj = tropical_proj
        self.tropical_norm = tropical_norm
        self.symmetric = symmetric
        self.use_edge_bias = use_edge_bias

        # currently implements only the symmetric Hilbert projective distance
        if not symmetric:
            raise ValueError("This implementation currently requires symmetric=True.")

        # Ordinary learned projections create separate query (Q), key (K), and value (V) representations 
        # before the tropical transformation = match the projection stage used in standard attention
        self.query_linear = nn.Linear(d_model,d_model,bias=False,)
        self.key_linear = nn.Linear(d_model,d_model,bias=False,)
        self.value_linear = nn.Linear(d_model,d_model,bias=False,)
    
        # recombine the concatenated attention-head outputs in the ordinary vector space
        self.out = nn.Linear(d_model, d_model, bias=False)

        # if tropical projection is "turned on"
        if self.tropical_proj:

            # Optionally apply separate max-plus linear transformations to each head's query, key, and 
            # value representations = creates tropical transformations for queries, keys, and values
            # these replace usual learned Q, K, V linear projections from standard projection
            self.query_trop = TropicalLinear(self.d_k, self.d_k)
            self.key_trop = TropicalLinear(self.d_k, self.d_k)
            self.value_trop = TropicalLinear(self.d_k, self.d_k)

        # if tropical normalization is turned on
        if self.tropical_norm:
            #  learn one normalization offset for each hidden feature
            self.lambda_param = nn.Parameter(torch.ones(1, 1, d_model, device=device))

    def normalize_tropical(self, x):
        
        """Subtract the learned tropical normalization offset.

        This operation shifts each hidden feature before the tropical distance and 
        aggregation calculations."""
        
        return x - self.lambda_param


    def forward(self, x, edge_bias=None, mask=None):
        
        """Apply multi-head tropical attention.

        Parameters
        x : torch.Tensor
            Edge embeddings with shape (batch_size, num_edges, d_model).

        edge_bias : torch.Tensor, optional
            Pairwise graph-structure bias with shape (batch_size, num_edges, num_edges).

        mask : torch.BoolTensor, optional
            Boolean mask with shape (batch_size, num_edges), where True identifies a valid 
            edge and False identifies padding.

        Returns
        output : torch.Tensor
            Updated edge embeddings with shape (batch_size, num_edges, d_model).

        attn_scores : torch.Tensor
            Unnormalized tropical attention scores with shape 
            (batch_size * n_heads, num_edges, num_edges)."""
        
        # extracts batch size and number of edges (tokens)
        batch_size, seq_len, _ = x.size()

        # Construct separate query, key, and value representations from the same input 
        # edge embeddings

        # Apply separate learned Q, K, and V projections.
        q = self.query_linear(x)
        k = self.key_linear(x)
        v = self.value_linear(x)

        # Map the projected representations into a nonnegative log domain.
        # ReLU removes negative values, and log1p computes log(1 + x)
        # while remaining well-defined when x is zero.
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
        
        # split the hidden representatin into multiple atentino heads
        # shape changes from [B,S,d_model] to [B,H,S,d_k]
        # batch size, number of heads, number of edges, dimension per head
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)  
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        
        # merge the batch and head dimensions so each attention head can be processed 
        # independently in parallel
        B = batch_size * self.n_heads
        
        # each attention head is treated like its own batch item
        # shape = [B*H,S,D]
        q = q.reshape(B, seq_len, self.d_k)  # [B, S, D]
        k = k.reshape(B, seq_len, self.d_k)
        v = v.reshape(B, seq_len, self.d_k)

        # tropical linear map - if enabled
        if self.tropical_proj:
            # apply the optional max-plus linear maps independently to the query, key, 
            # and value representation of each attention head
            q = self.query_trop(q)
            k = self.key_trop(k)
            v = self.value_trop(v)

        # compute Hilbert Projective Metric using symmetric tropical distance
        if self.symmetric:
            # compute all pairwise coordinate differences between query and
            # key edge representations
            diff = q.unsqueeze(2) - k.unsqueeze(1)  # [BH, S, S, D]
            
            # for every query-key pair, find the largest and smallest coordinate differences
            max_diff = diff.max(dim=-1).values  # [B, S, S]
            min_diff = diff.min(dim=-1).values  # [B, S, S]
            
            # compute the symmetric Hilbert projective distance:
            # d_H(q, k) = max_r(q_r - k_r) - min_r(q_r - k_r).
            d_trop = max_diff - min_diff    # [B, S, S]
            
            # convert distance into a similarity score - smaller distances
            # produce larger, less-negative attention scores
            attn_scores = - d_trop  
        
        # Add edge bias = if edge bias matrix is passed in encourages model to pay more 
        # attention to connected, adjacent, or structurally important edges
        # Add structural graph bias only when explicitly enabled.
        if self.use_edge_bias and edge_bias is not None:

            # replicate the graph-structure bias across all attention heads
            expanded_bias = edge_bias.unsqueeze(1).expand(-1,self.n_heads,-1,-1,)

            # and merge the batch and head dimensions to match attn_scores
            expanded_bias = expanded_bias.reshape(B,seq_len,seq_len,)

            # add graph structure directly to the pairwise tropical attention scores
            attn_scores = attn_scores + expanded_bias

           

        # mask padded edges = if some edge positions are padded 
        if mask is not None:
            # replicate the valid-edge mask across attention heads
            head_mask = mask.unsqueeze(1).expand(-1,self.n_heads,-1,)

            # and merge the batch and head dimensions to match attn_scores
            head_mask = head_mask.reshape(B,seq_len,)

            # query-key pair is valid only when both edge positions
            # correspond to real, non-padded edges
            pair_mask = (head_mask.unsqueeze(1) & head_mask.unsqueeze(2))
            mask_value = torch.finfo(attn_scores.dtype).min

            # assign invalid edge pairs an extremely negative value so they
            # cannot win the subsequent max-plus aggregation
            attn_scores = attn_scores.masked_fill(~pair_mask, mask_value)
        
        
        # perform max-plus multiplication between each attention score and value vector
        # ordinary multiplication is replaced by elementwise addition
        sum_sv = attn_scores.unsqueeze(-1) + v.unsqueeze(1)  # [B, S, S, D]

        # aggregate over all candidate value edges using a maximum instead of the weighted 
        # sum used by standard attention
        context = sum_sv.max(dim=2).values  # [B, S, D]
        
        # restore separate batch and head dimensions, concatenate the attention heads, 
        # and recover shape (batch_size, num_edges, d_model)
        context = context.reshape(batch_size, self.n_heads, seq_len, self.d_k).permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # map the aggregated context out of the log-domain representation using exp(x) - 1
        context = torch.expm1(context)

        # applies final output projection
        output = self.out(context)
        
        return output, attn_scores




class TropicalTransformerBlock(nn.Module):

    """Transformer-style encoder block using tropical attention.

    The block contains:

        1. Multi-head tropical attention
        2. An ordinary position-wise feed-forward network
        3. Residual connections
        4. Layer normalization
        5. Dropout

    Only the attention operation is tropical. The residual, normalization, feed-forward,
    and dropout components retain their conventional neural-network definitions."""

    def __init__(self,d_model,n_heads,device="cpu",dropout=0.1,use_edge_bias=True):

        # initializes PyTorch parent class
        super(TropicalTransformerBlock, self).__init__()

        # construct the tropical-attention layer using tropical Q/K/V projections, 
        # symmetric Hilbert distance, and optional graph-structure bias
        self.attn = TropicalAttention(d_model=d_model,n_heads=n_heads,device=device,tropical_proj=True,
                                      tropical_norm=False,symmetric=True,use_edge_bias=use_edge_bias,)

        # use the same ordinary position-wise feed-forward architecture as the Transformer 
        # baselines for a controlled comparison
        self.ff = nn.Sequential(
            nn.Linear(d_model,4 * d_model,),
            nn.ReLU(),
            nn.Linear(4 * d_model,d_model,),)

        # normalize the outputs of the attention and feed-forward residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_bias=None, mask=None):

        # apply tropical attention and add its output to the original edge embeddings
        # through a residual connection
        residual = x
        attn_out, attn_scores = self.attn(x, edge_bias=edge_bias,mask=mask,)
        x = self.norm1(residual + self.dropout(attn_out))


        # apply the ordinary position-wise feed-forward network and second residual connection
        residual = x
        ff_out = self.ff(x)
        x = self.norm2(residual + self.dropout(ff_out))


        return x, attn_scores
    


    
class TropicalInterdictionModel(nn.Module):
    
    """Tropical Attention Transformer for edge interdiction.

    Raw edge features are projected into a shared hidden space and processed by stacked
    tropical Transformer blocks. A classifier then produces one unnormalized interdiction
    logit for every edge.

    Parameters:
    input_dim : int = Number of features describing each edge.
    d_model : int, default=64 = Hidden embedding dimension.
    n_heads : int, default=4 = Number of tropical attention heads.
    num_layers : int, default=2 = Number of stacked tropical Transformer blocks.
    dropout : float, default=0.1 = Dropout probability used in each block.
    device : str or torch.device, default="cpu" 
        Device used when creating the optional tropical normalization parameter.
    use_edge_bias : bool, default=True
        Whether graph-structure bias is added to tropical attention scores.

    Input shape: (batch_size, num_edges, input_dim)

    Output shape: (batch_size, num_edges)"""
    
    # defines default setup
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2, dropout=0.1, 
                 device="cpu", use_edge_bias=True):
        
        # initializes PyTorch parent class
        super(TropicalInterdictionModel, self).__init__()

        # project each raw edge-feature vector into the model's shared hidden representation
        self.input_proj = nn.Linear(input_dim, d_model)

        # stack multiple tropical Transformer encoder blocks
        self.layers = nn.ModuleList([
            TropicalTransformerBlock(d_model=d_model, n_heads=n_heads, device=device,
                dropout=dropout, use_edge_bias=use_edge_bias)
            for _ in range(num_layers)]) # repeats for every layer defined

        # map each final edge embedding to one interdiction logit
        # map each final edge embedding through a nonlinear prediction
        # head and produce one raw interdiction score
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(), 
            nn.Linear(d_model, 1))

    def forward(self, edge_features, edge_bias=None, mask=None):
        
        # convert raw edge features into hidden edge embeddings
        x = self.input_proj(edge_features)

        # loops through each tropical attention layer
        for layer in self.layers:

            # refine the edge embeddings through each tropical Transformer block
            # returned attention scores are not required for interdiction prediction and are discarded
            x, _ = layer(x, edge_bias=edge_bias, mask=mask)


        # applies classifier to each edge
        # produce one unnormalized interdiction logit per edge
        logits = self.classifier(x).squeeze(-1)

         
        if mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            # assign padded positions the lowest representable value so they
            # cannot be selected as interdicted edges
            logits = logits.masked_fill(~mask, mask_value)

        # returns one interdiction score per edge
        return logits
