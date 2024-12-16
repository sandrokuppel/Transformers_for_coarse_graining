import numpy as np
import torch
from torch import nn


class TBlock(nn.Module):
    """
    Transformer Block Class
    
    This is the main building block of the transformer model. 
    -> multihead attention
    -> layer normalization
    -> feedforward neural network
    -> layer normalization 
    Has the option to return attention weights
    
    Parameters:
    ------------
    hp : dict
        hyperparameters of the model
    block_number : int
        block number of the transformer block (needed for getting attention weights)
    """
    def __init__(self, depth, dimension, heads, hidden_dim, dropout, block_number = None):
        super().__init__()
        heads = heads
        k = dimension
        hidden_dim = hidden_dim
        dropout = dropout
        self.depth = depth
        self.block_number=block_number

        self.attention = nn.MultiheadAttention(k, heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(k, hidden_dim),      
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(hidden_dim, k))
        
    def forward(self, x, get_weights=False):
        """
        Forward pass of the transformer block
        
        Parameters:
        ------------
        x : torch.Tensor
            input tensor
        get_weights : bool
            whether to return attention weights or not
        
        Returns:
        ------------
        torch.Tensor
            output tensor of the transformer block or attention weights
        """
        attended, weights = self.attention(x, x, x, need_weights=True,  average_attn_weights = False)
        if get_weights and self.block_number == self.depth-1:
            return weights
        attended = self.dropout1(attended)
        x = self.norm1(attended + x)
        feedforward = self.dropout2(self.ff(x))
        return self.norm2(feedforward + x)
    
class Encoder(nn.Module):
    """
    Encoder of Transformer

    Puts together the Tranformer blocks for the encoding

    Forward has the option to return the attention weights of the las transformer block
    """
    def __init__(self, depth, dimension, heads, hidden_dim, dropout):
        super().__init__()
        self.depth = depth
        self.tblocks= nn.ModuleList([
            TBlock(
                depth=depth,
                dimension=dimension,
                heads=heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                block_number=i
            )  for i in range(self.depth)
        ])

    def forward(self, x, len=None, get_weights=False):
        for i in range(self.depth):
            x = self.tblocks[i](x, get_weights)
        return x
    
class TPrep(nn.Module):
    """
    Transformer prepare embeddings class

    Takes input of shape (b, t, k0): (batch, sequence, input dimension)
    outputs CLS: classification token, token_embedding: embedded sequence (b,t,k), positional embedding: pe
    If model is pretrained, the embeddings are frozen -> only CLS token is learned

    Parameters:
    ------------
    input_dimension : int
        dimension of input
    dimension : int
        dimension of embedding
    sequence_length : int
        length of sequence
    cls_token : bool
        whether to use CLS token or not
    """
    def __init__(self, input_dimension, dimension, sequence_length=None, cls_token=True, pos_embedding=True):
        super().__init__()
        self.pos_embedding = pos_embedding
        self.k0 = input_dimension
        self.k = dimension
        self.seq_length = sequence_length
        self.cls_token = cls_token

        if cls_token:
            self.CLS = nn.Linear(1, self.k, bias=False)
        self.embed_tokens = nn.Sequential(nn.Linear(self.k0, self.k),
                                          nn.LayerNorm(self.k))
        if pos_embedding:
            self.pos_embedding = nn.Embedding(self.seq_length, self.k)    
            self.layer_norm = nn.LayerNorm(self.k)

    def forward(self, x):
        b, t, k0 = x.size()
        token_embedding = self.embed_tokens(x)                               # go to dimension k
        if self.cls_token:
            CLS = torch.tensor([1.],requires_grad=True).to(x.device)             # CLS token in shape (1,1
            CLS = self.CLS(CLS)[None, :].expand(b,self.k)[:,None,:].to(x.device)     #CLS token in shape (b,1,k)
        if self.pos_embedding:
            pe_out = self.pos_embedding.weight
            pe = pe_out[None, :,:].expand(b,self.seq_length,self.k).to(x.device) # expand to create for every batch 
            token_embedding = self.layer_norm(token_embedding + pe)
            if self.cls_token:
                return CLS, token_embedding, pe_out
            else:
                return token_embedding, pe_out
        else:
            if self.cls_token:
                return CLS, token_embedding
            else:
                return token_embedding