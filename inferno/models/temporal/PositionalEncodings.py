"""
Author: Radek Danecek
Copyright (c) 2023, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emote@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

import torch
import torch.nn as nn
import math


def positional_encoding_from_cfg(cfg, feature_dim= None):
    if feature_dim is None:
        feature_dim = cfg.feature_dim 
    if cfg.type == 'PeriodicPositionalEncoding': 
        # return PeriodicPositionalEncoding(cfg.feature_dim, **cfg.positional_encoding)
        return PeriodicPositionalEncoding(feature_dim, **cfg)
    elif cfg.type == 'PositionalEncoding':
        # return PositionalEncoding(cfg.feature_dim, **cfg.positional_encoding)
        return PositionalEncoding(feature_dim, **cfg)
    elif cfg.type == 'LearnedPositionEmbedding':
        return LearnedPositionEmbedding(cfg.max_seq_len, feature_dim)
    elif not cfg.type or str(cfg.type).lower() == 'none':
        return None
    raise ValueError("Unsupported positional encoding")


class PeriodicPositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600, op: str = 'add',  batch_first=True, **kwargs):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)        
        self.op = op
        
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.batch_first = batch_first
        if not self.batch_first:
            pe = pe.transpose(0,1).contiguous()
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first:
            T = x.size(1)
            pe = self.pe[:, :T, :]
        else:
            T = x.size(0)
            pe = self.pe[:T, :]
        if self.op in ['add', 'sum']:
            x = x + pe
        elif self.op in ['concat', 'cat', 'concatenate']:
            x = torch.cat([x, pe.repeat(1,x.shape[1],1)], dim=2)
        else: 
            raise ValueError('how must be either add or concat')
        return self.dropout(x)

    def output_size_factor(self): 
        if self.op in ['add', 'sum']:
            return 1
        elif self.op in ['concat', 'cat', 'concatenate']:
            return 2
        else:
            raise ValueError('how must be either add or concat')


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 600, op: str = 'add',  batch_first=True, **kwargs):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.op = op

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.batch_first = batch_first
        if self.batch_first:
            pe = pe.transpose(0,1).contiguous()
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] if batch_first=True else [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first:
            T = x.size(1)
            pe = self.pe[:, :T, :]
        else:
            T = x.size(0)
            pe = self.pe[:T, :]
        if self.op in ['add', 'sum']:
            x = x + pe
        elif self.op in ['concat', 'cat', 'concatenate']:
            x = torch.cat([x, pe.repeat(1,x.shape[1],1)], dim=2)
        else: 
            raise ValueError('how must be either add or concat')
        return self.dropout(x)

    def output_size_factor(self): 
        if self.op in ['add', 'sum']:
            return 1
        elif self.op in ['concat', 'cat', 'concatenate']:
            return 2
        else:
            raise ValueError('how must be either add or concat')


class LearnedPositionEmbedding(torch.nn.Module):
    """Learned Postional Embedding Layer"""

    def __init__(self, seq_length, dim,  
                 op: str = 'add',  batch_first=True, **kwargs):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(seq_length, dim))
        self.op = op 
        self.batch_first = batch_first

    def forward(self, x):
        T = x.shape[1]
        return x + self.pos_embedding[:T, :]
        # return x + self.pos_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] if batch_first=True else [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first:
            T = x.size(1)
            pe = self.pe[:, :T, :]
        else:
            T = x.size(0)
            pe = self.pe[:T, :]
        if self.op in ['add', 'sum']:
            x = x + pe
        elif self.op in ['concat', 'cat', 'concatenate']:
            x = torch.cat([x, pe.repeat(1,x.shape[1],1)], dim=2)
        else: 
            raise ValueError('how must be either add or concat')
        return self.dropout(x)

    def output_size_factor(self): 
        if self.op in ['add', 'sum']:
            return 1
        elif self.op in ['concat', 'cat', 'concatenate']:
            return 2
        else:
            raise ValueError('how must be either add or concat')
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] if batch_first=True else [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first:
            T = x.size(1)
            pe = self.pos_embedding[:T, :].unsqueeze(0)
        else:
            T = x.size(0)
            pe = self.pos_embedding[:T, :].unsqueeze(1)
        if self.op in ['add', 'sum']:
            x = x + pe
        elif self.op in ['concat', 'cat', 'concatenate']:
            x = torch.cat([x, pe.repeat(1,x.shape[1],1)], dim=2)
        else: 
            raise ValueError('how must be either add or concat')
        # return self.dropout(x)
        return x