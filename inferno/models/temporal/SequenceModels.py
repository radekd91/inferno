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
from typing import Any, Optional, Dict, List
import torch
import torch.nn as nn
from inferno.models.temporal.Bases import SequenceClassificationEncoder, ClassificationHead
from inferno.models.temporal.PositionalEncodings import PositionalEncoding, LearnedPositionEmbedding
from inferno.models.temporal.TransformerMasking import (init_alibi_biased_mask, init_alibi_biased_mask_future, 
                                                    init_mask, init_mask_future, init_faceformer_biased_mask, 
                                                    init_faceformer_biased_mask_future, init_faceformer_biased_mask_future)
from omegaconf import OmegaConf, ListConfig


def positional_encoding_from_cfg(cfg, feature_dim): 
    # if cfg.positional_encoding.type == 'PeriodicPositionalEncoding': 
    #     return PeriodicPositionalEncoding(cfg.feature_dim, **cfg.positional_encoding)
    # el
    if cfg.positional_encoding.type == 'PositionalEncoding':
        return PositionalEncoding(feature_dim, **cfg.positional_encoding)
    elif cfg.positional_encoding.type == 'LearnedPositionEmbedding':
        return LearnedPositionEmbedding(dim=feature_dim, **cfg.positional_encoding)
    elif not cfg.positional_encoding.type or str(cfg.positional_encoding.type).lower() == 'none':
        return None
    raise ValueError("Unsupported positional encoding")


class TransformerPooler(nn.Module):
    """
    inspired by: 
    https://huggingface.co/transformers/v3.3.1/_modules/transformers/modeling_bert.html#BertPooler 
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, sample, input_key = "encoded_sequence_feature"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = sample[input_key]
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        sample["pooled_sequence_feature"] = pooled_output
        return sample

class GRUPooler(nn.Module):
    """
    inspired by: 
    https://huggingface.co/transformers/v3.3.1/_modules/transformers/modeling_bert.html#BertPooler 
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, sample, input_key = "encoded_sequence_feature"):
        # We "pool" the model by simply taking the last output of the GRU
        hidden_states = sample[input_key]
        last_output = hidden_states[:, -1]
        pooled_output = self.dense(last_output)
        pooled_output = self.activation(pooled_output)
        sample["pooled_sequence_feature"] = pooled_output
        return sample


class TransformerEncoder(torch.nn.Module):

    def __init__(self, cfg, input_dim) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = input_dim
        # if self.input_dim == cfg.feature_dim:
        #     self.bottleneck = None
        # else:
        self._init_transformer()
        self._init_biased_mask()


    def _init_transformer(self):
        self.bottleneck = nn.Linear(self.input_dim, self.cfg.feature_dim)
        self.PE = positional_encoding_from_cfg(self.cfg, self.cfg.feature_dim)
        dim_factor = self._total_dim_factor()
        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=self.cfg.feature_dim * dim_factor, 
                    nhead=self.cfg.nhead, 
                    dim_feedforward=dim_factor*self.cfg.feature_dim, 
                    activation=self.cfg.activation,
                    dropout=self.cfg.dropout, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.num_layers)
        # self.decoder = nn.Linear(dim_factor*self.input_dim, self.decoder_output_dim())

    def _init_biased_mask(self):
        self.temporal_bias_type = self.cfg.get('temporal_bias_type', 'none')
        if self.temporal_bias_type == 'alibi':
            self.biased_mask = init_alibi_biased_mask(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len)
        elif self.temporal_bias_type == 'alibi_future':
            self.biased_mask = init_alibi_biased_mask_future(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len)
        elif self.temporal_bias_type == 'faceformer':
            self.biased_mask = init_faceformer_biased_mask(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len, period=self.cfg.period)
        elif self.temporal_bias_type == 'faceformer_future':
            self.biased_mask = init_faceformer_biased_mask_future(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len, period=self.cfg.period)
        elif self.temporal_bias_type == 'classic':
            self.biased_mask = init_mask(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len)
        elif self.temporal_bias_type == 'classic_future':
            self.biased_mask = init_mask_future(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len)
        elif self.temporal_bias_type == 'none':
            self.biased_mask = None
        else:
            raise ValueError(f"Unsupported temporal bias type '{self.temporal_bias_type}'")

    def encoder_input_dim(self):
        return self.input_dim

    def encoder_output_dim(self):
        return self.cfg.feature_dim

    def forward(self, sample, train=False, teacher_forcing=True): 
        if self.bottleneck is not None:
            sample["hidden_feature"] = self.bottleneck(sample["hidden_feature"])
        hidden_states = self._positional_enc(sample)
        encoded_feature = self._encode(sample, hidden_states)
        sample["encoded_sequence_feature"] = encoded_feature
        return sample
       
    def _pe_dim_factor(self):
        dim_factor = 1
        if self.PE is not None: 
            dim_factor = self.PE.output_size_factor()
        return dim_factor

    def _total_dim_factor(self): 
        return  self._pe_dim_factor()

    def _positional_enc(self, sample): 
        hidden_states = sample["hidden_feature"] 
        if self.PE is not None:
            hidden_states = self.PE(hidden_states)
        return hidden_states

    def _encode(self, sample, hidden_states):
        if self.biased_mask is not None: 
            mask = self.biased_mask[:, :hidden_states.shape[1], :hidden_states.shape[1]].clone() \
                .detach().to(device=hidden_states.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(hidden_states.shape[0], 1, 1)
        else: 
            mask = None
        encoded_feature = self.encoder(hidden_states, mask=mask)
        B, T = encoded_feature.shape[:2]
        encoded_feature = encoded_feature.view(B*T, -1)
        encoded_feature = self.encoder(hidden_states)
        encoded_feature = encoded_feature.view(B, T, -1)
        return encoded_feature


class TransformerEncoderNoBottleneck(TransformerEncoder):

    def _init_transformer(self):
        self.bottleneck = None
        self.PE = positional_encoding_from_cfg(self.cfg, self.input_dim,)
        dim_factor = self._total_dim_factor()
        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=self.input_dim, 
                    nhead=self.cfg.nhead, 
                    dim_feedforward=dim_factor*self.cfg.hidden_feature_dim, 
                    activation=self.cfg.activation,
                    dropout=self.cfg.dropout, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.num_layers)
        # self.decoder = nn.Linear(dim_factor*self.input_dim, self.decoder_output_dim())


def transformer_encoder_from_cfg(cfg, input_dim):
    if cfg.type == 'TransformerEncoder':
        return TransformerEncoder(cfg, input_dim)
    raise ValueError(f"Unsupported encoder type '{cfg.type}'")

def pooler_from_cfg(cfg):
    if cfg.type == 'TransformerEncoder':
        return TransformerEncoder(cfg)
    raise ValueError(f"Unsupported encoder type '{cfg.type}'")


class GRUSequenceClassifier(SequenceClassificationEncoder):

    def __init__(self, cfg, input_dim, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.latent_dim = cfg.feature_dim
        self.num_layers = cfg.num_layers
        self.bidirectional = cfg.bidirectional
        self.dropout = cfg.dropout
        # if self.input_dim == cfg.feature_dim:
        #     self.bottleneck = None
        # else:
        self.bottleneck = nn.Linear(self.input_dim, cfg.feature_dim)
        self.gru = torch.nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, 
            bidirectional=self.bidirectional, dropout=self.dropout, batch_first=True)
        self.pooler = GRUPooler(self.output_feature_dim(), self.output_feature_dim())

    def forward(self, sample, train=False, teacher_forcing=True): 
        if self.bottleneck is not None:
            sample["hidden_feature"] = self.bottleneck(sample["hidden_feature"])
        output, hidden_states = self.gru(sample["hidden_feature"])
        sample["encoded_sequence_feature"] = output
        sample = self.pooler(sample)
        return sample

    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        return self.cfg.feature_dim

    def output_feature_dim(self):
        return self.cfg.feature_dim * 2 if self.bidirectional else self.cfg.feature_dim

    def encoder_output_dim(self):
        return self.output_feature_dim()


class TransformerSequenceClassifier(SequenceClassificationEncoder):

    def __init__(self, cfg, input_dim, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        # self.num_classes = num_classes

        self.transformer_encoder = transformer_encoder_from_cfg(cfg.encoder, input_dim)
        self.pooler = TransformerPooler(self.transformer_encoder.encoder_output_dim(), self.transformer_encoder.encoder_output_dim())
        # self.classifier = nn.Linear(self.transformer_encoder.encoder_output_dim(), self.num_classes)

    def encoder_output_dim(self):
        return self.transformer_encoder.encoder_output_dim()

    def forward(self, sample):
        sample = self.transformer_encoder(sample)
        sample = self.pooler(sample)
        # sample = self.classifier(sample)
        return sample

    def get_trainable_parameters(self):
        return list(self.parameters())



class LinearClassificationHead(ClassificationHead): 
    def __init__(self, cfg, input_dim,  num_classes):
        super().__init__()
        self.cfg = cfg
        # self.input_dim = cfg.input_dim
        self.input_dim = input_dim
        # self.num_classes = cfg.num_classes
        self.num_classes = num_classes

        self.dropout = nn.Dropout(cfg.dropout_prob)
        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, sample, input_key="pooled_sequence_feature", output_key="predicted_logits"):
        sample[output_key] = self.classifier(sample[input_key])
        return sample

    def get_trainable_parameters(self):
        return list(self.parameters())
    

class MultiheadLinearClassificationHead(ClassificationHead):
    
    def __init__(self, cfg, input_dim, num_classes):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim

        assert isinstance(num_classes, (list, ListConfig))
        classification_heads = [LinearClassificationHead(cfg, input_dim, classes) for classes in num_classes]
        category_names=cfg.get('category_names', None)
        if category_names is None:
            head_names = [f"category_{i}" for i in range(len(num_classes))]
        else:
            head_names = category_names
        self.classification_heads = nn.ModuleDict(dict(zip(head_names, classification_heads)))

    def forward(self, sample, input_key="pooled_sequence_feature", output_key="predicted_logits"):
        for key, head in self.classification_heads.items():
            sample = head(sample, input_key, output_key + f"_{key}")
        return sample

    def get_trainable_parameters(self):
        return list(self.parameters())

