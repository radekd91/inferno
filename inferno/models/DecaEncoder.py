"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
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
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

Parts of the code were adapted from the original DECA release: 
https://github.com/YadiraF/DECA/ 
"""

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import inferno.models.ResNet as resnet
# import timeit

try:
    from .Swin import create_swin_backbone, swin_cfg_from_name
except ImportError as e:
    print("SWIN not found, will not be able to use SWIN models")


class BaseEncoder(nn.Module):
    def __init__(self, outsize, last_op=None):
        super().__init__()
        self.feature_size = 2048
        self.outsize = outsize
        # self.encoder = resnet.load_ResNet50Model()  # out: 2048
        self._create_encoder()
        ### regressor
        self._create_prediction_head()
        self.last_op = last_op

    def _create_prediction_head(self):
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.outsize)
        )

    def forward_features(self, inputs):
        return self.encoder(inputs)

    def forward_features_to_output(self, features):
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters

    def forward(self, inputs, output_features=False):
        # time = timeit.default_timer()
        features = self.forward_features(inputs)
        # time_features = timeit.default_timer() 
        parameters = self.forward_features_to_output(features)
        # time_output = timeit.default_timer() 
        # print(f"Time features:\t{time_features - time:0.05f}")
        # print(f"Time output:\t{time_output - time_features:0.05f}")
        if not output_features:
            return parameters
        return parameters, features

    def _create_encoder(self):
        raise NotImplementedError()

    def reset_last_layer(self):
        # initialize the last layer to zero to help the network 
        # predict the initial pose a bit more stable
        torch.nn.init.constant_(self.layers[-1].weight, 0)
        torch.nn.init.constant_(self.layers[-1].bias, 0)

    def get_feature_size(self):
        return self.feature_size


class ResnetEncoder(BaseEncoder):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__(outsize, last_op)
    #     feature_size = 2048
    #     self.encoder = resnet.load_ResNet50Model()  # out: 2048
    #     ### regressor
    #     self.layers = nn.Sequential(
    #         nn.Linear(feature_size, 1024),
    #         nn.ReLU(),
    #         nn.Linear(1024, outsize)
    #     )
    #     self.last_op = last_op

    def _create_encoder(self):
        self.encoder = resnet.load_ResNet50Model()  # out: 2048


class SecondHeadResnet(nn.Module):

    def __init__(self, enc : BaseEncoder, outsize, last_op=None):
        super().__init__()
        self.resnet = enc # yes, self.resnet is no longer accurate but the name is kept for legacy reasons (to be able to load old models)
        self.layers = nn.Sequential(
            nn.Linear(self.resnet.feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        if last_op == 'same':
            self.last_op = self.resnet.last_op
        else:
            self.last_op = last_op

    def forward_features(self, inputs):
        out1, features = self.resnet(inputs, output_features=True)
        return out1, features

    def forward_features_to_output(self, features):
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters


    def forward(self, inputs):
        out1, features = self.forward_features()
        out2 = self.forward_features_to_output(features)
        return out1, out2


    def train(self, mode: bool = True):
        #here we NEVER modify the eval/train status of the resnet backbone, only the FC layers of the second head
        self.layers.train(mode)
        return self

    def reset_last_layer(self):
        # initialize the last layer to zero to help the network 
        # predict the initial pose a bit more stable
        torch.nn.init.constant_(self.layers[-1].weight, 0)
        torch.nn.init.constant_(self.layers[-1].bias, 0)

    def get_feature_size(self): 
        return self.resnet.feature_size


class SwinEncoder(BaseEncoder):

    def __init__(self, swin_type, img_size, outsize, last_op=None):
        self.swin_type = swin_type
        self.img_size = img_size
        super().__init__(outsize, last_op)

    def _create_encoder(self):
        swin_cfg = swin_cfg_from_name(self.swin_type)
        self.encoder = create_swin_backbone(
            swin_cfg, self.feature_size, self.img_size, load_pretrained_swin=True, pretrained_model=self.swin_type)


    def forward_features(self, inputs):
        pooled_feature, patches = self.encoder(inputs, include_features=True, include_patches=False)
        return pooled_feature, patches


class SwinToken(SwinEncoder):

    def __init__(self, swin_type, img_size, token_dim_dict, transformer_cfg):
        self.token_dim_dict = token_dim_dict
        self.transformer_cfg = transformer_cfg
        outsize = sum(token_dim_dict.values())
        super().__init__(swin_type, img_size, outsize, last_op=None)

    def _create_prediction_head(self):
        ## create a learnable token for each token in the token dict
        self.token_querys = nn.ParameterDict()

        # self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.encoder.num_features, nhead=8)
        # self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)
        
        from inferno.models.temporal.SequenceModels import TransformerEncoderNoBottleneck as TransformerEnc
        self.transformer = TransformerEnc(self.transformer_cfg, self.encoder.num_features)

        self.token_heads = nn.ModuleDict()
        for token_name, token_size in self.token_dim_dict.items():
            # token dim will be dim of the last layer of the swin encoder
            token_size = self.encoder.num_features
            tokenQuery = nn.Parameter(torch.randn(1, token_size,))
            self.token_querys[token_name] = tokenQuery

            token_head = nn.Sequential(
                nn.Linear(token_size, token_size),
            )
            self.token_heads[token_name] = token_head
            
    def forward_features(self, inputs):
        x, patches = self.encoder.forward(inputs, include_features=False, include_patches=True)
        return x, patches 
    

    def forward_features_to_output(self, patches):
        """
        patches is in dim [batch, num_patches, patch_dim] or [B, L, C]
        """
        
        # get all tokens
        ##
        seq = []
        for key in self.token_querys.keys():
            token_query = self.token_querys[key][None, ...]
            token_query = token_query.expand(patches.shape[0], -1, -1)

            seq += [token_query]
        seq += [patches]
        x = torch.cat(seq, dim=1)

        transformer_sample = {}
        transformer_sample["hidden_feature"] = x
        transformer_sample = self.transformer(transformer_sample)
        x = transformer_sample["encoded_sequence_feature"]

        tokens = []
        features = []
        for ti, token_name in enumerate(self.token_querys.keys()):
            token_head = self.token_heads[token_name]
            token = x[:, ti, :]
            features += [token]
            token = token_head(token)
            tokens += [token]
        tokens = torch.cat(tokens, dim=-1)
        features = torch.cat(features, dim=-1)
        return tokens, features


    def forward(self, inputs, output_features=False):
        # time = timeit.default_timer()
        x, patches = self.forward_features(inputs)
        # time_features = timeit.default_timer() 
        parameters, features = self.forward_features_to_output(patches)
        # time_output = timeit.default_timer() 
        # print(f"Time features:\t{time_features - time:0.05f}")
        # print(f"Time output:\t{time_output - time_features:0.05f}")
        if not output_features:
            return parameters
        return parameters, features


    def reset_last_layer(self):
        # initialize the last layer to zero to help the network 
        # predict the initial pose a bit more stable
        for key in self.token_heads:
            torch.nn.init.constant_(self.token_heads[key][-1].weight, 0)
            torch.nn.init.constant_(self.token_heads[key][-1].bias, 0)


# class ResnetEncoder(nn.Module):
#     def __init__(self, append_layers = None):
#         super(ResnetEncoder, self).__init__()
#         # feature_size = 2048
#         self.feature_dim = 2048
#         self.encoder = resnet.load_ResNet50Model() #out: 2048
#         ### regressor
#         self.append_layers = append_layers
#         ## for normalize input images
#         MEAN = [0.485, 0.456, 0.406]
#         STD = [0.229, 0.224, 0.225]
#         self.register_buffer('MEAN', torch.tensor(MEAN)[None,:,None,None])
#         self.register_buffer('STD', torch.tensor(STD)[None,:,None,None])

#     def forward(self, inputs):
#         inputs = (inputs - self.MEAN)/self.STD
#         features = self.encoder(inputs)
#         if self.append_layers:
#             features = self.last_op(features)
#         return features

# class MLP(nn.Module):
#     def __init__(self, channels = [2048, 1024, 1], last_op = None):
#         super(MLP, self).__init__()
#         layers = []

#         for l in range(0, len(channels) - 1):
#             layers.append(nn.Linear(channels[l], channels[l+1]))
#             if l < len(channels) - 2:
#                 layers.append(nn.ReLU())
#         if last_op:
#             layers.append(last_op)

#         self.layers = nn.Sequential(*layers)

#     def forward(self, inputs):
#         outs = self.layers(inputs)
#         return outs
