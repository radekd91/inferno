
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as Functional

from ..DecaFLAME import FLAME


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, hidden=2):
        super().__init__()

        if hidden > 5:
            self.skips = [int(hidden / 2)]
        else:
            self.skips = []

        self.network = nn.ModuleList(
            [nn.Linear(z_dim, map_hidden_dim)] +
            [nn.Linear(map_hidden_dim, map_hidden_dim) if i not in self.skips else
             nn.Linear(map_hidden_dim + z_dim, map_hidden_dim) for i in range(hidden)]
        )

        self.output = nn.Linear(map_hidden_dim, map_output_dim)
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.output.weight *= 0.25

    def forward(self, z):
        h = z
        for i, l in enumerate(self.network):
            h = self.network[i](h)
            h = Functional.leaky_relu(h, negative_slope=0.2)
            if i in self.skips:
                h = torch.cat([z, h], 1)

        output = self.output(h)
        return output


class Generator(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, hidden, model_cfg, device,
            regress=True, 
            instantiate_flame=True):
        super().__init__()
        self.device = device
        self.cfg = model_cfg
        self.regress = regress

        if self.regress:
            self.regressor = MappingNetwork(z_dim, map_hidden_dim, map_output_dim, hidden).to(self.device)

        if instantiate_flame:
            self.generator = FLAME(model_cfg).to(self.device)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        try: 
            return  super().load_state_dict(state_dict, strict) 
        except RuntimeError as e: 
            error_message = e.args[0]
            # find all strings enclosed in double quotes 
            import re
            double_quote_strings = re.findall(r'"([^"]*)"', error_message)
            regressor_in_double_quote_strings = any(["regressor." in x for x in double_quote_strings])
            if regressor_in_double_quote_strings:
                # if any of the strings is "regressor", we must throw an error
                raise e
            else:
                if hasattr(self, "generator"): 
                    raise e
                else:
                    return super().load_state_dict(state_dict, strict=False) 

    def forward(self, arcface, decode_verts=True):
        if self.regress:
            shape = self.regressor(arcface)
        else:
            shape = arcface

        prediction = None
        if decode_verts:
            prediction, _, _ = self.generator(shape_params=shape)

        return prediction, shape
