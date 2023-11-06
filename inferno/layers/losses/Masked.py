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

import torch.nn.functional as F 
import torch 


class MaskedLoss(torch.nn.Module):
    
    def __init__(self, func=F.mse_loss, reduction='mean', starting_dim_to_collapse=1):
        super().__init__()
        self.func = func
        self.reduction = reduction 
        self.starting_dim_to_collapse = starting_dim_to_collapse
        assert reduction in ['mean', 'sum', 'none']

    def forward(self, input, target, mask=None):
        # input: (batch_size, seq_len, ...)
        # target: (batch_size, seq_len, ...)
        # mask: (batch_size, seq_len)
        assert input.shape == target.shape, f"input and target shapes must match, got {input.shape} and {target.shape}"
        if mask is None:
            return self.func(input, target, reduction=self.reduction)
        assert mask.shape[0] == input.shape[0]
        if self.starting_dim_to_collapse > 1: # if temporal dimension (B, T, ...), make sure mask has T dimensions
            assert mask.shape[1] == input.shape[1]
        else: 
            assert mask.ndim == 1 or (mask.ndim==2 and mask.shape[1] == 1) # for non temporal batching, the mask should be 1d (masking along the batch dimension only)
        
        loss = self.func(input, target, reduction='none')

        dims_to_collapse = list(range(self.starting_dim_to_collapse, len(input.shape)))
        if len(dims_to_collapse) > 0:
            if self.reduction == 'mean':
                loss = loss.mean(dim=dims_to_collapse)
            elif self.reduction == 'sum':
                loss = loss.sum(dim=dims_to_collapse)
        
        assert loss.shape == mask.shape, f"loss and mask shapes must match, got {loss.shape} and {mask.shape}"
        loss = loss * mask
        
        reduction_dim = self.starting_dim_to_collapse - 1
        if self.reduction == 'mean':
            mask_sum = mask.sum(dim=reduction_dim, keepdims=True)
            # if (mask_sum == 0).all():
            if (mask_sum == 0).any():
                print("[WARNING] Skipping loss calculation because mask is all zeros")
                return None
            loss = loss.sum(dim=reduction_dim, keepdims=True) / mask_sum
            loss_is_nan = loss.isnan()
            if loss_is_nan.any():
                loss = loss[~loss_is_nan]
        elif self.reduction == 'sum': 
            loss = loss.sum(dim=reduction_dim, keepdims=True)
        if self.reduction != 'none':
            assert loss.isnan().any() == False
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        return loss
    

class MaskedTemporalLoss(MaskedLoss):
    
    def __init__(self, func=F.mse_loss, reduction='mean'):
        super().__init__(func=func, reduction=reduction, starting_dim_to_collapse=2)


class MaskedTemporalMSELoss(MaskedTemporalLoss):
    
    def __init__(self, reduction='mean'):
        super().__init__(F.mse_loss, reduction=reduction)


class MaskedTemporalMAELoss(MaskedTemporalLoss):
    
    def __init__(self, reduction='mean'):
        super().__init__(F.l1_loss, reduction=reduction)


class MaskedMSELoss(MaskedLoss):
    
    def __init__(self, reduction='mean'):
        super().__init__(F.mse_loss, reduction=reduction)


class MaskedMAELoss(MaskedLoss):
    
    def __init__(self, reduction='mean'):
        super().__init__(F.l1_loss, reduction=reduction)

