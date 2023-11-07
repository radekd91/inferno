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
import torch.nn.functional as F
from inferno.layers.losses.Masked import MaskedTemporalMAELoss, MaskedTemporalMSELoss
from inferno.utils.DecaUtils import quaternion_to_angle_axis, quaternion_to_rotation_matrix


def convert_rot(r, input_rep, output_rep):
    from pytorch3d.transforms import matrix_to_quaternion, matrix_to_rotation_6d, rotation_6d_to_matrix
    from inferno.utils.DecaUtils import batch_rodrigues, aa2euler_batch, rot_mat_to_euler
    # assert input_rep != output_rep
    if input_rep == output_rep: 
        return r
    if input_rep in ['aa', 'axis-angle', 'axisangle']:
        if output_rep in ['quat', 'quaternion']: 
            return matrix_to_quaternion(batch_rodrigues(r))
        elif output_rep == 'euler': 
            return aa2euler_batch(r)
        elif output_rep in ['mat', 'matrix']:
            return batch_rodrigues(r)
        elif output_rep in ['6d', '6dof', '6dof']: 
            return matrix_to_rotation_6d(batch_rodrigues(r))
        else: 
            raise NotImplementedError(f"The conversion from {input_rep} {output_rep} is not yet implemented")
    elif input_rep in ['quat', 'quaternion']: 
        rot_mat = quaternion_to_rotation_matrix(r)
    elif input_rep in ['6d', '6dof', '6dof']: 
        rot_mat = rotation_6d_to_matrix(r)
    elif input_rep in ['mat', 'matrix']:
        rot_mat = r.view(-1, 3, 3)
    else:
        raise NotImplementedError(f"The conversion from {input_rep} {output_rep} is not yet implemented")
    # rot_mat = aa2euler_batch(r)
    if output_rep in ['quat', 'quaternion']: 
        return matrix_to_quaternion(rot_mat)
    elif output_rep == 'euler': 
        return rot_mat_to_euler(rot_mat)
    elif output_rep in ['6d', '6dof', '6dof']: 
        return matrix_to_rotation_6d(rot_mat)
    elif output_rep in ['aa', 'axis-angle', 'axisangle']:
        return quaternion_to_angle_axis(matrix_to_quaternion(rot_mat))
    raise NotImplementedError(f"The conversion from {input_rep} {output_rep} is not yet implemented")


def compute_rotation_loss(r1, r2, mask=None, 
        r1_input_rep='aa', r2_input_rep='aa', output_rep='aa',
        metric='l2'): 
    B = r1.shape[0]
    T = r1.shape[1] 
    r1 = convert_rot(r1.contiguous().view((B*T,-1)), r1_input_rep, output_rep).view((B,T,-1))
    r2 = convert_rot(r2.contiguous().view((B*T,-1)), r2_input_rep, output_rep).view((B,T,-1))
    
    # bt_reduction_dim = list(range(len(r1.shape)-1))
    # vec_reduction_dim = len(r1.shape) -1 

    # mask = mask if mask is not None else torch.ones(r1.shape[0], r1.shape[1], 1, device=r1.device)
    mask = mask if mask is not None else torch.ones(r1.shape[0], r1.shape[1], device=r1.device)
    # mask = torch.zeros(r1.shape[0], r1.shape[1], 1, device=r1.device)
    mask_sum = mask.sum().detach()
    if mask_sum < 0.0000001: # avoid division by zero
        # return torch.tensor(0.)
        return None

    
    # metric = 'l1'
    if metric == 'l1': 
        # diff = (r1 - r2)*mask
        # return diff.abs().sum(dim=vec_reduction_dim).sum(dim=bt_reduction_dim) / mask_sum
        # return F.l1_loss((r1*mask).view(B*T, -1), (r2*mask).view(B*T, -1), reduction ='sum' ) / mask_sum
        return MaskedTemporalMAELoss()(r1, r2, mask)
    elif metric == 'l2': 
        # return F.mse_loss((r1*mask).view(B*T, -1), (r2*mask).view(B*T, -1), reduction='sum' ) / mask_sum
        return MaskedTemporalMSELoss()(r1, r2, mask)
        ## return diff.square().sum(dim=vec_reduction_dim).sqrt().sum(dim=bt_reduction_dim) / mask_sum ## does not work, sqrt turns weights to NaNa after backward
    else: 
        raise ValueError(f"Unsupported metric for rotation loss: '{metric}'")
