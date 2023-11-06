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
import math 


def biased_mask_from_cfg(cfg):
    temporal_bias_type = cfg.get('type', 'none')
    max_len = cfg.get('max_seq_len', 1200)
    if temporal_bias_type == 'alibi':
        biased_mask = init_alibi_biased_mask(num_heads = cfg.nhead, max_seq_len = max_len)
    elif temporal_bias_type == 'alibi_future':
        biased_mask = init_alibi_biased_mask_future(num_heads = cfg.nhead, max_seq_len = max_len)
    elif temporal_bias_type == 'faceformer':
        biased_mask = init_faceformer_biased_mask(num_heads = cfg.nhead, max_seq_len = max_len, period=cfg.period)
    elif temporal_bias_type == 'faceformer_future':
        biased_mask = init_faceformer_biased_mask_future(num_heads = cfg.nhead, max_seq_len = max_len, period=cfg.period)
    elif temporal_bias_type == 'classic':
        biased_mask = init_mask(num_heads = cfg.nhead, max_seq_len = max_len)
    elif temporal_bias_type == 'classic_future':
        biased_mask = init_mask_future(num_heads = cfg.nhead, max_seq_len = max_len)
    elif temporal_bias_type in ['none', None, False]:
        biased_mask = None
    else:
        raise ValueError(f"Unsupported temporal bias type '{temporal_bias_type}'")
    return biased_mask


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   
    else:                                                 
        closest_power_of_2 = 2**math.floor(math.log2(n)) 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_alibi_biased_mask(num_heads, max_seq_len):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the ALiBi paper.
    This means that the upper triangle of the mask is filled with -inf, 
    the diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    That lowers the attention to the past (the number gets lower the further away from the diagonal it is).
    """
    period = 1
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


def init_alibi_biased_mask_future(num_heads, max_seq_len):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the ALiBi paper but 
    with not with the future masked out.
    The diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    The upper triangle is filled symmetrically with the lower triangle.
    That lowers the attention to the past and the future (the number gets lower the further away from the diagonal it is).
    """
    period = 1
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # mask = alibi - torch.flip(alibi, [1, 2])
    mask = alibi + torch.flip(alibi, [1, 2])
    return mask


def init_faceformer_biased_mask(num_heads, max_seq_len, period):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the FaceFormer paper.
    This means that the upper triangle of the mask is filled with -inf, 
    the diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    That lowers the attention to the past (the number gets lower the further away from the diagonal it is).
    The biased mask has a period, if larger than 1, the bias is repeated period times before coming to the next value.
    If the period is 1, the mask is the same as the alibi mask.
    """
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

def init_faceformer_biased_mask_future(num_heads, max_seq_len, period):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the FaceFormer paper but 
    with not with the future masked out. The biased mask has a perioud, after which it repeats
    The diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    The upper triangle is filled symmetrically with the lower triangle.
    That lowers the attention to the past and the future (the number gets lower the further away from the diagonal it is).
    The biased mask has a period, if larger than 1, the bias is repeated period times before coming to the next value.
    If the period is 1, the mask is the same as the alibi mask.
    """
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # mask = alibi - torch.flip(alibi, [1, 2])
    mask = alibi + torch.flip(alibi, [1, 2])
    return mask

def init_mask(num_heads, max_seq_len):
    """
    Returns a mask for the self-attention layer. 
    The upper triangle of the mask is filled with -inf, the lower triangle is filled with 0. 
    The diagonal is filled with 0.
    """
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.unsqueeze(0).repeat(num_heads,1,1)

def init_mask_future(num_heads, max_seq_len):
    """
    Returns a mask for the self-attention layer. The mask is filled with 0s.
    """
    return torch.zeros(num_heads, max_seq_len, max_seq_len)


# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    smaller_dim = min(T, S)
    for i in range(smaller_dim):
            mask[i, i] = 0
    return (mask==1).to(device=device)


if __name__ == "__main__": 
    """
    Test the masks
    """
    num_heads = 8  
    max_seq_len = 32
    period = 2


    mask_faceformer = init_faceformer_biased_mask(num_heads, max_seq_len, period)
    mask_faceformer_future = init_faceformer_biased_mask_future(num_heads, max_seq_len, period)
    
    mask_alibi = init_alibi_biased_mask(num_heads, max_seq_len)
    mask_alibi_future = init_alibi_biased_mask_future(num_heads, max_seq_len)

    mask = init_mask(num_heads, max_seq_len)
    mask_future = init_mask_future(num_heads, max_seq_len)

    print("mask_alibi")
    print(mask_alibi[0])
    print(mask_alibi[-1])
    print("mask_alibi_future")
    print(mask_alibi_future[0])
    print(mask_alibi_future[-1])
    print("mask_faceformer")
    print(mask_faceformer[0])
    print(mask_faceformer[-1])
    print("mask_faceformer_future")
    print(mask_faceformer_future[0])
    print(mask_faceformer_future[-1])
    print("mask")
    print(mask)
    print(mask)