import torch 
import numpy as np
from typing import Dict
from collections.abc import Mapping

def slice_tensors_in_dict(d, start, end, dim):
    """
    Recursively slices tensors in a nested dictionary along a specified dimension 

    :param d: The input dictionary with tensors. 
    :param start: The start index for slicing.
    :param end: The end index for slicing.
    :param dim: The dimension along which to slice.
    :return: A new dictionary with sliced tensors.
    """
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            new_dict[key] = value.narrow(dim, start, end - start)
        elif isinstance(value, np.ndarray):
            # For numpy ndarrays 
            slices = [slice(None)] * value.ndim
            slices[dim] = slice(start, end)
            new_dict[key] = value[tuple(slices)]
        elif isinstance(value, Mapping):
            new_dict[key] = slice_tensors_in_dict(value, start, end, dim)
        else:
            new_dict[key] = value
    return new_dict

def dict_to_device(d, device): 
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)
        elif isinstance(v, dict):
            d[k] = dict_to_device(v, device)
        else: 
            pass
    return d


def dict_get(d, key): 
    if "," not in key: 
        return d[key]
    newkey = key.split(",")[0]
    return dict_get(d[newkey], ",".join(key.split(",")[1:]))


def check_nan(sample: Dict): 
    ok = True
    nans = []
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN found in '{key}'")
                nans.append(key)
                ok = False
                # raise ValueError("Nan found in sample")
    if len(nans) > 0:
        raise ValueError(f"NaN found in {nans}")
    return ok


def detach_dict(d): 
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.detach()
        elif isinstance(v, dict):
            d[k] = detach_dict(v)
        else: 
            pass
    return d