import torch 
from typing import Dict

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