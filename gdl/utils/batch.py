import torch 


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

