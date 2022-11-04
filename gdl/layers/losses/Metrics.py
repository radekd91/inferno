import torch
import torch.nn.functional as F
from munch import Munch
from omegaconf import DictConfig
from .BarlowTwins import BarlowTwinsLossHeadless, BarlowTwinsLoss
from .Masked import MaskedMSELoss, MaskedMAELoss, MaskedTemporalMSELoss, MaskedTemporalMAELoss


def cosine_sim_negative(*args, **kwargs):
    return (1. - F.cosine_similarity(*args, **kwargs)).mean()


def metric_from_str(metric, **kwargs):
    if metric == "cosine_similarity":
        return cosine_sim_negative
    elif metric in ["l1", "l1_loss", "mae"]:
        return torch.nn.functional.l1_loss
    elif metric in ["masked_l1", "masked_l1_loss", "masked_mae"]:
        return MaskedMAELoss()
    elif metric in ["temporal_masked_l1", "temporal_l1_loss", "temporal_mae"]:
        return MaskedTemporalMAELoss()
    elif metric in ["mse", "mse_loss", "l2", "l2_loss"]:
        return torch.nn.functional.mse_loss
    elif metric in ["masked_mse", "masked_mse_loss", "masked_l2", "masked_l2_loss"]:
        return MaskedMSELoss()
    elif metric in ["temporal_mse", "temporal_mse_loss", "temporal_l2", "temporal_l2_loss"]:
        return MaskedTemporalMSELoss()
    elif metric == "barlow_twins_headless":
        return BarlowTwinsLossHeadless(**kwargs)
    elif metric == "barlow_twins":
        return BarlowTwinsLoss(**kwargs)
    else:
        raise ValueError(f"Invalid metric for deep feature loss: {metric}")


def metric_from_cfg(metric):
    if metric.type == "cosine_similarity":
        return cosine_sim_negative
    elif metric.type in ["l1", "l1_loss", "mae"]:
        return torch.nn.functional.l1_loss
    elif metric.type in ["masked_l1", "masked_l1_loss", "masked_mae"]:
        return MaskedMAELoss()
    elif metric.type in ["temporal_masked_l1", "temporal_l1_loss", "temporal_mae"]:
        return MaskedTemporalMAELoss()
    elif metric.type in ["mse", "mse_loss", "l2", "l2_loss"]:
        return torch.nn.functional.mse_loss
    elif metric.type in ["masked_mse", "masked_mse_loss", "masked_l2", "masked_l2_loss"]:
        return MaskedMSELoss()
    elif metric.type in ["temporal_mse", "temporal_mse_loss", "temporal_l2", "temporal_l2_loss"]:
        return MaskedTemporalMSELoss()
    elif metric.type == "barlow_twins_headless":
        return BarlowTwinsLossHeadless(metric.feature_size)
    elif metric.type == "barlow_twins":
        layer_sizes = metric.layer_sizes if 'layer_sizes' in metric.keys() else None
        return BarlowTwinsLoss(metric.feature_size, layer_sizes)
    else:
        raise ValueError(f"Invalid metric for deep feature loss: {metric}")


def get_metric(metric):
    if isinstance(metric, str):
        return metric_from_str(metric)
    if isinstance(metric, (DictConfig, Munch)):
        return metric_from_cfg(metric)
    if isinstance(metric, dict):
        return metric_from_cfg(Munch(metric))
    raise ValueError(f"invalid type: '{type(metric)}'")