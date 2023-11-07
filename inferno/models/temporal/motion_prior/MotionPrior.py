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

import pytorch_lightning as pl 
import torch
from torch import nn
from typing import Any, Optional 
from inferno.models.temporal.Bases import ShapeModel, Preprocessor
from torch.nn.functional import mse_loss, l1_loss, cosine_similarity
from inferno.utils.other import class_from_str
from inferno.layers.losses.RotationLosses import compute_rotation_loss, convert_rot
import numpy as np

class MotionEncoder(torch.nn.Module): 

    def forward(self, input):
        raise NotImplementedError()

    def get_trainable_parameters(self): 
        return [p for p in self.parameters() if p.requires_grad]


class MotionDecoder(torch.nn.Module):

    def forward(self, input):
        raise NotImplementedError()

    def get_trainable_parameters(self): 
        return [p for p in self.parameters() if p.requires_grad]


class MotionQuantizer(torch.nn.Module):
    
    def forward(self, input):
        raise NotImplementedError()

    def get_trainable_parameters(self): 
        return [p for p in self.parameters() if p.requires_grad]


# def motion_encoder_from_cfg(cfg) -> MotionEncoder:
#     if cfg.type == "L2lEncoder":
#         return L2lEncoder(cfg)
#     else:
#         raise NotImplementedError(f"Motion encoder type '{cfg.type}' not implemented")


# def motion_decoder_from_cfg(cfg) -> MotionDecoder:
#     if cfg.type == "L2lDecoder":
#         return L2lDecoder(cfg)
#     else:
#         raise NotImplementedError(f"Motion decoder type '{cfg.type}' not implemented")
        

# def motion_codebook_from_cfg(cfg) -> MotionDecoder:
#     if cfg.type == "L2lCodebook":
#         return L2lCodebook(cfg)
#     else:
#         raise NotImplementedError(f"Motion decoder type '{cfg.type}' not implemented")
        


def temporal_trim_dict(sample, start_frame, end_frame, is_batched=False):
    for key in sample.keys():
        if isinstance(sample[key], (np.ndarray, torch.Tensor)):
            if key in ["gt_expression", "gt_exp", "gt_vertices"]:
                print(key)
            if len(sample[key].shape) >= 2 + int(is_batched):
                if is_batched:
                    sample[key] = sample[key][:, start_frame:end_frame]
                else:
                    sample[key] = sample[key][start_frame:end_frame]
        # elif isinstance(sample[key], str):
        #     pass
        elif isinstance(sample[key], list):
            if not isinstance(sample[key][0], (str)):
                if is_batched:
                    sample[key] = [s[:, start_frame:end_frame] for s in sample[key]]
                else:
                    sample[key] = sample[key][start_frame:end_frame]
        elif isinstance(sample[key], dict):
            sample[key] = temporal_trim_dict(sample[key], start_frame, end_frame, is_batched=is_batched)
        else:
            raise NotImplementedError(f"Type {type(sample[key])} not implemented")
    return sample


def rotation_dim(rotation_representation):
    if rotation_representation in ["quaternion", "quat"]:
        return 4
    elif rotation_representation == "euler":
        return 3
    elif rotation_representation in ["axis_angle", "aa"]:
        return 3
    elif rotation_representation in ["mat", "matrix"]:
        return 9
    elif rotation_representation in ["6d", "6dof"]:
        return 6
    else:
        raise NotImplementedError(f"Rotation representation '{rotation_representation}' not implemented")


class MotionPrior(pl.LightningModule):

    def __init__(self, 
        cfg,
        motion_encoder : MotionEncoder,
        motion_decoder : MotionDecoder,
        motion_quantizer: Optional[MotionQuantizer] = None,
        # shape_model: Optional[ShapeModel] = None,
        preprocessor: Optional[Preprocessor] = None,
        postprocessor: Optional[Preprocessor] = None,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.motion_encoder = motion_encoder
        self.motion_decoder = motion_decoder
        self.motion_quantizer = motion_quantizer
        # self.shape_model = shape_model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def to(self, *args: Any, **kwargs: Any) -> "DeviceDtypeModuleMixin":
        if self.preprocessor is not None:
            self.preprocessor = self.preprocessor.to(*args, **kwargs)
        if self.postprocessor is not None:
            self.postprocessor = self.postprocessor.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def discard_encoder(self):
        """
        Deletes the encoder from the model. Useful for saving memory when only using the decoder in downstream tasks.
        """
        self.motion_encoder = None

    def discard_decoder(self):
        """
        Deletes the decoder from the model. Useful for saving memory when only using the encoder in downstream tasks.
        """
        self.motion_decoder = None

    def discard_quantizer(self):
        """
        Deletes the quantizer from the model. Useful for saving memory when only using the encoder in downstream tasks.
        """
        self.motion_quantizer = None

    def latent_frame_size(self):
        """
        How many real temporal frames are encoded in a single latent frame.
        """
        return self.motion_encoder.latent_temporal_factor()

    def quant_factor(self):
        """
        Number of times a latent frame is doubled in size by to produce a real frame.
        """
        return self.motion_encoder.quant_factor()

    def bottleneck_dim(self):
        return self.motion_encoder.bottleneck_dim()

    def quant_factor(self): 
        return self.motion_encoder.quant_factor()

    def get_flame(self):
        # from inferno.models.temporal.Preprocessors import FlamePreprocessor
        # if isinstance(self.preprocessor, FlamePreprocessor):
        if "FlamePreprocessor" in self.preprocessor.__class__.__name__:
            return self.preprocessor.flame
        
    def set_flame_tex(self, flame_tex):
        if "FlamePreprocessor" in self.preprocessor.__class__.__name__:
            self.preprocessor.flame_tex = flame_tex

    @property
    def max_seq_length(self):
        return 5000

    def get_input_sequence_dim(self):
        dim = 0
        for key in self.cfg.model.sequence_components.keys():
            if self.cfg.model.sequence_components[key] == "rot":
                dim += rotation_dim(self.cfg.model.rotation_representation)
            elif self.cfg.model.sequence_components[key] == "flame_verts":
                dim += 5023 * 3
            else: 
                
                dim += self.cfg.model.sequence_components[key]
        return dim

    def _rotation_representation(self):
        return self.cfg.model.rotation_representation

    def get_trainable_parameters(self):
        trainable_params = []
        trainable_params += self.motion_encoder.get_trainable_parameters()
        trainable_params += self.motion_decoder.get_trainable_parameters()
        if self.motion_quantizer is not None:
            trainable_params += self.motion_quantizer.get_trainable_parameters()
        # if self.shape_model is not None:
        #     trainable_params += self.shape_model.get_trainable_parameters()
        return trainable_params

    def configure_optimizers(self):
        trainable_params = []
        trainable_params += list(self.get_trainable_parameters())

        if self.cfg.learning.optimizer == 'Adam':
            opt = torch.optim.Adam(
                trainable_params,
                lr=self.cfg.learning.learning_rate,
                amsgrad=False)
        # elif self.cfg.learning.optimizer == 'AdaBound':
        #     opt = adabound.AdaBound(
        #         trainable_params,
        #         lr=self.cfg.learning.learning_rate,
        #         final_lr=self.cfg.learning.final_learning_rate
        #     )
        elif self.cfg.learning.optimizer == 'SGD':
            opt = torch.optim.SGD(
                trainable_params,
                lr=self.cfg.learning.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: '{self.cfg.learning.optimizer}'")

        optimizers = [opt]
        schedulers = []

        opt_dict = {}
        opt_dict['optimizer'] = opt
        if 'learning_rate_patience' in self.cfg.learning.keys():
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                                   patience=self.cfg.learning.learning_rate_patience,
                                                                   factor=self.cfg.learning.learning_rate_decay,
                                                                   mode=self.cfg.learning.lr_sched_mode)
            schedulers += [scheduler]
            opt_dict['lr_scheduler'] = scheduler
            opt_dict['monitor'] = 'val_loss_total'
        elif 'learning_rate_decay' in self.cfg.learning.keys():
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.cfg.learning.learning_rate_decay)
            opt_dict['lr_scheduler'] = scheduler
            schedulers += [scheduler]
        return opt_dict

    def preprocess(self, batch):
        if self.preprocessor is not None:
            batch = self.preprocessor(batch, input_key=None, output_prefix="gt_", with_grad=False)
        if "template" in batch.keys():
            batch["gt_vertex_offsets"] = batch["gt_vertices"] - batch["template"][:,None, ...]
        return batch

    def compose_sequential_input(self, batch):
        prefix = "gt_"
        if "reconstruction" in batch.keys():
            assert len(batch["reconstruction"]) <= 1, "Only one type of reconstruction is supported for now."
            reconstruction_key = list(batch["reconstruction"].keys())[0]
            rec_dict = batch["reconstruction"][reconstruction_key]
        else:
            rec_dict = batch
        # self._input_sequence_keys = [prefix + key for key in self.cfg.model.sequence_components.keys()]
        # self._input_feature_dims = {key: rec_dict[key].shape[2] for key in self._input_sequence_keys}
        # self._input_feature_dims = self.cfg.model.sequence_components
        input_sequences = [] 
        for key in self.cfg.model.sequence_components.keys():
            if self.cfg.model.sequence_components[key] == "rot":
                rotation = rec_dict[prefix + key][..., :rotation_dim(self.cfg.model.rotation_representation)]
                if self._rotation_representation() != "aa":
                    rotation = convert_rot(rotation, "aa", self._rotation_representation())
                input_sequences += [rotation]
            elif self.cfg.model.sequence_components[key] == "flame_verts":
                input_sequences += [rec_dict[prefix + key][..., :5023 * 3]]
            else:
                input_sequences += [rec_dict[prefix + key][..., :self.cfg.model.sequence_components[key]]]
        batch["input_sequence"] = torch.cat(input_sequences, dim=2)
        return batch

    def encode(self, batch):
        batch = self.motion_encoder(batch)
        return batch

    def quantize(self, batch, training_or_validation=False):
        if self.motion_quantizer is not None:
            # if training 
            if training_or_validation:
                training_step = self.global_step 
            else:
                training_step = None
            batch = self.motion_quantizer(batch, step=training_step)
        return batch

    def decode(self, batch, assert_size=False):
        key = "quantized_features" if self.motion_quantizer is not None else "encoded_features"
        batch = self.motion_decoder(batch, input_key=key)
        if assert_size:
            assert batch["decoded_sequence"].shape == batch["input_sequence"].shape, \
                "Decoded sequence shape does not match input sequence shape."
        return batch

    def decompose_sequential_output(self, batch):
        output_sequence_key = "decoded_sequence"
        prefix = "reconstructed_"
        start_idx = 0
        for i, key in enumerate(self.cfg.model.sequence_components.keys()):
            # dim = self._input_feature_dims[key]
            dim = self.cfg.model.sequence_components[key]
            if dim == "rot":
                dim = rotation_dim(self._rotation_representation()) 
            elif dim == "flame_verts":
                dim = 5023 * 3
            batch[prefix + key] = batch[output_sequence_key][:, :, start_idx:start_idx + dim]
            start_idx += dim
        return batch

    def postprocess(self, batch):
        if "reconstruction" in batch.keys(): # is the pseudo-GT reconstruction fed to the model?
            reconstruction_key = list(batch["reconstruction"].keys())[0]
            rec_batch = batch["reconstruction"][reconstruction_key].copy() 
        else: 
            rec_batch = batch.copy()
        for key in self.cfg.model.sequence_components.keys():
            if "gt_" + key in rec_batch.keys():
                del rec_batch["gt_" + key]
            rec_batch["gt_" + key] = batch["reconstructed_" + key].contiguous()
        if "gt_vertices" in rec_batch.keys():
            del rec_batch["gt_vertices"]

        if self.postprocessor is not None:
            rec_batch = self.postprocessor(rec_batch, input_key=None, output_prefix="reconstructed_", with_grad=True)
        
        if "template" in batch.keys() and "reconstructed_vertex_offsets" in rec_batch.keys():
            rec_batch["reconstructed_vertices"] = rec_batch["reconstructed_vertex_offsets"] + batch["template"][:, None, ...]
        
        batch["reconstructed_vertices"] = rec_batch["reconstructed_vertices"]
        return batch

    def temporal_trim_batch(self, batch):
        """ Trim batch to a compatible length for the model. 
        The sequence lenghts must be divisible by the 2 ** model's quant size """
        quant_factor = self.cfg.model.sizes.quant_factor 
        factor = 2 ** quant_factor
        T = batch["gt_expression"].shape[1] if "gt_expression" in batch.keys() else batch["gt_vertices"].shape[1]
        T_trim = (T // factor) * factor
        batch = temporal_trim_dict(batch, start_frame=0, end_frame=T_trim, is_batched=True)
        return batch

    def encoding_step(self, batch, train=False, validation=False): 
        batch = self.preprocess(batch)
        batch = self.compose_sequential_input(batch)
        batch = self.encode(batch)
        batch = self.quantize(batch, training_or_validation=train or validation)
        return batch

    def input_key_for_decoding_step(self):
        return "encoded_features"

    def output_key_for_decoding_step(self):
        return "decoded_sequence"

    def decoding_step(self, batch, train=False, validation=False):
        batch = self.decode(batch, assert_size=False)
        batch = self.decompose_sequential_output(batch)
        batch = self.postprocess(batch)
        return batch

    def forward(self, batch, train=False, validation=False):
        if train is False and validation is False: 
            # this should only be neccesry in testing (when batch size is 1, and sequence length is arbitraty)
            batch = self.temporal_trim_batch(batch)
        batch = self.preprocess(batch)
        batch = self.compose_sequential_input(batch)
        batch = self.encode(batch)
        batch = self.quantize(batch, training_or_validation=train or validation)
        batch = self.decode(batch, assert_size=True)
        batch = self.decompose_sequential_output(batch)
        batch = self.postprocess(batch)
        return batch

    def _compute_loss(self, sample, training, validation, loss_name, loss_cfg): 
        if "reconstruction" in sample.keys():
            rec_dict = sample["reconstruction"][list(sample["reconstruction"].keys())[0]]
        else:
            rec_dict = sample

        if loss_name == "reconstruction": 
            metric_from_cfg = loss_cfg.get("metric", "mse_loss")
            metric = class_from_str(metric_from_cfg, torch.nn.functional)
            loss = metric(sample[loss_cfg.input_key], sample[loss_cfg.output_key])
            return loss
        elif loss_name == "geometry_reconstruction":
            metric_from_cfg = loss_cfg.get("metric", "mse_loss")
            metric = class_from_str(metric_from_cfg, torch.nn.functional)
            loss_value = metric(rec_dict[loss_cfg.input_key], sample[loss_cfg.output_key])
            return loss_value
        elif loss_name == "exp_loss": 
            metric_from_cfg = loss_cfg.get("metric", "mse_loss")
            metric = class_from_str(metric_from_cfg, torch.nn.functional)
            d = sample[loss_cfg.output_key].shape[-1]
            loss_value = metric(rec_dict[loss_cfg.input_key][...,:d], sample[loss_cfg.output_key])
            return loss_value
        elif loss_name in ["jawpose_loss", "jaw_loss"]:
            # loss = class_from_str(loss_cfg["loss"])(metric=metric)
            pred_jaw_ = sample["reconstructed_jaw"]
            gt_jaw_ = rec_dict["gt_jaw"]
            loss_value = compute_rotation_loss(
                pred_jaw_, 
                gt_jaw_,  
                r1_input_rep=self._rotation_representation(), 
                r2_input_rep="aa", # gt is in axis-angle
                output_rep=loss_cfg.get('rotation_rep', 'quat'), 
                metric=loss_cfg.get('metric', 'l2'), 
                )
            return loss_value
        elif loss_name == "codebook_alignment": 
            return sample["codebook_alignment"]
        elif loss_name == "codebook_commitment":
            return sample["codebook_commitment"]
        elif loss_name == "perplexity":
            return sample["perplexity"]
        elif loss_name == "kl_divergence":
            return sample["kl_divergence"]
        elif loss_name == "gumble_tau":
            return sample["gumble_tau"]
        else: 
            raise ValueError("Unknown loss name: {}".format(loss_name))

    def compute_loss(self, sample, training, validation): 
        """
        Compute the loss for the given sample. 
        """
        losses = {}
        # loss_weights = {}
        metrics = {}

        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            assert loss_name not in losses.keys()
            losses["loss_" + loss_name] = self._compute_loss(sample, training, validation, loss_name, loss_cfg)

        for metric_name, metric_cfg in self.cfg.learning.metrics.items():
            assert metric_name not in metrics.keys()
            with torch.no_grad():
                metrics["metric_" + metric_name] = self._compute_loss(sample, training, validation, metric_name, metric_cfg)

        total_loss = None
        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            term = losses["loss_" + loss_name] 
            if term is not None:
                if isinstance(term, torch.Tensor) and term.isnan().any():
                    print(f"[WARNING]: loss '{loss_name}' is NaN. Skipping this term.")
                    continue
                if total_loss is None: 
                    total_loss = 0.
                weighted_term =  (term * loss_cfg["weight"])
                total_loss = total_loss + weighted_term
                losses["loss_" + loss_name + "_w"] = weighted_term

        losses["loss_total"] = total_loss
        return total_loss, losses, metrics


    def training_step(self, batch, batch_idx, *args, **kwargs):
        training = True 
        # forward pass
        sample = self.forward(batch, train=training, validation=False, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(sample, training=training, validation=False, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"train/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss

    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        training = False 
        # forward pass
        sample = self.forward(batch, train=training, validation=True, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(sample, training=training, validation=True, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"val/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        training = False 
        # forward pass
        sample = self.forward(batch, train=training, validation=False, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(sample, training=training, validation=False, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"test/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss


