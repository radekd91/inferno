f"""
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


import os
import sys
# torch.backends.cudnn.benchmark = True
from pathlib import Path
from typing import Any, Optional

import adabound
import cv2
import numpy as np
import pytorch_lightning.plugins.environments.lightning_environment as le
import torch
import torch.nn.functional as F
from munch import Munch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
# from time import time
from skimage.io import imread
from skimage.transform import resize

import inferno.utils.DecaUtils as util
from inferno.datasets.AffectNetDataModule import AffectNetExpressions
from inferno.datasets.AffWild2Dataset import Expression7
from inferno.layers.losses.EmoNetLoss import (create_au_loss, create_emo_loss, EmoNetLoss)
from inferno.layers.losses.VGGLoss import VGG19Loss
from inferno.models.temporal.Bases import ShapeModel
from inferno.models.temporal.external.LipReadingLoss import LipReadingLoss
from inferno.models.temporal.Renderers import Renderer
from inferno.utils.lightning_logging import (_log_array_image, _log_wandb_image,
                                         _torch_image2np)
from inferno.utils.other import class_from_str, get_path_to_assets
from inferno.utils.MediaPipeLandmarkUtils import draw_mediapipe_landmarks, draw_mediapipe_landmark_flame_subset
from .FaceEncoder import encoder_from_cfg
from .Losses import (FanContourLandmarkLoss, LandmarkLoss,
                     MediaPipeLandmarkLoss, MediaPipeLipDistanceLoss,
                     MediaPipeMouthCornerLoss, MediaPipleEyeDistanceLoss,
                     PhotometricLoss, GaussianRegLoss, LightRegLoss)

from inferno.utils.batch import dict_get, check_nan
# import timeit


def shape_model_from_cfg(cfg): 
    cfg_shape = cfg.model.shape_model

    if cfg_shape.type == "FlameShapeModel":
        from inferno.models.temporal.TemporalFLAME import FlameShapeModel
        shape_model = FlameShapeModel(cfg_shape.flame)
    else: 
        raise ValueError(f"Unsupported shape model type: '{cfg_shape.type}'")
    return shape_model


def renderer_from_cfg(cfg):
    cfg_renderer = cfg.model.renderer

    if cfg_renderer is None:
        return None

    if cfg_renderer.type == "DecaLandmarkProjector":
        from inferno.models.temporal.Renderers import FlameLandmarkProjector
        renderer = FlameLandmarkProjector(cfg_renderer)
    elif cfg_renderer.type == "DecaRenderer":
        from inferno.models.temporal.Renderers import FlameRenderer
        renderer = FlameRenderer(cfg_renderer)
    else: 
        raise ValueError(f"Unsupported renderer type: '{cfg_renderer.type}'")
    return renderer


def losses_from_cfg(losses_cfg, device): 
    loss_functions = Munch()
    for loss_name, loss_cfg in losses_cfg.items():
        loss_type = loss_name if 'type' not in loss_cfg.keys() else loss_cfg['type']
        if loss_type == "emotion_loss":
            assert 'emotion_loss' not in loss_functions.keys() # only allow one emotion loss
            assert not loss_cfg.trainable # only fixed loss is supported
            loss_func = create_emo_loss(device, 
                                        emoloss=loss_cfg.network_path,
                                        trainable=loss_cfg.trainable, 
                                        emo_feat_loss=loss_cfg.emo_feat_loss,
                                        normalize_features=loss_cfg.normalize_features, 
                                        dual=False)
            loss_func.eval()
            loss_func.requires_grad_(False)
        elif loss_type == "lip_reading_loss": 
            from inferno.models.temporal.external.LipReadingLoss import \
                LipReadingLoss
            loss_func = LipReadingLoss(device, 
                loss_cfg.metric)
            loss_func.eval()
            loss_func.requires_grad_(False)
        elif loss_type == "face_recognition":
            raise NotImplementedError("TODO: implement face recognition loss")
        #     loss_func = FaceRecognitionLoss(loss_cfg)
        elif loss_type == "au_loss": 
            raise NotImplementedError("TODO: implement AU loss")
        elif loss_type == "landmark_loss_mediapipe":
            loss_func = MediaPipeLandmarkLoss(device, loss_cfg.metric)
        elif loss_type == "landmark_loss_fan_contour": 
            loss_func = FanContourLandmarkLoss(device, loss_cfg.metric)
        elif loss_type == "landmark_loss_fan": 
            loss_func = LandmarkLoss(device, loss_cfg.metric)
        elif loss_type == "lip_distance_mediapipe": 
            loss_func = MediaPipeLipDistanceLoss(device, loss_cfg.metric)
        elif loss_type == "mouth_corner_distance_mediapipe": 
            loss_func = MediaPipeMouthCornerLoss(device, loss_cfg.metric)
        elif loss_type == "eye_distance_mediapipe": 
            loss_func = MediaPipleEyeDistanceLoss(device, loss_cfg.metric)
        elif loss_type == "photometric_loss":
            loss_func = PhotometricLoss(device, loss_cfg.metric)
        elif loss_type == "vgg_loss":
            loss_func = VGG19Loss(device, loss_cfg.metric)
        elif loss_type == "expression_reg":
            loss_func = GaussianRegLoss()
        elif loss_type == "tex_reg":
            loss_func = GaussianRegLoss()
        elif loss_type == "light_reg":
            loss_func = LightRegLoss()
        else: 
            raise NotImplementedError(f"Unsupported loss type: '{loss_type}'")
        loss_functions[loss_name] = loss_func

    return loss_functions


def unring_view_dict(value, shape): 
    if isinstance(value, torch.Tensor):
        return value.view(shape, *value.shape[2:])
    elif isinstance(value, dict):
        return {k: unring_view_dict(v, shape) for k, v in value.items()}
    elif isinstance(value, list):
        return [unring_view_dict(v, shape) for v in value]


def rering_view_dict(value, ring_size): 
    if isinstance(value, torch.Tensor):
        B = value.shape[0] 
        batch_size = B // ring_size 
        assert B % ring_size == 0, f"Batch size must be divisible by ring size but {B} mod {ring_size} == {B%ring_size}"
        return value.view(batch_size, ring_size, *value.shape[1:])
    elif isinstance(value, dict):
        return {k: rering_view_dict(v, ring_size) for k, v in value.items()}
    elif isinstance(value, list):
        return [rering_view_dict(v, batch_size, ring_size) for v in value]


class FaceReconstructionBase(LightningModule):
    """
    DecaModule is a PL module that implements DECA-inspired face reconstruction networks. 
    """

    def __init__(self, cfg, 
                *args: Any, 
                **kwargs: Any) -> None:
        """
        :param model_params: a DictConfig of parameters about the model itself
        :param learning_params: a DictConfig of parameters corresponding to the learning process (such as optimizer, lr and others)
        :param inout_params: a DictConfig of parameters about input and output (where checkpoints and visualizations are saved)
        """
        super().__init__()
        self.cfg = cfg

        self.face_encoder = encoder_from_cfg(cfg)
        self.shape_model = shape_model_from_cfg(cfg)
        # self.preprocessor = preprocessor
        self.renderer = renderer_from_cfg(cfg)

        self._setup_losses()

    def _setup_losses(self):
        # set up losses that need instantiation (such as loading a network, ...)
        self.losses = losses_from_cfg(self.cfg.learning.losses, self.device)
        self.metrics = losses_from_cfg(self.cfg.learning.metrics, self.device)


    def to(self, device=None, **kwargs):
        super().to(device=device, **kwargs)
        for key, value in self.losses.items():
            self.losses[key] = value.to(device)
        for key, value in self.metrics.items():
            self.metrics[key] = value.to(device)
        return self


    def get_trainable_parameters(self):
        trainable_params = []
        trainable_params += self.face_encoder.get_trainable_parameters()
        if self.shape_model is not None:
            trainable_params += self.shape_model.get_trainable_parameters()
        if self.renderer is not None:
            trainable_params += self.renderer.get_trainable_parameters()
        return trainable_params


    def configure_optimizers(self):
        trainable_params = []
        trainable_params += list(self.get_trainable_parameters())

        if self.cfg.learning.optimizer == 'Adam':
            opt = torch.optim.Adam(
                trainable_params,
                lr=self.cfg.learning.learning_rate,
                amsgrad=False)
        elif self.cfg.learning.optimizer == 'AdaBound':
            opt = adabound.AdaBound(
                trainable_params,
                lr=self.cfg.learning.learning_rate,
                final_lr=self.cfg.learning.final_learning_rate
            )

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

    def training_step(self, batch, batch_idx, *args, **kwargs):
        training = True 
        # forward pass
        sample = self.forward(batch, train=training, validation=False, **kwargs)
        
        # loss 
        
        # time = timeit.default_timer()
        
        total_loss, losses, metrics = self.compute_loss(sample, training=training, validation=False, **kwargs)
        
        # time_loss = timeit.default_timer() - time
        # print(f"Time loss:\t{time_loss:0.05f}")

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"train/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            # self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss

    def validation_step(self, batch, batch_idx, *args, **kwargs): 
        training = False 

        # forward pass
        sample = self.forward(batch, train=training, validation=True, teacher_forcing=True, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(sample, training=training, validation=True, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"val_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"val/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            # self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended
        
        return total_loss, losses_and_metrics_to_log

    def get_input_image_size(self): 
        return (self.cfg.model.image_size, self.cfg.model.image_size)

    def uses_texture(self):
        """
        Check if the model uses texture
        """
        return self.cfg.model.use_texture

    def visualize(self, visdict, savepath, catdim=1):
        return self.deca.visualize(visdict, savepath, catdim)

    def train(self, mode: bool = True):
        super().train(mode) 

        return self

    # def to(self, *args, **kwargs):
    #     super().to(*args, **kwargs)
    #     return self

    # def cuda(self, device=None):
    #     super().cuda(device)
    #     return self

    # def cpu(self):
    #     super().cpu()
    #     return self

    def forward(self, batch, training=True, validation=False, render=True, **kwargs):
        if training or validation:
            if "image" not in batch.keys():
                if "video_masked" in batch.keys():
                    batch["image"] = batch["video_masked"]
                    batch["image_original"] = batch["video"]
                else:
                    batch["image"] = batch["video"]
                    batch["image_original"] = batch["video"]

            if "mica_video_masked" in batch.keys():
                batch["mica_images"] = batch["mica_video_masked"]
                batch["mica_images_original"]  = batch["mica_video"]
            elif "mica_video" in batch.keys():
                batch["mica_images"] = batch["mica_video"]
                    
        if "image" not in batch.keys():
            raise ValueError("Batch must contain 'image' key")
        
        ## 0) "unring" the ring dimension if need be 
        # time = timeit.default_timer()
        batch, ring_size = self.unring(batch)
        # time_unring =  timeit.default_timer()

        ## 1) encode images 
        batch = self.encode(batch, training=training)
        # time_encode =   timeit.default_timer()

        ## 3) exchange/disenanglement step (if any)
        batch = self.exchange(batch, ring_size, training=training, validation=validation, **kwargs)
        # time_exchange =   timeit.default_timer()

        ## 4) decode latents
        batch = self.decode(batch, training=training)
        # time_decode =   timeit.default_timer()
        
        ## 5) render
        if render and self.renderer is not None:
            batch = self.render(batch, training=training)
            # time_render =  timeit.default_timer()

        ## 6) rering 
        batch = self.rering(batch, ring_size)
        # time_rering =  timeit.default_timer()

        
        # print(f"Time unring:\t{time_unring - time:0.05f}")
        # print(f"Time encode:\t{time_encode - time_unring:0.05f}")
        # print(f"Time exchange:\t{time_exchange - time_encode:0.05f}")
        # print(f"Time decode:\t{time_decode - time_exchange:0.05f}")
        # print(f"Time render:\t{time_render - time_decode:0.05f}")
        # print(f"Time rering:\t{time_rering - time_render:0.05f}")
        return batch


    def visualize_batch(self, batch, batch_idx, prefix, in_batch_idx=None):
        batch, ring_size = self.unring(batch)
        
        B = batch['image'].shape[0]
        if in_batch_idx is None: 
            in_batch_idx = list(range(B))
        elif isinstance(in_batch_idx, int): 
            in_batch_idx = [in_batch_idx]
        
        # here is where we select the images to visualize or create them if need be 

        visdict = {}
        visdict['image'] = []
        visdict['image_original'] = []
        if 'predicted_image' in batch.keys():
            visdict['predicted_image'] = []
        if 'predicted_mask' in batch.keys():
            visdict['predicted_mask'] = []
        visdict['shape_image'] = []

        visdict['landmarks_gt_fan'] =  []
        visdict['landmarks_gt_mediapipe'] = []
        visdict['landmarks_pred_fan'] =  []
        visdict['landmarks_pred_mediapipe'] = []
        
        if isinstance(batch['segmentation'], torch.Tensor):
            visdict["mask"] = []
        else:
            for key, value in batch["segmentation"].items():
                visdict["mask_" + key] = []

        verts = batch['verts']
        trans_verts = batch['trans_verts']
        shape_images = self.renderer.render.render_shape(verts, trans_verts)

        for b in in_batch_idx:
            image = _torch_image2np(batch['image'][b]).clip(0, 1)
            visdict['image'] += [(image * 255.).astype(np.uint8)]
            
            image_original = _torch_image2np(batch['image_original'][b]).clip(0, 1) 
            visdict['image_original'] += [(image_original* 255.).astype(np.uint8)]

            if isinstance(batch['segmentation'], torch.Tensor):
                mask = batch['segmentation'][b].cpu().numpy().clip(0, 1)
                mask = (mask * 255.).astype(np.uint8)
                visdict['mask'] += [mask]
            else: 
                for key, value in batch["segmentation"].items():
                    mask = value[b].cpu().numpy().clip(0, 1)
                    mask = (mask * 255.).astype(np.uint8)
                    visdict["mask_" + key] = [mask]

            if 'predicted_image' in batch.keys():
                visdict['predicted_image'] += [(_torch_image2np(batch['predicted_image'][b]).clip(0, 1) * 255.).astype(np.uint8)]

            if 'predicted_mask' in batch.keys():
                visdict['predicted_mask'] += [(_torch_image2np(batch['predicted_mask'][b]).clip(0, 1) * 255.).astype(np.uint8)]

            if 'fan3d' in batch['landmarks'].keys():
                landmark_gt_fan = util.tensor_vis_landmarks_single_image(
                image_original, batch['landmarks']['fan3d'][b].cpu().numpy()) 
                visdict['landmarks_gt_fan'] += [(landmark_gt_fan * 255.).astype(np.uint8)]
            
            landmarks_gt_mediapipe = draw_mediapipe_landmarks(image_original, 
                        batch['landmarks']['mediapipe'][b].cpu().numpy()).astype(np.uint8)
            visdict['landmarks_gt_mediapipe'] += [landmarks_gt_mediapipe]
            
            landmarks_pred_fan = util.tensor_vis_landmarks_single_image(
                image_original, batch['predicted_landmarks'][b].detach().cpu().numpy())
            visdict['landmarks_pred_fan'] +=  [(landmarks_pred_fan * 255.).astype(np.uint8)]
            landmarks_pred_mediapipe = draw_mediapipe_landmark_flame_subset(
                image_original, batch['predicted_landmarks_mediapipe'][b].detach().cpu().numpy()).astype(np.uint8)
            visdict['landmarks_pred_mediapipe'] += [landmarks_pred_mediapipe]
        
            visdict['shape_image'] += [(_torch_image2np(shape_images[b]) * 255.).astype(np.uint8)]

        return visdict


    def unring(self, batch):
        """
        This is where the "ring" dimension (if any) would get flattented, etc. 
        """
        image = batch['image']
        ndim = len(image.shape) 
        # image is either [B, 3, H, W] or [B, K, 3, H, W]
        ring_size = -1
        if ndim == 5: 
            B, K, C, H, W = image.shape
            ring_size = K
            # for every entry in batch (that corresponds to each image), undo the ring dimension
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.ndim <= 1:
                    continue
                batch[key] = unring_view_dict(value, B*K)
                # if value.shape[0] == B and value.shape[1] == K: 
                #     batch[key] = value.view(B*K, *value.shape[2:]) 
        check_nan(batch)
        return batch, ring_size
    
    def rering(self, batch, ring_size):
        """
        This is where the "ring" dimension (if any) would get restored, etc. 
        """
        if ring_size == -1:
            return batch
        # for every entry in batch (that corresponds to each image), undo the ring dimension
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim <= 1:
                continue
            batch[key] = rering_view_dict(value, ring_size)
        check_nan(batch)
        return batch

    def exchange(self, batch, ring_size, training, validation, **kwargs):
        """
        This is where disentanglement/exchange step would happen (such as DECA-like shape exchange if any). 
        By default there's nothing. Implement in a sub-class if need be. 
        """
        return batch

    def encode(self, batch, training=True, validation=False, return_features=False):
        """
        Forward encoding pass of the model. Takes a batch of images and returns the corresponding latent codes for each image.
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size]. 
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'. 
        For a testing pass, the images suffice. 
        :param training: Whether the forward pass is for training or testing.
        """
        batch = self.face_encoder(batch, return_features=return_features)
        check_nan(batch)
        return batch

    def decode(self, batch, training=True, validation=False):
        """
        Decodes the predicted latents into the predicted shape
        """
        batch = self.shape_model(batch)
        check_nan(batch)
        return batch
    
    def render(self, batch, training=True, validation=False):
        """
        Renders the predicted shape
        """
        batch = self.renderer(batch)
        check_nan(batch)
        return batch

    def compute_loss(self, sample, training, validation): 
        """
        Compute the loss for the given sample. 

        """
        losses = {}
        # loss_weights = {}
        metrics = {}

        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            assert loss_name not in losses.keys()
            losses["loss_" + loss_name] = self.compute_loss_term(sample, training, validation, loss_name, loss_cfg, self.losses)

        for metric_name, metric_cfg in self.cfg.learning.metrics.items():
            assert metric_name not in metrics.keys()
            with torch.no_grad():
                metrics["metric_" + metric_name] = self.compute_loss_term(sample, training, validation, metric_name, metric_cfg, self.metrics)

        total_loss = None
        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            term = losses["loss_" + loss_name] 
            if term is not None:
                if isinstance(term, torch.Tensor) and term.isnan().any():
                    print(f"[WARNING]: loss '{loss_name}' is NaN. Skipping this term.")
                    continue
                if total_loss is None: 
                    total_loss = 0.
                weighted_term =  term * loss_cfg["weight"]
                total_loss = total_loss + weighted_term
                losses["loss_" + loss_name + "_w"] = weighted_term

        losses["loss_total"] = total_loss
        check_nan(losses)
        return total_loss, losses, metrics

    
    def compute_loss_term(self, sample, training, validation, loss_name, loss_cfg, loss_functions):
        loss_type = loss_name if 'loss_type' not in loss_cfg.keys() else loss_cfg['loss_type']
        mask_invalid = loss_cfg.get('mask_invalid', False) # mask invalid frames 

        loss_func = loss_functions[loss_name]

        predicted_key = loss_cfg.get('predicted_key', None)
        target_key = loss_cfg.get('target_key', None)

        if mask_invalid:
            if mask_invalid == "mediapipe_landmarks": 
                # frames with invalid mediapipe landmarks will be masked for loss computation
                mask = sample["landmarks_validity"]["mediapipe"].to(dtype=torch.bool)
                # dict_get(sample, mask_invalid)
            else: 
                raise ValueError(f"mask_invalid value of '{mask_invalid}' not supported")
        else:
            mask = None
        
        if not target_key: ## UNARY LOSSES
            predicted = sample[predicted_key] 
            loss_value = loss_func(predicted, mask=mask)

        else: ## BINARY LOSSES
            predicted = dict_get(sample, predicted_key)
            target = dict_get(sample, target_key )

            # if is image-based loss 
            if isinstance(loss_func, (PhotometricLoss, LipReadingLoss, VGG19Loss, EmoNetLoss)):
                masking = loss_cfg.masking_type 

                # retrieve the mask
                if masking == "rendered": 
                    image_mask = sample["predicted_mask"]
                elif masking == 'none':
                    image_mask = None
                else:
                    try:
                        gt_mask = dict_get(sample, loss_cfg.get('mask_key', None)) 
                    except Exception: 
                        try:
                            gt_mask = sample["segmentation"][loss_cfg.get('mask_key', None)]
                        except Exception:
                            gt_mask = sample["segmentation"] 
                            assert isinstance(gt_mask, torch.Tensor)
                    if masking == "gt":
                        image_mask = gt_mask
                    elif masking == "gt_rendered_union": 
                        image_mask = torch.logical_or(gt_mask, sample["predicted_mask"])
                    elif masking == "gt_rendered_intersection": 
                        image_mask = torch.logical_and(gt_mask, sample["predicted_mask"])
                    else: 
                        raise ValueError(f"Unsupported masking type: '{masking}'")
                        
                
                # apply mask
                if image_mask is not None: 
                    if image_mask.ndim != predicted.ndim: 
                        # unsqueeze the channel dimension
                        image_mask = image_mask.unsqueeze(-3)
                        assert image_mask.ndim == predicted.ndim
                    predicted = predicted * image_mask
                    target = target * image_mask

            if isinstance(loss_func, LipReadingLoss):
                # LipReadingLoss expects images of shape (B, T, C, H, W)
                if predicted.ndim != 5:
                    predicted = predicted.unsqueeze(1)
                    target = target.unsqueeze(1)
                assert predicted.ndim == 5
                assert target.ndim == 5

                predicted_mouth = loss_func.crop_mouth(predicted, 
                                    dict_get(sample, loss_cfg.get('predicted_landmarks', None)), 
                                    )
                target_mouth = loss_func.crop_mouth(target,
                                        dict_get(sample, loss_cfg.get('target_landmarks', None)),
                                        )
                sample[predicted_key + "_mouth"] = predicted_mouth
                sample[target_key + "_mouth"] = target_mouth

                if loss_cfg.get('per_frame', True): 
                    # if True, loss is computed per frame (no temporal context)
                    
                    predicted_mouth = predicted_mouth.view(-1, *predicted_mouth.shape[2:])
                    target_mouth = target_mouth.view(-1, *target_mouth.shape[2:])
                    mask = mask.view(-1, *mask.shape[2:])
            
                loss_value = loss_func(target_mouth, predicted_mouth, mask=mask)
            else:
                loss_value = loss_func(predicted, target, mask=mask)
                # loss_value = loss_func(predicted, target)
        return loss_value


    def test_step(self, batch, batch_idx, dataloader_idx=None, **kwargs):
        """
        Testing step override of pytorch lightning module. It makes the encoding, decoding passes, computes the loss and logs the losses/visualizations
        without gradient  
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size]. 
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'. 
        :batch_idx batch index
        """
        training = False
        # forward pass
        sample = self.forward(batch, train=training, validation=False, **kwargs)
        
        # loss 
        
        # time = timeit.default_timer()
        
        total_loss, losses, metrics = self.compute_loss(sample, training=training, validation=False, **kwargs)
        
        # time_loss = timeit.default_timer() - time
        # print(f"Time loss:\t{time_loss:0.05f}")

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"test/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            # self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return None

    @property
    def process(self):
        if not hasattr(self,"process_"):
            import psutil
            self.process_ = psutil.Process(os.getpid())
        return self.process_


    ### STEP ENDS ARE PROBABLY NOT NECESSARY BUT KEEP AN EYE ON THEM IF MULI-GPU TRAINING DOESN'T WORK
    # def training_step_end(self, batch_parts):
    #     return self._step_end(batch_parts)
    #
    # def validation_step_end(self, batch_parts):
    #     return self._step_end(batch_parts)
    #
    # def _step_end(self, batch_parts):
    #     # gpu_0_prediction = batch_parts.pred[0]['pred']
    #     # gpu_1_prediction = batch_parts.pred[1]['pred']
    #     N = len(batch_parts)
    #     loss_dict = {}
    #     for key in batch_parts[0]:
    #         for i in range(N):
    #             if key not in loss_dict.keys():
    #                 loss_dict[key] = batch_parts[i]
    #             else:
    #                 loss_dict[key] = batch_parts[i]
    #         loss_dict[key] = loss_dict[key] / N
    #     return loss_dict


    def vae_2_str(self, valence=None, arousal=None, affnet_expr=None, expr7=None, prefix=""):
        caption = ""
        if len(prefix) > 0:
            prefix += "_"
        if valence is not None and not np.isnan(valence).any():
            caption += prefix + "valence= %.03f\n" % valence
        if arousal is not None and not np.isnan(arousal).any():
            caption += prefix + "arousal= %.03f\n" % arousal
        if affnet_expr is not None and not np.isnan(affnet_expr).any():
            caption += prefix + "expression= %s \n" % AffectNetExpressions(affnet_expr).name
        if expr7 is not None and not np.isnan(expr7).any():
            caption += prefix +"expression= %s \n" % Expression7(expr7).name
        return caption


    @classmethod
    def instantiate(cls, cfg, stage=None, prefix=None, checkpoint=None, checkpoint_kwargs=None) -> 'FaceReconstructionBase':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = FaceReconstructionBase(cfg)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = FaceReconstructionBase.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=True, 
                **checkpoint_kwargs
            )
        return model

