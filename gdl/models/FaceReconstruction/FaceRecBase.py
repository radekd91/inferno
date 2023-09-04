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

import gdl.utils.DecaUtils as util
from gdl.datasets.AffectNetDataModule import AffectNetExpressions
from gdl.datasets.AffWild2Dataset import Expression7
from gdl.layers.losses.EmoNetLoss import (create_au_loss, create_emo_loss)
from gdl.layers.losses.VGGLoss import VGG19Loss
from gdl.models.temporal.Bases import ShapeModel
from gdl.models.temporal.external.LipReadingLoss import LipReadingLoss
from gdl.models.temporal.Renderers import Renderer
from gdl.utils.lightning_logging import (_log_array_image, _log_wandb_image,
                                         _torch_image2np)
from gdl.utils.other import class_from_str, get_path_to_assets

from .FaceEncoder import encoder_from_cfg
from .Losses import (FanContourLandmarkLoss, LandmarkLoss,
                     MediaPipeLandmarkLoss, MediaPipeLipDistanceLoss,
                     MediaPipeMouthCornerLoss, MediaPipleEyeDistanceLoss,
                     PhotometricLoss, GaussianRegLoss, LightRegLoss)


def shape_model_from_cfg(cfg): 
    cfg_shape = cfg.model.shape_model

    if cfg_shape.type == "FlameShapeModel":
        from gdl.models.temporal.TemporalFLAME import FlameShapeModel
        shape_model = FlameShapeModel(cfg_shape.flame)
    else: 
        raise ValueError(f"Unsupported shape model type: '{cfg_shape.type}'")
    return shape_model

def renderer_from_cfg(cfg):
    cfg_renderer = cfg.model.renderer

    if cfg_renderer is None:
        return None

    if cfg_renderer.type == "DecaLandmarkProjector":
        from gdl.models.temporal.Renderers import FlameLandmarkProjector
        renderer = FlameLandmarkProjector(cfg_renderer)
    elif cfg_renderer.type == "DecaRenderer":
        from gdl.models.temporal.Renderers import FlameRenderer
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
            from gdl.models.temporal.external.LipReadingLoss import \
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


def rering_view_dict(value, batch_size, ring_size): 
    if isinstance(value, torch.Tensor):
        return value.view(batch_size, ring_size, *value.shape[1:])
    elif isinstance(value, dict):
        return {k: unring_view_dict(v, batch_size, ring_size) for k, v in value.items()}
    elif isinstance(value, list):
        return [unring_view_dict(v, batch_size, ring_size) for v in value]


def dict_get(d, key): 
    if "," not in key: 
        return d[key]
    newkey = key.split(",")[0]
    return dict_get(d[newkey], ",".join(key.split(",")[1:]))


class FaceReconstructionBase(LightningModule):
    """
    DecaModule is a PL module that implements DECA-inspired face reconstruction networks. 
    """

    def __init__(self, cfg, 
                # face_encoder : FaceEncoderBase = None,
                # shape_model: ShapeModel = None,
                # # preprocessor: Optional[Preprocessor] = None,
                # renderer: Optional[Renderer] = None,
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
        # losses_and_metrics = {**self.cfg.learning.losses, **self.cfg.learning.metrics}
        # for loss_name, loss_cfg in self.cfg.learning.losses.items():
        
        self.losses = losses_from_cfg(self.cfg.learning.losses, self.device)
        self.metrics = losses_from_cfg(self.cfg.learning.metrics, self.device)

        # for loss_name, loss_cfg in losses_and_metrics.items():
        #     loss_type = loss_name if 'loss_type' not in loss_cfg.keys() else loss_cfg['loss_type']
        #     if loss_type == "emotion_loss":
        #         assert 'emotion_loss' not in self.loss_functions.keys() # only allow one emotion loss
        #         assert not loss_cfg.trainable # only fixed loss is supported
        #         self.loss_functions.emotion_loss = create_emo_loss(self.device, 
        #                                     emoloss=loss_cfg.network_path,
        #                                     trainable=loss_cfg.trainable, 
        #                                     emo_feat_loss=loss_cfg.emo_feat_loss,
        #                                     normalize_features=loss_cfg.normalize_features, 
        #                                     dual=False)
        #         self.loss_functions.emotion_loss.eval()
        #         self.loss_functions.emotion_loss.requires_grad_(False)
        #     elif loss_type == "lip_reading_loss": 
        #         from gdl.models.temporal.external.LipReadingLoss import LipReadingLoss
        #         self.loss_functions.lip_reading_loss = LipReadingLoss(self.device, 
        #             loss_cfg.get('metric', 'cosine_similarity'))
        #         self.loss_functions.lip_reading_loss.eval()
        #         self.loss_functions.lip_reading_loss.requires_grad_(False)
        #     elif loss_type == "face_recognition":
        #         self.face_recognition_loss = FaceRecognitionLoss(loss_cfg)
        #     elif loss_type == "au_loss": 
        #         raise NotImplementedError("TODO: implement AU loss")
            
        #     elif loss_type == "landmark_loss_mediapipe":
        #         self.loss_functions.landmark_loss_mediapipe = MediaPipeLandmarkLoss(self.device, loss_cfg.loss_type)

        #     elif loss_type == "landmark_loss_fan_contour": 
        #         self.loss_functions.landmark_loss_fan_contour = FanContourLandmarkLoss(self.device, loss_cfg.loss_type)

        #     elif loss_type == "landmark_loss_fan": 
        #         self.loss_functions.landmark_loss_fan = LandmarkLoss(self.device, loss_cfg.loss_type)


            # elif loss_type == "emotion_video_loss": 
            #     self.neural_losses.video_emotion_loss = create_video_emotion_loss(loss_cfg)
            #     self.neural_losses.video_emotion_loss.eval()
            #     self.neural_losses.video_emotion_loss.requires_grad_(False)
        # raise NotImplementedError("TODO: implement metrics")


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
        total_loss, losses, metrics = self.compute_loss(sample, training=training, validation=False, **kwargs)

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
                batch["image"] = batch["video_masked"]
                batch["original_image"] = batch["video"]
                    
        if "image" not in batch.keys():
            raise ValueError("Batch must contain 'image' key")
        
        # 0) "unring" the ring dimension if need be 
        batch, ring_size = self.unring(batch)
        
        # 1) encode images 
        batch = self.encode(batch, training=training)
        
        # 2) exchange/disenanglement step (if any)
        batch = self.exchange(batch, ring_size, training=training, validation=validation, **kwargs)

        # 2) decode latents
        batch = self.decode(batch, training=training)
        
        # 3) render
        if render and self.renderer is not None:
            batch = self.render(batch, training=training)
        
        return batch

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
        return batch, ring_size

    def exchange(self, batch, ring_size, training, validation, **kwargs):
        """
        This is where disentanglement/exchange step would happen (such as DECA-like shape exchange if any). 
        By default there's nothing. Implement in a sub-class if need be. 
        """
        return batch

    def encode(self, batch, training=True, validation=False):
        """
        Forward encoding pass of the model. Takes a batch of images and returns the corresponding latent codes for each image.
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size]. 
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'. 
        For a testing pass, the images suffice. 
        :param training: Whether the forward pass is for training or testing.
        """
        batch = self.face_encoder(batch)
        return batch

    def decode(self, batch, training=True, validation=False):
        """
        Decodes the predicted latents into the predicted shape
        """
        batch = self.shape_model(batch)
        return batch
    
    def render(self, batch, training=True, validation=False):
        """
        Renders the predicted shape
        """
        batch = self.renderer(batch)
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
                weighted_term =  (term * loss_cfg["weight"])
                total_loss = total_loss + weighted_term
                losses["loss_" + loss_name + "_w"] = weighted_term

        losses["loss_total"] = total_loss
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
            predicted = sample[target_key] 
            loss_value = loss_func(predicted, mask=mask)

        else: ## BINARY LOSSES
            predicted = dict_get(sample, predicted_key)
            target = dict_get(sample, target_key )

            if isinstance(loss_func, LipReadingLoss):
                # LipReadingLoss expects images of shape (B, T, C, H, W)
                predicted = predicted.unsqueeze(1)
                target = target.unsqueeze(1)

                predicted_mouth = loss_func.crop_mouth(predicted, 
                                    dict_get(sample, loss_cfg.get('predicted_landmarks', None)), 
                                    convert_grayscale=True
                                    )
                target_mouth = loss_func.crop_mouth(target,
                                        dict_get(sample, loss_cfg.get('target_landmarks', None)),
                                        convert_grayscale=True
                                        )
                sample[predicted_key + "_mouth"] = predicted_mouth
                sample[target_key + "_mouth"] = target_mouth
            
                loss_value = loss_func(target_mouth, predicted_mouth, mask=mask)
            else:
                loss_value = loss_func(predicted, target, mask=mask)
        return loss_value


    def _val_to_be_logged(self, d):
        if not hasattr(self, 'val_dict_list'):
            self.val_dict_list = []
        self.val_dict_list += [d]

    def _train_to_be_logged(self, d):
        if not hasattr(self, 'train_dict_list'):
            self.train_dict_list = []
        self.train_dict_list += [d]

    def _get_logging_prefix(self):
        prefix = self.stage_name + str(self.mode.name).lower()
        return prefix

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Testing step override of pytorch lightning module. It makes the encoding, decoding passes, computes the loss and logs the losses/visualizations
        without gradient  
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size]. 
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'. 
        :batch_idx batch index
        """
        prefix = self._get_logging_prefix()
        losses_and_metrics_to_log = {}

        # if dataloader_idx is not None:
        #     dataloader_str = str(dataloader_idx) + "_"
        # else:
        dataloader_str = ''
        stage_str = dataloader_str + 'test_'

        with torch.no_grad():
            training = False
            testing = True
            values = self.encode(batch, training=training)
            values = self.decode(values, training=training)
            if 'mask' in batch.keys():
                losses_and_metrics = self.compute_loss(values, batch, training=False, testing=testing)
                # losses_and_metrics_to_log = {prefix + '_' + stage_str + key: value.detach().cpu() for key, value in losses_and_metrics.items()}
                losses_and_metrics_to_log = {prefix + '_' + stage_str + key: value.detach().cpu().item() for key, value in losses_and_metrics.items()}
            else:
                losses_and_metric = None

        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = self.current_epoch
        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'step'] = torch.tensor(self.global_step, device=self.device)
        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'batch_idx'] = torch.tensor(batch_idx, device=self.device)
        # losses_and_metrics_to_log[stage_str + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        # losses_and_metrics_to_log[stage_str + 'step'] = torch.tensor(self.global_step, device=self.device)
        # losses_and_metrics_to_log[stage_str + 'batch_idx'] = torch.tensor(batch_idx, device=self.device)
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = self.current_epoch
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'step'] = self.global_step
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'batch_idx'] = batch_idx
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'mem_usage'] = self.process.memory_info().rss
        losses_and_metrics_to_log[stage_str + 'epoch'] = self.current_epoch
        losses_and_metrics_to_log[stage_str + 'step'] = self.global_step
        losses_and_metrics_to_log[stage_str + 'batch_idx'] = batch_idx
        losses_and_metrics_to_log[stage_str + 'mem_usage'] = self.process.memory_info().rss

        if self.logger is not None:
            # self.logger.log_metrics(losses_and_metrics_to_log)
            self.log_dict(losses_and_metrics_to_log, sync_dist=True, on_step=False, on_epoch=True)

        # if self.global_step % 200 == 0:
        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']

        if self.deca.config.test_vis_frequency > 0:
            # Log visualizations every once in a while
            if batch_idx % self.deca.config.test_vis_frequency == 0:
                # if self.trainer.is_global_zero:
                visualizations, grid_image = self._visualization_checkpoint(values['verts'], values['trans_verts'], values['ops'],
                                               uv_detail_normals, values, self.global_step, stage_str[:-1], prefix)
                visdict = self._create_visualizations_to_log(stage_str[:-1], visualizations, values, batch_idx, indices=0, dataloader_idx=dataloader_idx)
                self.logger.log_metrics(visdict)
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


    def _create_visualizations_to_log(self, stage, visdict, values, step, indices=None,
                                      dataloader_idx=None, output_dir=None):
        mode_ = str(self.mode.name).lower()
        prefix = self._get_logging_prefix()

        output_dir = output_dir or self.inout_params.full_run_dir

        log_dict = {}
        for key in visdict.keys():
            images = _torch_image2np(visdict[key])
            if images.dtype == np.float32 or images.dtype == np.float64 or images.dtype == np.float16:
                images = np.clip(images, 0, 1)
            if indices is None:
                indices = np.arange(images.shape[0])
            if isinstance(indices, int):
                indices = [indices,]
            if isinstance(indices, str) and indices == 'all':
                image = np.concatenate([images[i] for i in range(images.shape[0])], axis=1)
                savepath = Path(f'{output_dir}/{prefix}_{stage}/{key}/{self.current_epoch:04d}_{step:04d}_all.png')
                # im2log = Image(image, caption=key)
                if isinstance(self.logger, WandbLogger):
                    im2log = _log_wandb_image(savepath, image)
                else:
                    im2log = _log_array_image(savepath, image)
                name = prefix + "_" + stage + "_" + key
                if dataloader_idx is not None:
                    name += "/dataloader_idx_" + str(dataloader_idx)
                log_dict[name] = im2log
            else:
                for i in indices:
                    caption = key + f" batch_index={step}\n"
                    caption += key + f" index_in_batch={i}\n"
                    if self.emonet_loss is not None:
                        if key == 'inputs':
                            if mode_ + "_valence_input" in values.keys():
                                caption += self.vae_2_str(
                                    values[mode_ + "_valence_input"][i].detach().cpu().item(),
                                    values[mode_ + "_arousal_input"][i].detach().cpu().item(),
                                    np.argmax(values[mode_ + "_expression_input"][i].detach().cpu().numpy()),
                                    prefix="emonet") + "\n"
                            if 'va' in values.keys() and mode_ + "valence_gt" in values.keys():
                                # caption += self.vae_2_str(
                                #     values[mode_ + "_valence_gt"][i].detach().cpu().item(),
                                #     values[mode_ + "_arousal_gt"][i].detach().cpu().item(),
                                caption += self.vae_2_str(
                                    values[mode_ + "valence_gt"][i].detach().cpu().item(),
                                    values[mode_ + "arousal_gt"][i].detach().cpu().item(),
                                    prefix="gt") + "\n"
                            if 'expr7' in values.keys() and mode_ + "_expression_gt" in values.keys():
                                caption += "\n" + self.vae_2_str(
                                    expr7=values[mode_ + "_expression_gt"][i].detach().cpu().numpy(),
                                    prefix="gt") + "\n"
                            if 'affectnetexp' in values.keys() and mode_ + "_expression_gt" in values.keys():
                                caption += "\n" + self.vae_2_str(
                                    affnet_expr=values[mode_ + "_expression_gt"][i].detach().cpu().numpy(),
                                    prefix="gt") + "\n"
                        elif 'geometry_detail' in key:
                            if "emo_mlp_valence" in values.keys():
                                caption += self.vae_2_str(
                                    values["emo_mlp_valence"][i].detach().cpu().item(),
                                    values["emo_mlp_arousal"][i].detach().cpu().item(),
                                    prefix="mlp")
                            if 'emo_mlp_expr_classification' in values.keys():
                                caption += "\n" + self.vae_2_str(
                                    affnet_expr=values["emo_mlp_expr_classification"][i].detach().cpu().argmax().numpy(),
                                    prefix="mlp") + "\n"
                        elif key == 'output_images_' + mode_:
                            if mode_ + "_valence_output" in values.keys():
                                caption += self.vae_2_str(values[mode_ + "_valence_output"][i].detach().cpu().item(),
                                                                 values[mode_ + "_arousal_output"][i].detach().cpu().item(),
                                                                 np.argmax(values[mode_ + "_expression_output"][i].detach().cpu().numpy())) + "\n"

                        elif key == 'output_translated_images_' + mode_:
                            if mode_ + "_translated_valence_output" in values.keys():
                                caption += self.vae_2_str(values[mode_ + "_translated_valence_output"][i].detach().cpu().item(),
                                                                 values[mode_ + "_translated_arousal_output"][i].detach().cpu().item(),
                                                                 np.argmax(values[mode_ + "_translated_expression_output"][i].detach().cpu().numpy())) + "\n"


                        # elif key == 'output_images_detail':
                        #     caption += "\n" + self.vae_2_str(values["detail_output_valence"][i].detach().cpu().item(),
                        #                                  values["detail_output_valence"][i].detach().cpu().item(),
                        #                                  np.argmax(values["detail_output_expression"][
                        #                                                i].detach().cpu().numpy()))
                    savepath = Path(f'{output_dir}/{prefix}_{stage}/{key}/{self.current_epoch:04d}_{step:04d}_{i:02d}.png')
                    image = images[i]
                    # im2log = Image(image, caption=caption)
                    if isinstance(self.logger, WandbLogger):
                        im2log = _log_wandb_image(savepath, image, caption)
                    elif self.logger is not None:
                        im2log = _log_array_image(savepath, image, caption)
                    else:
                        im2log = _log_array_image(None, image, caption)
                    name = prefix + "_" + stage + "_" + key
                    if dataloader_idx is not None:
                        name += "/dataloader_idx_" + str(dataloader_idx)
                    log_dict[name] = im2log
        return log_dict

    def _visualization_checkpoint(self, verts, trans_verts, ops, uv_detail_normals, additional, batch_idx, stage, prefix,
                                  save=False):
        batch_size = verts.shape[0]
        visind = np.arange(batch_size)
        shape_images = self.deca.render.render_shape(verts, trans_verts)
        if uv_detail_normals is not None:
            detail_normal_images = F.grid_sample(uv_detail_normals.detach(), ops['grid'].detach(),
                                                 align_corners=False)
            shape_detail_images = self.deca.render.render_shape(verts, trans_verts,
                                                           detail_normal_images=detail_normal_images)
        else:
            shape_detail_images = None

        visdict = {}
        if 'images' in additional.keys():
            visdict['inputs'] = additional['images'][visind]

        if 'images' in additional.keys() and 'lmk' in additional.keys():
            visdict['landmarks_gt'] = util.tensor_vis_landmarks(additional['images'][visind], additional['lmk'][visind])

        if 'images' in additional.keys() and 'predicted_landmarks' in additional.keys():
            visdict['landmarks_predicted'] = util.tensor_vis_landmarks(additional['images'][visind],
                                                                     additional['predicted_landmarks'][visind])

        if 'predicted_images' in additional.keys():
            visdict['output_images_coarse'] = additional['predicted_images'][visind]

        if 'predicted_translated_image' in additional.keys() and additional['predicted_translated_image'] is not None:
            visdict['output_translated_images_coarse'] = additional['predicted_translated_image'][visind]

        visdict['geometry_coarse'] = shape_images[visind]
        if shape_detail_images is not None:
            visdict['geometry_detail'] = shape_detail_images[visind]

        if 'albedo_images' in additional.keys():
            visdict['albedo_images'] = additional['albedo_images'][visind]

        if 'masks' in additional.keys():
            visdict['mask'] = additional['masks'].repeat(1, 3, 1, 1)[visind]
        if 'albedo' in additional.keys():
            visdict['albedo'] = additional['albedo'][visind]

        if 'predicted_detailed_image' in additional.keys() and additional['predicted_detailed_image'] is not None:
            visdict['output_images_detail'] = additional['predicted_detailed_image'][visind]

        if 'predicted_detailed_translated_image' in additional.keys() and additional['predicted_detailed_translated_image'] is not None:
            visdict['output_translated_images_detail'] = additional['predicted_detailed_translated_image'][visind]

        if 'shape_detail_images' in additional.keys():
            visdict['shape_detail_images'] = additional['shape_detail_images'][visind]

        if 'uv_detail_normals' in additional.keys():
            visdict['uv_detail_normals'] = additional['uv_detail_normals'][visind] * 0.5 + 0.5

        if 'uv_texture_patch' in additional.keys():
            visdict['uv_texture_patch'] = additional['uv_texture_patch'][visind]

        if 'uv_texture_gt' in additional.keys():
            visdict['uv_texture_gt'] = additional['uv_texture_gt'][visind]

        if 'translated_uv_texture' in additional.keys() and additional['translated_uv_texture'] is not None:
            visdict['translated_uv_texture'] = additional['translated_uv_texture'][visind]

        if 'uv_vis_mask_patch' in additional.keys():
            visdict['uv_vis_mask_patch'] = additional['uv_vis_mask_patch'][visind]

        if save:
            savepath = f'{self.inout_params.full_run_dir}/{prefix}_{stage}/combined/{self.current_epoch:04d}_{batch_idx:04d}.png'
            Path(savepath).parent.mkdir(exist_ok=True, parents=True)
            visualization_image = self.deca.visualize(visdict, savepath)
            return visdict, visualization_image[..., [2, 1, 0]]
        else:
            visualization_image = None
            return visdict, None


    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'FaceReconstructionBase':
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
                strict=False, 
                **checkpoint_kwargs
            )
        return model

