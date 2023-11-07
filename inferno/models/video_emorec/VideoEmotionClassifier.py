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
from typing import Any, Optional, Dict, List
from inferno.models.temporal.Bases import TemporalFeatureEncoder, SequenceClassificationEncoder, Preprocessor, ClassificationHead
from inferno.models.temporal.AudioEncoders import Wav2Vec2Encoder
from inferno.models.temporal.SequenceModels import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from inferno.utils.other import get_path_to_assets
from omegaconf import OmegaConf

class VideoClassifierBase(pl.LightningModule): 

    def __init__(self, 
                 cfg, 
                 preprocessor: Optional[Preprocessor] = None,
                 feature_model: Optional[TemporalFeatureEncoder] = None,
                 fusion_layer: Optional[nn.Module] = None,
                 sequence_encoder: Optional[SequenceClassificationEncoder] = None,
                 classification_head: Optional[ClassificationHead] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.feature_model = feature_model
        self.fusion_layer = fusion_layer
        self.sequence_encoder = sequence_encoder
        self.classification_head = classification_head

    def get_trainable_parameters(self):
        trainable_params = []
        if self.feature_model is not None:
            trainable_params += self.feature_model.get_trainable_parameters()
        if self.sequence_encoder is not None:
            trainable_params += self.sequence_encoder.get_trainable_parameters()
        if self.classification_head is not None:
            trainable_params += self.classification_head.get_trainable_parameters()
        return trainable_params

    @property
    def max_seq_length(self):
        return 5000

    def configure_optimizers(self):
        trainable_params = []
        trainable_params += list(self.get_trainable_parameters())

        if trainable_params is None or len(trainable_params) == 0:
            print("[WARNING] No trainable parameters found.")
            return 

        if self.cfg.learning.optimizer == 'Adam':
            opt = torch.optim.Adam(
                trainable_params,
                lr=self.cfg.learning.learning_rate,
                amsgrad=False)
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

    @torch.no_grad()
    def preprocess_input(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        if self.preprocessor is not None:
            if self.device != self.preprocessor.device:
                self.preprocessor.to(self.device)
            sample = self.preprocessor(sample, input_key="video", train=train, test_time=not train, **kwargs)
        # sample = detach_dict(sample)
        return sample 

    def signal_fusion(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        # video_feat = sample["visual_feature"] # b, t, fv
        # audio_feat = sample["audio_feature"] # b, t, fa 

        modality_list = self.cfg.model.get('modality_list', None)
        modality_features = [sample[key] for key in modality_list]

        if self.cfg.model.fusion_type != "tensor_low_rank": 
            assert self.fusion_layer is None

        if self.cfg.model.fusion_type in ["concat", "cat", "concatenate"]:
            fused_feature = torch.cat(modality_features, dim=2) # b, t, fv + fa
        elif self.cfg.model.fusion_type in ["add", "sum"]:
            # stack the tensors and then sum them up 
            fused_feature = torch.cat(modality_features, dim=0)
            fused_feature = fused_feature.sum(dim=0)
        elif self.cfg.model.fusion_type in ["max"]:
            fused_feature = torch.stack(modality_features, dim=0).max(dim=0)
        elif self.cfg.model.fusion_type in ["tensor"]:
            for fi, feat in enumerate(modality_features): 
                modality_features[fi] = torch.cat([feat, torch.ones(*feat.shape[:-1], 1, device=feat.device)], dim=-1)
            if len(modality_features) == 1:
                raise ValueError(f"Unsupported fusion type {self.cfg.model.fusion_type} for {len(modality_features)}")
            elif len(modality_features) == 2:
                # concatenate one to each feature 
                fused_feature = torch.einsum("bti,btj->btij", modality_features[0], modality_features[1])
                fused_feature = fused_feature.view(fused_feature.shape[0], fused_feature.shape[1], -1)
            elif len(modality_features) == 3: 
                fusion_cfg = self.cfg.model.get("fusion_cfg", None) 
                n_modal = fusion_cfg.get('num_rank', len(modality_features))
                if n_modal == 2:
                    # outer product along the last dimensions
                    fused_01 = torch.einsum("bti,btj->btij", modality_features[0], modality_features[1])
                    fused_12 = torch.einsum("bti,btj->btij", modality_features[1], modality_features[2])
                    fused_20 = torch.einsum("bti,btj->btij", modality_features[2], modality_features[0])
                    fused_feature = torch.stack([fused_01, fused_12, fused_20], dim=-1)
                    fused_feature = fused_feature.view(fused_feature.shape[0], fused_feature.shape[1], -1)
                elif n_modal == 3:
                    # outer product along the last dimensions
                    fused_01 = torch.einsum("bti,btj->btij", modality_features[0], modality_features[1])
                    fused_012 = torch.einsum("btij,btk->btijk", fused_12, modality_features[2])
                    fused_feature = fused_012.view(fused_012.shape[0], fused_012.shape[1], -1)
                else: 
                    raise ValueError(f"Unsupported fusion type {self.cfg.model.fusion_type} for {len(modality_features)} modalities and {n_modal} ranks")
            else: 
                raise ValueError(f"Unsupported fusion type {self.cfg.model.fusion_type} for {len(modality_features)} modalities")
        elif self.cfg.model.fusion_type in ["tensor_low_rank"]:
            fused_feature = self.fusion_layer(modality_features)
        else: 
            raise ValueError(f"Unknown fusion type {self.fusion_type}")
        sample["hidden_feature"] = fused_feature
        
        # if self.post_fusion_projection is not None: 
        #     sample["fused_feature"] = self.post_fusion_projection(sample["fused_feature"]) 
        # if self.post_fusion_norm is not None: 
        #     sample["fused_feature"] = self.post_fusion_norm(sample["fused_feature"])
        return sample

    def is_multi_modal(self):
        modality_list = self.cfg.model.get('modality_list', None) 
        return modality_list is not None and len(modality_list) > 1

    def forward(self, sample: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        """
        sample: Dict[str, torch.Tensor]
            - gt_emo_feature: (B, T, F)
        """
        # T = sample[input_key].shape[1]
        if "gt_emo_feature" in sample:
            T = sample['gt_emo_feature'].shape[1]
        else: 
            T = sample['video'].shape[1]
        if self.max_seq_length < T: # truncate
            print("[WARNING] Truncating audio sequence from {} to {}".format(T, self.max_seq_length))
            sample = truncate_sequence_batch(sample, self.max_seq_length)

        # preprocess input (for instance get 3D pseudo-GT )
        sample = self.preprocess_input(sample, train=train, **kwargs)
        check_nan(sample)

        if self.feature_model is not None:
            sample = self.feature_model(sample, train=train, **kwargs)
            check_nan(sample)
        else:
            input_key = "gt_emo_feature" # TODO: this needs to be redesigned 
            sample["hidden_feature"] = sample[input_key]


        if self.is_multi_modal():
            sample = self.signal_fusion(sample, train=train, **kwargs)

        if self.sequence_encoder is not None:
            sample = self.sequence_encoder(sample) #, train=train, validation=validation, **kwargs)
            check_nan(sample)

        if self.classification_head is not None:
            sample = self.classification_head(sample)
            check_nan(sample)

        return sample

    def compute_loss(self, sample, training, validation): 
        """
        Compute the loss for the given sample. 
        """
        losses = {}
        metrics = {}

        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            assert loss_name not in losses.keys()
            losses["loss_" + loss_name] = self._compute_loss(sample, loss_name, loss_cfg)
            # losses["loss_" + loss_name] = self._compute_loss(sample, training, validation, loss_name, loss_cfg)

        for metric_name, metric_cfg in self.cfg.learning.metrics.items():
            assert metric_name not in metrics.keys()
            with torch.no_grad():
                metrics["metric_" + metric_name] = self._compute_loss(sample, metric_name, metric_cfg)
                # metrics["metric_" + metric_name] = self._compute_loss(sample, training, validation, metric_name, metric_cfg)

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

        # def _compute_loss(self, sample, training, validation, loss_name, loss_cfg): 
        #     raise NotImplementedError("Please implement this method in your child class")

    def _compute_loss(self, sample, loss_name, loss_cfg): 
        # TODO: this could be done nicer (have a dict with name - loss functor)
        loss_type = loss_name if 'loss_type' not in loss_cfg.keys() else loss_cfg['loss_type']

        if "cross_entropy" in loss_type: 
            label = sample[loss_cfg["output_key"]]
            if loss_cfg["output_key"] == "gt_expression_intensity":
                label -= 1 # expression intensity is in 1-3 range, but we need 0-2 for cross entropy
            loss_value = F.cross_entropy(sample[loss_cfg["input_key"]], label)
        else: 
            raise ValueError(f"Unsupported loss type: '{loss_type}'")
        return loss_value

    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        training = True 
        # forward pass
        sample = self.forward(batch, train=training, validation=False, **kwargs)
        # sample = self.forward(batch, train=training, validation=False, teacher_forcing=False, **kwargs)
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
        # losses_and_metrics_to_log = {"val_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"val/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss, losses_and_metrics_to_log

    def test_step(self, batch, batch_idx, *args, **kwargs):
        training = False 

        # forward pass
        sample = self.forward(batch, train=training, teacher_forcing=False, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(sample, training, validation=False, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"test/" + k: v.item() if isinstance(v, (torch.Tensor,)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        if self.logger is not None:
            # self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss

    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'VideoClassifierBase':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = VideoClassifierBase(cfg, prefix)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = VideoClassifierBase.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                strict=False, 
                **checkpoint_kwargs)
            # if stage == 'train':
            #     mode = True
            # else:
            #     mode = False
            # model.reconfigure(cfg, prefix, downgrade_ok=True, train=mode)
        return model

def sequence_encoder_from_cfg(cfg, feature_dim):
    if cfg.type == "TransformerSequenceClassifier":
        return TransformerSequenceClassifier(cfg, feature_dim)
    elif cfg.type == "GRUSequenceClassifier":
        return GRUSequenceClassifier(cfg, feature_dim)
    else:
        raise ValueError(f"Unknown sequence classifier model: {cfg.model}")

def classification_head_from_cfg(cfg, feature_size, num_classes):
    if cfg.type == "LinearClassificationHead":
        return LinearClassificationHead(cfg, feature_size, num_classes)
    elif cfg.type == "MultiheadLinearClassificationHead":
        return MultiheadLinearClassificationHead(cfg, feature_size, num_classes)
    else:
        raise ValueError(f"Unknown classification head model: {cfg.model}")


class EmoSwin(TemporalFeatureEncoder): 

    def __init__(self, cfg):
        super().__init__()
        swin_cfg_path = Path(cfg.model_path)
        self.trainable = cfg.trainable
        if not swin_cfg_path.is_absolute():
            swin_cfg_path = get_path_to_assets() / "EmotionRecognition" / "image_based_networks" / swin_cfg_path / "cfg.yaml"
        # read the config file using omegaconf
        from inferno.models.EmoSwinModule import EmoSwinModule
        with open(swin_cfg_path, "r") as f:
            swin_cfg = OmegaConf.load(f)
        self.swin = EmoSwinModule(swin_cfg)

        if not self.trainable:
            for param in self.swin.parameters():
                param.requires_grad = False
            self.swin.eval()

    def _forward(self, sample, train=False, desired_output_length=None, **kwargs): 
        swin_batch = {} 
        # collapse the batch dimension and the temporal dimension
        B, T = sample["video"].shape[:2]
        images = sample["video"].view(B*T, *sample["video"].shape[2:])
        swin_batch["image"] = images
        swin_out = self.swin(swin_batch)
        emo_feature =  swin_out["emo_feat_2"]
        emo_feature = emo_feature.view(B, T, -1)
        sample["hidden_feature"] = emo_feature
        return sample


    def forward(self, sample, train=False, desired_output_length=None, **kwargs): 
        if self.trainable:
            return self._forward(sample, train=train, desired_output_length=desired_output_length, **kwargs)
        else:
            with torch.no_grad():
                return self._forward(sample, train=train, desired_output_length=desired_output_length, **kwargs)
            
    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def output_feature_dim(self): 
        return self.swin.swin.num_features


def feature_enc_from_cfg(cfg):
    if cfg is None or not cfg.type: 
        return None
    elif cfg.type == "wav2vec2": 
        encoder = Wav2VecFeature(cfg)
        return encoder
    elif cfg.type == "emoswin":
        encoder = EmoSwin(cfg) 
        return encoder
    # elif cfg.model == "transformer":
    #     return TransformerSequenceFeature(cfg, feature_dim, num_labels)
    else:
        raise ValueError(f"Unknown sequence classifier model: {cfg.model}")

class Wav2VecFeature(TemporalFeatureEncoder): 

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Wav2Vec2Encoder(cfg.model_specifier, cfg.trainable, cfg.get('with_processor', True), 
            expected_fps=cfg.get('model_expected_fps', 50), # 50 fps is the default for wav2vec2 (but not sure if this holds universally)
            target_fps=cfg.get('target_fps', 25), # 25 fps is the default since we use 25 fps for the videos 
            freeze_feature_extractor=cfg.get('freeze_feature_extractor', True),
            dropout_cfg=cfg.get('dropout_cfg', None),
        )
    
    def forward(self, sample, train=False, desired_output_length=None, **kwargs): 
        sample = self.encoder(sample, train=train, desired_output_length=desired_output_length, **kwargs)
        sample["hidden_feature"] = sample["audio_feature"]
        return sample

    def get_trainable_parameters(self): 
        return self.encoder.get_trainable_parameters()

    def output_feature_dim(self): 
        return self.encoder.output_feature_dim()



class MostFrequentEmotionClassifier(VideoClassifierBase): 

    def __init__(self, cfg):
        super().__init__(cfg)
    
    def get_trainable_parameters(self):
        return super().get_trainable_parameters()

    def forward(self, batch, train=True, teacher_forcing=True, **kwargs):
        sample = batch
        scores = sample['gt_expression'].mean(dim=1)
        sample['predicted_logits'] = scores
        return sample

    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'VideoEmotionClassifier':
        """
        Function that instantiates the model from checkpoint or config
        """
        model = MostFrequentEmotionClassifier(cfg)
        return model


class LowRankTensorFusion(torch.nn.Module): 

    """
    Inspired by: https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/master/model.py
    """

    def __init__(self, rank, feature_dims, output_dim) -> None:
        super().__init__()
        self.rank = rank
        self.feature_dims = feature_dims 
        self.output_dim = output_dim

        factor_list = [] 
        for i in range(len(feature_dims)):
            factor_list.append(nn.Parameter(torch.Tensor(self.rank, self.feature_dims[i] + 1, self.output_dim)))
        self.factors = nn.ParameterList(factor_list)

        # self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob) 
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))
        
        for param in self.factors:
            nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def output_feature_dim(self): 
        return self.output_dim

    def forward(self, modality_features): 
        assert len(modality_features) == len(self.factors)
        _modality_feats = []
        _sizes = []
        T = None
        B = None
        for i, feat in enumerate(modality_features): 
            _modality_feats += [torch.cat([torch.autograd.Variable(torch.ones(*feat.shape[:-1], 1, device=feat.device), requires_grad=False), feat], dim=-1)]
            if len(feat.shape) == 3: # batch, time, feat
                # _sizes += [feat.shape[0:2]]
                if T is not None: 
                    assert T == feat.shape[1], "All modalities must have the same time dimension"
                T = feat.shape[1]
                if B is not None: 
                    assert B == feat.shape[0], "All modalities must have the same batch dimension"
                B = feat.shape[0]
                _modality_feats[i] = _modality_feats[i].view(-1, _modality_feats[i].shape[-1])
            assert len(_modality_feats[i].shape) == 2, "All modalities must have the same time dimension"

        fusion_feats = []
        for i, feat in enumerate(_modality_feats):
            fusion_feats += [torch.matmul(feat, self.factors[i])]
        
        fusion_zy = torch.stack(fusion_feats, dim=-1)
        fusion_zy = torch.prod(fusion_zy, dim=-1) 
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        if T is not None:
            output = output.view(B, T, -1)
        else: 
            output = output.view(B, -1)
        return output


class VideoEmotionClassifier(VideoClassifierBase): 

    def __init__(self, 
                 cfg
        ):
        self.cfg = cfg
        preprocessor = None
        feature_model = feature_enc_from_cfg(cfg.model.get('feature_extractor', None))
        fusion_layer = None
        if not self.is_multi_modal():
            feature_size = feature_model.output_feature_dim() if feature_model is not None else cfg.model.input_feature_size
        else: 
            if self.cfg.model.fusion_type == 'tensor':
                assert len(self.cfg.model.modality_list) == 2 
                feature_size = ( cfg.model.input_feature_size + 1) * (feature_model.output_feature_dim() + 1) 
            elif self.cfg.model.fusion_type == 'tensor_low_rank': 
                assert len(self.cfg.model.modality_list) == 2 
                fusion_cfg = self.cfg.model.fusion_cfg
                fusion_layer = LowRankTensorFusion(fusion_cfg.rank, [cfg.model.input_feature_size, feature_model.output_feature_dim()], fusion_cfg.output_dim)
                feature_size = fusion_layer.output_feature_dim()
            else:
                feature_size = feature_model.output_feature_dim() + cfg.model.input_feature_size
        sequence_classifier = sequence_encoder_from_cfg(cfg.model.get('sequence_encoder', None), feature_size)
        classification_head = classification_head_from_cfg(cfg.model.get('classification_head', None), 
                                                           sequence_classifier.encoder_output_dim(), 
                                                           cfg.model.output.num_classes, 
                                                           )

        super().__init__(cfg,
            preprocessor = preprocessor,
            feature_model = feature_model,
            fusion_layer = fusion_layer,
            sequence_encoder = sequence_classifier,  
            classification_head = classification_head,  
        )


    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'VideoEmotionClassifier':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = VideoEmotionClassifier(cfg)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = VideoEmotionClassifier.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=False, 
                **checkpoint_kwargs
            )
            # if stage == 'train':
            #     mode = True
            # else:
            #     mode = False
            # model.reconfigure(cfg, prefix, downgrade_ok=True, train=mode)
        return model


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


def truncate_sequence_batch(sample: Dict, max_seq_length: int) -> Dict:
    """
    Truncate the sequence to the given length. 
    """
    # T = sample["audio"].shape[1]
    # if max_seq_length < T: # truncate
    for key in sample.keys():
        if isinstance(sample[key], torch.Tensor): # if temporal element, truncate
            if sample[key].ndim >= 3:
                sample[key] = sample[key][:, :max_seq_length, ...]
        elif isinstance(sample[key], Dict): 
            sample[key] = truncate_sequence_batch(sample[key], max_seq_length)
        elif isinstance(sample[key], List):
            pass
        else: 
            raise ValueError(f"Invalid type '{type(sample[key])}' for key '{key}'")
    return sample
