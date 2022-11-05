from gdl.models.talkinghead.TalkingHeadBase import TalkingHeadBase
import numpy as np
from torch import nn
import torch.nn.functional as F
from gdl.models.temporal.BlockFactory import *
from gdl.models.rotation_loss import compute_rotation_loss, convert_rot
from gdl.layers.losses.EmoNetLoss import create_emo_loss
from omegaconf import open_dict
from munch import Munch
from gdl.layers.losses.Masked import MaskedTemporalMSELoss

class FaceFormer(TalkingHeadBase):

    def __init__(self, cfg):
        audio_encoder = audio_model_from_cfg(cfg.model.audio)
        with open_dict( cfg.model.sequence_encoder):
            cfg.model.sequence_encoder.input_feature_dim = audio_encoder.output_feature_dim()
        sequence_encoder = sequence_encoder_from_cfg(cfg.model.sequence_encoder)
        sequence_decoder = sequence_decoder_from_cfg(cfg)
        preprocessor_cfg = cfg.model.get('preprocessor', None)
        preprocessor = preprocessor_from_cfg(preprocessor_cfg) if preprocessor_cfg is not None else None 
        renderer_cfg = cfg.model.get('renderer', None)
        renderer = renderer_from_cfg(renderer_cfg) if renderer_cfg is not None else None 
        
        super().__init__(cfg, 
            audio_model=audio_encoder, 
            sequence_encoder= sequence_encoder, 
            sequence_decoder= sequence_decoder, 
            # shape_model=face_model,
            preprocessor=preprocessor,
            renderer=renderer,
            )

        self._setup_losses()

    def _setup_losses(self):
        self.neural_losses = Munch()
        # set up losses that need instantiation (such as loading a network, ...)
        losses_and_metrics = {**self.cfg.learning.losses, **self.cfg.learning.metrics}
        # for loss_name, loss_cfg in self.cfg.learning.losses.items():
        for loss_name, loss_cfg in losses_and_metrics.items():
            loss_type = loss_name if 'loss_type' not in loss_cfg.keys() else loss_cfg['loss_type']
            if loss_type == "emotion_loss":
                assert 'emotion_loss' not in self.neural_losses.keys() # only allow one emotion loss
                assert not loss_cfg.trainable # only fixed loss is supported
                self.neural_losses.emotion_loss = create_emo_loss(self.device, 
                                            emoloss=loss_cfg.network_path,
                                            trainable=loss_cfg.trainable, 
                                            emo_feat_loss=loss_cfg.emo_feat_loss,
                                            normalize_features=loss_cfg.normalize_features, 
                                            dual=False)
                self.neural_losses.emotion_loss.eval()
                self.neural_losses.emotion_loss.requires_grad_(False)
            elif loss_type == "lip_reading_loss": 
                from gdl.models.temporal.external.LipReadingLoss import LipReadingLoss
                self.neural_losses.lip_reading_loss = LipReadingLoss(self.device)
                self.neural_losses.lip_reading_loss.eval()
                self.neural_losses.lip_reading_loss.requires_grad_(False)
            # elif loss_type == "face_recognition":
            #     self.face_recognition_loss = FaceRecognitionLoss(loss_cfg)

    def to(self, *args, **kwargs):
        if hasattr(self, 'neural_losses'):
            for key, module in self.neural_losses.items():
                self.neural_losses[key] = module.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @property
    def max_seq_length(self):
        if 'positional_encoding' in self.cfg.model.sequence_decoder:
            return self.cfg.model.sequence_decoder.positional_encoding.get('max_len', 5000)
        return 5000

    def _rotation_representation(self):
        return self.sequence_decoder._rotation_representation()

    def _compute_loss(self, sample, training, validation, loss_name, loss_cfg): 
        loss_type = loss_name if 'loss_type' not in loss_cfg.keys() else loss_cfg['loss_type']
        
        mask_invalid = loss_cfg.get('mask_invalid', False) # mask invalid frames 
        if mask_invalid:
            if mask_invalid == "mediapipe_landmarks": 
                # frames with invalid mediapipe landmarks will be masked for loss computation
                mask = sample["landmarks_validity"]["mediapipe"].to(dtype=torch.bool)
            else: 
                raise ValueError(f"mask_invalid value of '{mask_invalid}' not supported")
        else:
            mask = torch.ones((*sample["predicted_vertices"].shape[:2], 1), device=sample["predicted_vertices"].device, dtype=torch.bool)

        # mask = torch.zeros_like(mask)

        B = sample["predicted_vertices"].shape[0] # total batch size
        T = sample["predicted_vertices"].shape[1] # sequence size
        B_orig = B // self.disentangle_expansion_factor(training, validation) # original batch size before disentanglement expansion
        # effective batch size for computation of this particular loss 
        # some loss functions should be computed on only the original part of the batch (vertex error)
        # and some on everything (lip reading loss)
        B_eff = B if loss_cfg.get('apply_on_disentangled', False) else B_orig 

        mask_sum = mask[:B_eff].sum().item()

        if mask_invalid and mask_sum == 0:
            return 0.

        if loss_type in ["jawpose_loss", "jaw_loss"]:
            pred_jaw_ = sample["predicted_jaw"][:B_eff]
            gt_jaw_ = sample["gt_jaw"][:B_eff]
            mask_ = mask[:B_eff, ..., 0]
            loss_value = compute_rotation_loss(
                pred_jaw_, 
                gt_jaw_,  
                r1_input_rep=self._rotation_representation(), 
                r2_input_rep="aa", # gt is in axis-angle
                output_rep=loss_cfg.get('rotation_rep', 'quat'), 
                metric=loss_cfg.get('metric', 'l2'), 
                mask = mask_
                )

        elif loss_type in ["expression_loss", "exp_loss"]:
            min_dim = min(sample["predicted_exp"].shape[-1], sample["gt_exp"].shape[-1])
            pred_exp_ = sample["predicted_exp"][:B_eff][...,:min_dim]
            gt_exp_ = sample["gt_exp"][:B_eff][...,:min_dim]
            mask_ = mask[:B_eff, ..., 0]
            loss_value = MaskedTemporalMSELoss()(pred_exp_, gt_exp_, mask=mask_)

        elif loss_type == "vertex_loss":
            pred_vertices_ = sample["predicted_vertices"][:B_eff]
            gt_vertices_ = sample["gt_vertices"][:B_eff]
            mask_ = mask[:B_eff, ..., 0]
            loss_value = MaskedTemporalMSELoss()(pred_vertices_, gt_vertices_, mask=mask_)

        # velocity losses
        elif loss_type == "vertex_velocity_loss":
            pred_vertices_ = sample["predicted_vertices"][:B_eff]
            gt_vertices_ = sample["gt_vertices"][:B_eff]
            mask_ = mask[:B_eff, ..., 0]
            loss_value = velocity_loss(pred_vertices_, gt_vertices_, MaskedTemporalMSELoss(), mask=mask_)

        elif loss_type in ["expression_velocity_loss", "exp_velocity_loss"]:
            min_dim = min(sample["predicted_exp"].shape[-1], sample["gt_exp"].shape[-1])
            pred_vertices_ = sample["predicted_exp"][:B_eff][...,:min_dim]
            gt_vertices_ = sample["gt_exp"][:B_eff][...,:min_dim]
            mask_ = mask[:B_eff, ..., 0]
            loss_value = velocity_loss(pred_vertices_, gt_vertices_, MaskedTemporalMSELoss(), mask=mask_)

        elif loss_type in ["jawpose_velocity_loss", "jaw_velocity_loss"]:
            mask_ = mask[:B_eff, ..., 0]
            pred_jaw = sample["predicted_jaw"][:B_eff].view(B_eff, T, -1)
            gt_jaw = sample["gt_jaw"][:B_eff].view(B_eff, T, -1)
            loss_value = rotation_velocity_loss(pred_jaw, gt_jaw,
                r1_input_rep=self._rotation_representation(), 
                r2_input_rep="aa", # gt is in axis-angle
                output_rep=loss_cfg.get('rotation_rep', 'quat'), 
                metric=loss_cfg.get('metric', 'l2'), 
                mask = mask_
            )

        elif loss_type == "expression_reg": 
            loss_value = (torch.sum(sample["predicted_exp"] ** 2, dim=-1) / 2).mean()

        elif loss_type == "emotion_loss":
            cam_name = list(sample["predicted_video"].keys())[0]
            assert len(list(sample["predicted_video"].keys())) == 1, "More cameras are not supported yet"
            # T = sample["predicted_video"][cam_name].shape[1] 
            rest = sample["predicted_video"][cam_name].shape[2:]
            for cam_name in sample["predicted_video"].keys():
                target_method = loss_cfg.get('target_method_image', None)
                if target_method is None:
                    target_dict = sample
                else: 
                    target_dict = sample["reconstruction"][target_method]

                gt_vid = target_dict["gt_video"][cam_name][:B_eff].view(B_eff*T, *rest)
                pred_vid = sample["predicted_video"][cam_name][:B_eff].view(B_eff*T, *rest)
                mask_ = mask[:B_eff, ..., 0].view(B_eff*T)
                _, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss, _ = \
                    self.neural_losses.emotion_loss.compute_loss(gt_vid, pred_vid, mask=mask_)

            loss_value = emo_feat_loss_2

        elif loss_type == "emotion_loss_disentangled":
            assert B_orig != B, "No disentanglement done"
            assert self.disentangle_type == "condition_exchange", f"Only 'condition_exchange' is supported"
            cam_name = list(sample["predicted_video"].keys())[0]
            assert len(list(sample["predicted_video"].keys())) == 1, "More cameras are not supported yet"
            # T = sample["predicted_video"][cam_name].shape[1] 
            rest = sample["predicted_video"][cam_name].shape[2:]
            for cam_name in sample["predicted_video"].keys():
                target_method = loss_cfg.get('target_method_image', None)
                if target_method is None:
                    target_dict = sample
                else: 
                    target_dict = sample["reconstruction"][target_method]
                # gt video of the non-exchanged part of the batch -> [:B_eff]
                gt_vid = target_dict["gt_video"][cam_name][:B_orig].view(B_orig*T, *rest)
                # predicted video of the exchanged part of the batch -> [B_eff:]
                pred_vid = sample["predicted_video"][cam_name][B_orig:].view(B_orig*T, *rest) 
                
                condition_indices_1 = sample["condition_indices"][:B_orig]
                condition_indices_2 = sample["condition_indices"][B_orig:]
                input_indices_1 = sample["input_indices"][:B_orig]
                input_indices_2 = sample["input_indices"][B_orig:]

                assert ( condition_indices_1 == condition_indices_2).sum() == 0,  "Disentanglement exchange not done correctly"
                assert ( condition_indices_2 != input_indices_1[condition_indices_2]).sum() == 0, "Disentanglement exchange not done correctly"

                gt_vid = target_dict["gt_video"][cam_name][:B_orig][condition_indices_1].view(B_orig*T, *rest)
                pred_vid = sample["predicted_video"][cam_name][B_orig:][condition_indices_2].view(B_orig*T, *rest) 
                mask_ = mask[:B_orig, ..., 0].view(B_orig*T)
                _, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss, _ = \
                    self.neural_losses.emotion_loss.compute_loss(gt_vid, pred_vid, mask=mask_)   

            loss_value = emo_feat_loss_2

        elif loss_type == "lip_reading_loss":
            B_eff = B if loss_cfg.get('apply_on_disentangled', True) else B_orig  ## WARNING, HERE TRUE BY DEFAULT
            cam_name = list(sample["predicted_mouth_video"].keys())[0]
            assert len(list(sample["predicted_mouth_video"].keys())) == 1, "More cameras are not supported yet"
            # T = sample["predicted_mouth_video"][cam_name].shape[1] 
            rest = sample["predicted_mouth_video"][cam_name][:B_eff].shape[2:]

            for cam_name in sample["predicted_mouth_video"].keys():
                target_method = loss_cfg.get('target_method_image', None)
                if target_method is None:
                    target_dict = sample
                else: 
                    target_dict = sample["reconstruction"][target_method]
                
                # TODO: look into whether the lip reading loss can be used for a batch of videos (currently only one video)
                # gt_vid = target_dict["gt_mouth_video"][cam_name].view(B*T, *rest)
                # pred_vid = sample["predicted_mouth_video"][cam_name].view(B*T, *rest)
                # mask_ = mask.view(B*T, -1).unsqueeze(-1).unsqueeze(-1)
                # mask_ = mask_.expand(*pred_vid.shape)
                # gt_vid = gt_vid[mask_].view(B*mask_sum, *gt_vid.shape[1:])
                # pred_vid = pred_vid[mask_].view(B*mask_sum, *pred_vid.shape[1:])

                # # the old way (loop over batch dimension) - fixed after finding the spectre bug
                # loss_values = []
                # for bi in range(B_eff):
                #     gt_vid = target_dict["gt_mouth_video"][cam_name][bi].view(T, *rest)
                #     pred_vid = sample["predicted_mouth_video"][cam_name][bi].view(T, *rest)
                #     loss = self.neural_losses.lip_reading_loss.compute_loss(gt_vid, pred_vid,  mask[bi])
                #     loss_values.append(loss)
                # loss_value = torch.stack(loss_values).mean()

                # the new way (vectorized) 
                gt_vid = target_dict["gt_mouth_video"][cam_name]
                pred_vid = sample["predicted_mouth_video"][cam_name]
                loss_value = self.neural_losses.lip_reading_loss.compute_loss(gt_vid, pred_vid,  mask)
                # loss_value_ = self.neural_losses.lip_reading_loss.compute_loss(gt_vid, pred_vid,  mask)

                # print("loss_value", loss_value)
                # print("loss_value_", loss_value_)
                # pass
        
        elif loss_type == "lip_reading_loss_disentangled":
            assert B_orig != B, "No disentanglement done"
            assert self.disentangle_type == "condition_exchange", f"Only 'condition_exchange' is supported"
            
            B_eff = B if loss_cfg.get('apply_on_disentangled', True) else B_orig  ## WARNING, HERE TRUE BY DEFAULT
            cam_name = list(sample["predicted_mouth_video"].keys())[0]
            assert len(list(sample["predicted_mouth_video"].keys())) == 1, "More cameras are not supported yet"
            # T = sample["predicted_mouth_video"][cam_name].shape[1] 
            rest = sample["predicted_mouth_video"][cam_name][:B_orig].shape[2:]

            input_indices_1 = sample["input_indices"][:B_orig]
            input_indices_2 = sample["input_indices"][B_orig:]
            condition_indices_1 = sample["condition_indices"][:B_orig]
            condition_indices_2 = sample["condition_indices"][B_orig:]

            assert ( condition_indices_1 == condition_indices_2).sum() == 0, "Disentanglement exchange not done correctly"
            assert ( condition_indices_2 != input_indices_1[condition_indices_2]).sum() == 0, "Disentanglement exchange not done correctly"

            for cam_name in sample["predicted_mouth_video"].keys():
                target_method = loss_cfg.get('target_method_image', None)
                if target_method is None:
                    target_dict = sample
                else: 
                    target_dict = sample["reconstruction"][target_method]
                
                # TODO: look into whether the lip reading loss can be used for a batch of videos (currently only one video)
                # gt_vid = target_dict["gt_mouth_video"][cam_name].view(B*T, *rest)
                # pred_vid = sample["predicted_mouth_video"][cam_name].view(B*T, *rest)
                # mask_ = mask.view(B*T, -1).unsqueeze(-1).unsqueeze(-1)
                # mask_ = mask_.expand(*pred_vid.shape)
                # gt_vid = gt_vid[mask_].view(B*mask_sum, *gt_vid.shape[1:])
                # pred_vid = pred_vid[mask_].view(B*mask_sum, *pred_vid.shape[1:])

                # # the old way (loop over batch dimension) - fixed after finding the spectre bug
                # loss_values = []
                # for bi in range(B_orig):
                #     gt_vid = target_dict["gt_mouth_video"][cam_name][0 + input_indices_1[bi]].view(T, *rest)
                #     pred_vid = sample["predicted_mouth_video"][cam_name][B_orig + input_indices_2[bi]].view(T, *rest)
                #     loss = self.neural_losses.lip_reading_loss.compute_loss(gt_vid, pred_vid,  mask[bi])
                #     loss_values.append(loss)
                # loss_value = torch.stack(loss_values).mean()

                gt_vid = target_dict["gt_mouth_video"][cam_name][:B_orig]
                pred_vid = sample["predicted_mouth_video"][cam_name][B_orig + input_indices_2]
                loss_value = self.neural_losses.lip_reading_loss.compute_loss(gt_vid, pred_vid,  mask)
        
        else: 
            raise ValueError(f"Unknown loss type: {loss_type}")
        return loss_value


    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'FaceFormer':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = FaceFormer(cfg)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = FaceFormer.load_from_checkpoint(
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


def velocity_loss(x1, x2, metric, reduction=None, mask=None):
    v1 = x1[:, 1:, ...] - x1[:, :-1, ...]
    v2 = x2[:, 1:, ...] - x2[:, :-1, ...]
    if mask is not None:
        assert reduction is None, "Cannot use reduction and mask at the same time"
        mask = torch.logical_and(mask[:, 1:, ...], mask[:, :-1, ...])
        return metric(v1, v2, mask=mask)
    reduction = reduction or 'mean'
    return metric(v1, v2, reduction=reduction)


def rotation_velocity_loss(r1, r2,
        r1_input_rep='aa', r2_input_rep='aa', output_rep='aa',
        metric='l2', 
        reduction='mean', 
        mask=None,
        ): 
    B = r1.shape[0]
    T = r1.shape[1]  
    r1 = convert_rot(r1.contiguous().view((B*T,-1)), r1_input_rep, output_rep).view((B,T,-1))
    r2 = convert_rot(r2.contiguous().view((B*T,-1)), r2_input_rep, output_rep).view((B,T,-1))
     
    v1 = r1[:, 1:, ...] - r1[:, :-1, ...]
    v2 = r2[:, 1:, ...] - r2[:, :-1, ...]

    B = v1.shape[0]
    T = v1.shape[1] 
        
    # computing the product of all other dims explicitly (instead of using -1) to avoid shape-related crashes in corner cases
    _other_dims = np.array(v1.shape[2:], dtype=np.int32)
    _collapsed_shape = int(np.prod(_other_dims)) 
    if metric == 'l1': 
        # diff = (r1 - r2)*mask
        # return diff.abs().sum(dim=vec_reduction_dim).sum(dim=bt_reduction_dim) / mask_sum
        if mask is None:
            return F.l1_loss(v1.view(B*T, _collapsed_shape), v2.view(B*T, _collapsed_shape), reduction=reduction) 
        else:
            mask = torch.logical_and(mask[:, 1:, ...], mask[:, :-1, ...])
            return MaskedTemporalMSELoss(reduction=reduction)(v1, v2, mask)
    elif metric == 'l2': 
        if mask is None:
            return F.mse_loss(v1.view(B*T, _collapsed_shape), v2.view(B*T, _collapsed_shape), reduction=reduction)
        else:
            mask = torch.logical_and(mask[:, 1:, ...], mask[:, :-1, ...])
            return MaskedTemporalMSELoss(reduction=reduction)(v1, v2, mask)
        # return diff.square().sum(dim=vec_reduction_dim).sqrt().sum(dim=bt_reduction_dim) / mask_sum ## does not work, sqrt turns weights to NaNa after backward
    else: 
        raise ValueError(f"Unsupported metric for rotation loss: '{metric}'")

    return metric(v1, v2)

