from gdl.models.talkinghead.TalkingHeadBase import TalkingHeadBase
import numpy as np
from torch import nn
import torch.nn.functional as F
from gdl.models.temporal.BlockFactory import *
from gdl.models.rotation_loss import compute_rotation_loss, convert_rot
from omegaconf import open_dict


class FaceFormer(TalkingHeadBase):

    def __init__(self, cfg):
        audio_encoder = audio_model_from_cfg(cfg.model.audio)
        with open_dict( cfg.model.sequence_encoder):
            cfg.model.sequence_encoder.input_feature_dim = audio_encoder.output_feature_dim()
        sequence_encoder = sequence_encoder_from_cfg(cfg.model.sequence_encoder)
        sequence_decoder = sequence_decoder_from_cfg(cfg)
        preprocessor_cfg = cfg.model.get('preprocessor', None)
        preprocessor = preprocessor_from_cfg(preprocessor_cfg) if preprocessor_cfg is not None else None 

        super().__init__(cfg, 
            audio_model=audio_encoder, 
            sequence_encoder= sequence_encoder, 
            sequence_decoder= sequence_decoder, 
            # shape_model=face_model,
            preprocessor=preprocessor,
            )

    @property
    def max_seq_length(self):
        if 'positional_encoding' in self.cfg.model.sequence_decoder:
            return self.cfg.model.sequence_decoder.positional_encoding.get('max_len', 5000)
        return 5000

    def _rotation_representation(self):
        return self.sequence_decoder._rotation_representation()

    def _compute_loss(self, sample, loss_name, loss_cfg): 
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
        mask_sum = mask.sum().item()

        if mask_invalid and mask_sum == 0:
            return 0.

        if loss_type in ["jawpose_loss", "jaw_loss"]:
            mask_ = mask.repeat(1,1, sample["predicted_jaw"].shape[2])
            pred_jaw = sample["predicted_jaw"][mask_].view(mask.shape[0], mask_sum, -1)
            gt_jaw = sample["gt_jaw"][mask_].view(mask.shape[0], mask_sum, -1)
            loss_value = compute_rotation_loss(
                pred_jaw, 
                gt_jaw,  
                r1_input_rep=self._rotation_representation(), 
                r2_input_rep="aa", # gt is in axis-angle
                output_rep=loss_cfg.get('rotation_rep', 'quat'), 
                metric=loss_cfg.get('metric', 'l2'), 
                )
        elif loss_type in ["expression_loss", "exp_loss"]:
            min_dim = min(sample["predicted_exp"].shape[-1], sample["gt_exp"].shape[-1])
            mask_ = mask.repeat(1,1, min_dim)
            pred_exp = sample["predicted_exp"][...,:min_dim][mask_].view(mask.shape[0], mask_sum, -1)[:,:,:min_dim]
            gt_exp = sample["gt_exp"][...,:min_dim][mask_].view(mask.shape[0], mask_sum, -1)
            loss_value = F.mse_loss(pred_exp, gt_exp)
            # loss_value = F.mse_loss(mask*sample["predicted_exp"][..., :min_dim], mask*sample["gt_exp"][..., :min_dim], 
            #     reduction='sum') / mask_sum
        elif loss_type == "vertex_loss":
            mask_ = mask.repeat(1,1, sample["predicted_vertices"].shape[2])
            pred_vertices = sample["predicted_vertices"][mask_].view(mask.shape[0], mask_sum, -1)
            gt_vertices = sample["gt_vertices"][mask_].view(mask.shape[0], mask_sum, -1)
            loss_value = F.mse_loss(pred_vertices, gt_vertices) 

            # v_pred = sample["predicted_vertices"][0][0].view(-1, 3).cpu().numpy()
            # v_gt = sample["gt_vertices"][0][0].view(-1, 3).cpu().numpy()
            # f = self.sequence_decoder.flame.faces_tensor.cpu().numpy() 
            # # import pyvista as pv
            # # mesh_pred = pv.PolyData(v_pred, f)
            # # mesh_gt = pv.PolyData(v_gt, f)
            # # mesh_pred.plot()
            # # mesh_gt.plot()
            # import trimesh
            # import numpy as np
            # mesh_pred = trimesh.Trimesh(vertices=v_pred, faces=f, face_colors=np.ones_like(f, dtype=np.uint8)*128)
            # mesh_gt = trimesh.Trimesh(vertices=v_gt, faces=f, face_colors=np.ones_like(f, dtype=np.uint8)*64)

            # # plot the mesh
            # scene = trimesh.Scene([mesh_pred, mesh_gt])
            # scene.show()
            
        # velocity losses
        elif loss_type == "vertex_velocity_loss":
            mask_ = mask.repeat(1,1, sample["predicted_vertices"].shape[2])
            pred_vertices = sample["predicted_vertices"][mask_].view(mask.shape[0], mask_sum, -1)
            gt_vertices = sample["gt_vertices"][mask_].view(mask.shape[0], mask_sum, -1)
            loss_value = velocity_loss(pred_vertices, gt_vertices, F.mse_loss)
            # loss_value = velocity_loss(mask*sample["predicted_vertices"], mask*sample["gt_vertices"], 
            #     F.mse_loss, reduction='sum') / mask_sum
        elif loss_type in ["expression_velocity_loss", "exp_velocity_loss"]:
            min_dim = min(sample["predicted_exp"].shape[-1], sample["gt_exp"].shape[-1])
            mask_ = mask.repeat(1,1, min_dim)
            pred_exp = sample["predicted_exp"][...,:min_dim][mask_].view(mask.shape[0], mask_sum, -1)[:,:,:min_dim]
            gt_exp = sample["gt_exp"][...,:min_dim][mask_].view(mask.shape[0], mask_sum, -1)
            loss_value = velocity_loss(pred_exp, gt_exp, F.mse_loss)
            # loss_value = velocity_loss(mask*sample["predicted_exp"][..., :min_dim], mask*sample["gt_exp"][..., :min_dim],
            #     F.mse_loss, reduction='sum') / mask_sum
        elif loss_type in ["jawpose_velocity_loss", "jaw_velocity_loss"]:
            mask_ = mask.repeat(1,1, sample["predicted_jaw"].shape[2])
            pred_jaw = sample["predicted_jaw"][mask_].view(mask.shape[0], mask_sum, -1)
            gt_jaw = sample["gt_jaw"][mask_].view(mask.shape[0], mask_sum, -1)
            loss_value = rotation_velocity_loss(pred_jaw, gt_jaw,
                r1_input_rep=self._rotation_representation(), 
                r2_input_rep="aa", # gt is in axis-angle
                output_rep=loss_cfg.get('rotation_rep', 'quat'), 
                metric=loss_cfg.get('metric', 'l2')
            )
            # loss_value = rotation_velocity_loss(mask*sample["predicted_jaw"], mask*sample["gt_jaw"],
            #     r1_input_rep=self._rotation_representation(),
            #     r2_input_rep="aa", # gt is in axis-angle
            #     output_rep=loss_cfg.get('rotation_rep', 'quat'),
            #     metric=loss_cfg.get('metric', 'l2'),
            #     reduction='sum'
            # ) / mask_sum

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


def velocity_loss(x1, x2, metric, reduction='mean'): 
    v1 = x1[:, 1:, ...] - x1[:, :-1, ...]
    v2 = x2[:, 1:, ...] - x2[:, :-1, ...]
    return metric(v1, v2, reduction=reduction)


def rotation_velocity_loss(r1, r2,
        r1_input_rep='aa', r2_input_rep='aa', output_rep='aa',
        metric='l2', 
        reduction='mean'
        ): 
    B = r1.shape[0]
    T = r1.shape[1]  
    r1 = convert_rot(r1.contiguous().view((B*T,-1)), r1_input_rep, output_rep).view((B,T,-1))
    r2 = convert_rot(r2.contiguous().view((B*T,-1)), r2_input_rep, output_rep).view((B,T,-1))
     
    v1 = r1[:, 1:, ...] - r1[:, :-1, ...]
    v2 = r2[:, 1:, ...] - r2[:, :-1, ...]

    B = v1.shape[0]
    T = v1.shape[1] 
        # metric = 'l1'
    _other_dims = np.array(v1.shape[2:], dtype=np.int32)
    _collapsed_shape = int(np.prod(_other_dims))
    if metric == 'l1': 
        # diff = (r1 - r2)*mask
        # return diff.abs().sum(dim=vec_reduction_dim).sum(dim=bt_reduction_dim) / mask_sum
        return F.l1_loss(v1.view(B*T, _collapsed_shape), v2.view(B*T, _collapsed_shape), reduction=reduction) 
    elif metric == 'l2': 
        return F.mse_loss(v1.view(B*T, _collapsed_shape), v2.view(B*T, _collapsed_shape), reduction=reduction)
        # return diff.square().sum(dim=vec_reduction_dim).sqrt().sum(dim=bt_reduction_dim) / mask_sum ## does not work, sqrt turns weights to NaNa after backward
    else: 
        raise ValueError(f"Unsupported metric for rotation loss: '{metric}'")

    return metric(v1, v2)

