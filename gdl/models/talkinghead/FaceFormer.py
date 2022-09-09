from gdl.models.talkinghead.TalkingHeadBase import TalkingHeadBase
from torch import nn
import torch.nn.functional as F
from gdl.models.temporal.BlockFactory import *
from gdl.models.rotation_loss import compute_rotation_loss
from omegaconf import open_dict


class FaceFormer(TalkingHeadBase):

    def __init__(self, cfg):
        audio_encoder = audio_model_from_cfg(cfg.model.audio)
        with open_dict( cfg.model.sequence_encoder):
            cfg.model.sequence_encoder.input_feature_dim = audio_encoder.output_feature_dim()
        sequence_encoder = sequence_encoder_from_cfg(cfg.model.sequence_encoder)
        sequence_decoder = sequence_decoder_from_cfg(cfg)

        super().__init__(cfg, 
            audio_model=audio_encoder, 
            sequence_encoder= sequence_encoder, 
            sequence_decoder= sequence_decoder, 
            # shape_model=face_model,
            )

    def _rotation_representation(self):
        return self.sequence_decoder._rotation_representation()

    def _compute_loss(self, sample, loss_name, loss_cfg): 
        loss_type = loss_name if 'loss_type' not in loss_cfg.keys() else loss_cfg['loss_type']
        
        if loss_type in ["jawpose_loss", "jaw_loss"]:
            loss_value = compute_rotation_loss(sample["predicted_jaw"], sample["gt_jaw"],  
                r1_input_rep=self._rotation_representation(), 
                r2_input_rep="aa", # gt is in axis-angle
                output_rep=loss_cfg.get('rotation_rep', 'quat'), 
                metric=loss_cfg.get('metric', 'l2')
                )
        elif loss_type in ["expression_loss", "exp_loss"]:
            loss_value = F.mse_loss(sample["predicted_exp"], sample["gt_exp"])
        elif loss_type == "vertex_loss":
            loss_value = F.mse_loss(sample["predicted_vertices"], sample["gt_vertices"])
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
            # model = FaceFormer(cfg, prefix)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = FaceFormer.load_from_checkpoint(checkpoint_path=checkpoint, 
                strict=False, **checkpoint_kwargs)
            # if stage == 'train':
            #     mode = True
            # else:
            #     mode = False
            # model.reconfigure(cfg, prefix, downgrade_ok=True, train=mode)
        return model
