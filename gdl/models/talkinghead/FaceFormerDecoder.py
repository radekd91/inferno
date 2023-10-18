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

import torch 
from torch import nn
import math
from gdl.models.rotation_loss import convert_rot
from gdl.models.MLP import MLP
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from omegaconf import DictConfig, OmegaConf, open_dict
from gdl.models.temporal.PositionalEncodings import positional_encoding_from_cfg
from gdl.models.temporal.TransformerMasking import init_mask_future, init_mask, init_faceformer_biased_mask, init_faceformer_biased_mask_future
from gdl.models.temporal.motion_prior.MotionPrior import MotionPrior
from gdl.models.temporal.motion_prior.L2lMotionPrior import L2lVqVae, create_squasher
    
from gdl.utils.other import class_from_str
from gdl.models.IO import get_checkpoint_with_kwargs
import sys


class AutoRegressiveDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, sample, train=False, teacher_forcing=True): 
        # teacher_forcing = not train
        hidden_states = sample["seq_encoder_output"]
        sample["hidden_feature"] = hidden_states # first "hidden state" is the audio feature
        if teacher_forcing:
            sample = self._teacher_forced_step(sample)
        else:
            # rec_dict = batch if self.rec_type is None else batch["reconstruction"][self.rec_type]
            # num_frames = rec_dict["gt_vertices"].shape[1] if "gt_vertices" in rec_dict.keys() else hidden_states.shape[1]
            num_frames = sample["gt_vertices"].shape[1] if "gt_vertices" in sample.keys() else hidden_states.shape[1]
            num_frames = min(num_frames, self._max_auto_regressive_steps())
            for i in range(num_frames):
                sample = self._autoregressive_step(sample, i)
        sample = self._post_prediction(sample)
        return sample

    # def rec_dict(batch): 
    #     return get_rec_dict(batch, self.rec_type)

    # @property
    # def rec_type(self):
    #     None

    def _max_auto_regressive_steps(self):
        raise NotImplementedError("")

    def _teacher_forced_step(self): 
        raise NotImplementedError("")

    def _autoregressive_step(self): 
        raise NotImplementedError("")

    def _post_prediction(self): 
        """
        Adds the template vertices to the predicted offsets
        """
        raise NotImplementedError("")


def style_from_cfg(cfg):
    style_type = cfg.get('style_embedding', 'onehot_linear')
    if isinstance(style_type, (DictConfig, dict)): 
        # new preferred way
        style_cfg = style_type
        style_type = style_cfg.type 
        if style_type == 'emotion_linear' or style_type == 'video_emotion_linear':
            if style_cfg.use_shape: 
                with open_dict(style_cfg) as c:
                    c.shape_dim = cfg.flame.n_shape
            return LinearEmotionCondition(style_cfg, output_dim=cfg.feature_dim)
        elif style_type in ['onehot_linear', 'onehot_identity_linear']:
            return OneHotIdentityCondition(style_cfg, output_dim=cfg.feature_dim)
        raise ValueError(f"Unsupported style embedding type '{style_type}'")

    # old way (for backwards compatibility)
    if style_type == 'onehot_linear':
        return nn.Linear(cfg.num_training_subjects, cfg.feature_dim, bias=False)
    elif style_type == 'none':
        return None
    else:
        raise ValueError(f"Unsupported style embedding type '{style_type}'")


class StyleConditioning(torch.nn.Module): 

    def __init__(self): 
        super().__init__()


    def forward(self, sample, **kwargs): 
        raise NotImplementedError("Subclasses must implement this method")


class OneHotIdentityCondition(StyleConditioning): 

    def __init__(self, cfg, output_dim): 
        super().__init__()
        self.output_dim = output_dim
        self.map = nn.Linear(cfg.num_training_subjects, output_dim, bias=False) 

    def forward(self, sample, **kwargs):
        return self.map(sample["one_hot"])


class EmotionCondition(StyleConditioning): 

    def __init__(self, cfg, output_dim): 
        super().__init__()
        self.output_dim = output_dim
        self.cfg = cfg
        self.condition_dim = 0

        if self.cfg.get('use_video_expression', False):
            self.condition_dim += self.cfg.n_expression

        if self.cfg.get('use_video_feature', False):
            self.condition_dim += self.cfg.n_vid_expression_feature

        if self.cfg.get('gt_expression_label', False):
            self.condition_dim += self.cfg.n_expression
        
        if self.cfg.get('gt_expression_intensity', False):
            self.condition_dim += self.cfg.n_intensities

        if self.cfg.get('gt_expression_identity', False):
            self.condition_dim += self.cfg.n_identities

        if self.cfg.use_expression:
            self.condition_dim += self.cfg.n_expression

        if self.cfg.use_valence:
            self.condition_dim += 1

        if self.cfg.use_arousal:
            self.condition_dim += 1

        if self.cfg.use_emotion_feature:
            self.condition_dim += self.cfg.num_features

        if self.cfg.use_shape:
            self.condition_dim += self.cfg.shape_dim

    def _gather_condition(self, sample):
        condition = []
        try:
            T = sample["gt_vertices"].shape[1]
        except KeyError:
            T = sample["raw_audio"].shape[1]
        if self.cfg.get('use_video_expression', False):
            assert len(sample["gt_emotion_video_logits"]) == 1, "Only one video expression supported" 
            cam = list(sample["gt_emotion_video_logits"].keys())[0]
            video_classification = torch.nn.functional.softmax(sample["gt_emotion_video_logits"][cam], dim=-1)
            # expand to match the sequence length
            # T = sample["gt_vertices"].shape[1]
            video_classification = video_classification.unsqueeze(1).expand(-1, T, -1)
            condition += [video_classification]
        if self.cfg.get('use_video_feature', False): 
            assert len(sample["gt_emotion_video_features"]) == 1, "Only one video expression supported" 
            cam = list(sample["gt_emotion_video_features"].keys())[0]
            video_feats = torch.nn.functional.softmax(sample["gt_emotion_video_features"][cam], dim=-1)
            # expand to match the sequence length
            assert self.cfg.n_vid_expression_feature == video_feats.shape[-1], "Number of video features does not match"

            video_feats = video_feats.unsqueeze(1).expand(-1, T, -1)
            condition += [video_feats]
        if self.cfg.get('gt_expression_label', False): # mead GT expression label
            # T = sample["gt_vertices"].shape[1]
            if "gt_expression_label_condition" in sample:
                expressions = sample["gt_expression_label_condition"]
            else:
                expressions = torch.nn.functional.one_hot(sample["gt_expression_label"], 
                                                        num_classes=self.cfg.n_expression).to(device=sample["gt_expression_label"].device)
                # if expressions.ndim == 3: # B, T, num expressions
                #     expressions = expressions.unsqueeze(1).expand(-1, T, -1)
            if expressions.ndim == 2: # B, num expressions, missing temporal dimension -> expand
                expressions = expressions.unsqueeze(1) 
            if expressions.shape[1] == 1: # B, 1, num expressions -> expand
                expressions = expressions.expand(-1, T, -1)
            expressions = expressions.to(dtype=torch.float32)
            assert expressions.ndim == 3, "Expressions must have 3 dimensions"
            condition += [expressions]
        if self.cfg.get('gt_expression_intensity', False): # mead GT expression intensity
            # T = sample["gt_vertices"].shape[1]
            if "gt_expression_intensity_condition" in sample:
                intensities = sample["gt_expression_intensity_condition"]
            else:
                intensities = torch.nn.functional.one_hot(sample["gt_expression_intensity"] -1, 
                    num_classes=self.cfg.n_intensities).to(device=sample["gt_expression_intensity"].device)
            if intensities.ndim == 2: # B, num intensities, missing temporal dimension -> expand
                intensities = intensities.unsqueeze(1) 
            if intensities.shape[1] == 1: # B, 1, num intensities -> expand    
                intensities = intensities.expand(-1, T, -1)
            intensities = intensities.to(dtype=torch.float32)
            assert intensities.ndim == 3, "Intensities must have 3 dimensions"
            condition += [intensities]

        if self.cfg.get('gt_expression_identity', False): 
            # T = sample["gt_vertices"].shape[1]
            if "gt_expression_identity_condition" in sample:
                identities = sample["gt_expression_identity_condition"]
            else:
                identities = torch.nn.functional.one_hot(sample["gt_expression_identity"], 
                                                        num_classes=self.cfg.n_identities).to(device=sample["gt_expression_identity"].device)
            if identities.ndim == 2: # B, num identities, missing temporal dimension -> expand
                identities = identities.unsqueeze(1)
            if identities.shape[1] == 1: # B, 1, num identities -> expand
                identities = identities.expand(-1, T, -1)
            identities = identities.to(dtype=torch.float32)
            assert identities.ndim == 3, "Identities must have 3 dimensions"
            condition += [identities]

        if self.cfg.use_expression: # pseudo GT  label (from Resnet AffectNet)
            condition += [sample["gt_expression"]] 
        if self.cfg.use_valence:
            condition += [sample["gt_valence"]]
        if self.cfg.use_arousal:
            condition += [sample["gt_arousal"]]
        if self.cfg.use_emotion_feature:
            condition += [sample["gt_emo_feat_2"]]
        if self.cfg.use_shape:
            # rec_dict = get_rec_dict(sample, self.cfg.get('rec_type', None))
            # T = rec_dict["gt_vertices"].shape[1]
            # shape_cond = rec_dict["gt_shape"].unsqueeze(1)
            T = sample["gt_vertices"].shape[1]
            shape_cond = sample["gt_shape"].unsqueeze(1)
            # shape_cond = shape_cond.repeat(1, T, 1)
            shape_cond = shape_cond.expand(shape_cond.shape[0], T, shape_cond.shape[2])
            condition += [shape_cond]

        condition = torch.cat(condition, dim=-1)
        return condition

    def forward(self, sample, **kwargs):
        condition = self._gather_condition(sample)
        return self.map(condition)
        

class LinearEmotionCondition(EmotionCondition): 

    def __init__(self, cfg, output_dim): 
        super().__init__(cfg, output_dim)
        self.map = nn.Linear(self.condition_dim, output_dim, 
            bias=cfg.get('use_bias', True))
            # bias=False)


class FaceFormerDecoderBase(AutoRegressiveDecoder):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # periodic positional encoding 
        # self.PPE = PeriodicPositionalEncoding(cfg.feature_dim, period = cfg.period)
        self.PE = positional_encoding_from_cfg(cfg.positional_encoding, cfg.feature_dim )
        self.max_len = cfg.max_len
        # temporal bias
        self.temporal_bias_type = cfg.get('temporal_bias_type', 'faceformer')
        if self.temporal_bias_type == 'faceformer':
            self.biased_mask = init_faceformer_biased_mask(num_heads = cfg.nhead, max_seq_len = cfg.max_len, period=cfg.period)
        elif self.temporal_bias_type == 'faceformer_future':
            self.biased_mask = init_faceformer_biased_mask_future(num_heads = cfg.nhead, max_seq_len = cfg.max_len, period=cfg.period)
        elif self.temporal_bias_type == 'classic':
            self.biased_mask = init_mask(num_heads = cfg.nhead, max_seq_len = cfg.max_len)
        elif self.temporal_bias_type == 'classic_future':
            self.biased_mask = init_mask_future(num_heads = cfg.nhead, max_seq_len = cfg.max_len)
        elif self.temporal_bias_type == 'none':
            self.biased_mask = None
        else:
            raise ValueError(f"Unsupported temporal bias type '{self.temporal_bias_type}'")

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = cfg.feature_dim, 
            nhead=cfg.nhead, 
            dim_feedforward = cfg.feature_dim, batch_first=True)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_layers)
        self.vertice_map = nn.Linear(cfg.vertices_dim, cfg.feature_dim)

        self.style_type = cfg.get('style_embedding', 'onehot_linear')
        self.obj_vector = style_from_cfg(cfg)

        self.use_alignment_bias = cfg.get('use_alignment_bias', True)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.biased_mask is not None:
            self.biased_mask = self.biased_mask.to(*args, **kwargs)

    def get_shape_model(self):
        return None

    def get_trainable_parameters(self):
        return list(self.parameters())

    def _max_auto_regressive_steps(self):
        if self.biased_mask is not None:
            return self.biased_mask.shape[1]
        else: 
            return self.max_len

    def _autoregressive_step(self, sample, i):
        hidden_states = sample["hidden_feature"]
        if i==0:
            # one_hot = sample["one_hot"]
            # obj_embedding = self.obj_vector(one_hot)
            # vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            # style_emb = vertice_emb
            # sample["style_emb"] = style_emb
            style_emb = self._style(sample, hidden_states.device)
            sample["style_emb"] = style_emb
            vertice_emb = style_emb[:,0:1,:] # take the first to kick off the auto-regressive process
            if self.PE is not None:
                vertices_input = self.PE(style_emb)
            else: 
                vertices_input = vertice_emb
        else:
            vertice_emb = sample["embedded_output"]
            style_emb = sample["style_emb"]
            if self.PE is not None:
                vertices_input = self.PE(vertice_emb)
            else: 
                vertices_input = vertice_emb
        
        vertices_out = self._decode(sample, vertices_input, hidden_states)
        sample["predicted_vertices"] = vertices_out

        new_output = self.vertice_map(vertices_out[:,-1,:]).unsqueeze(1)
        style_T = style_emb.shape[1]
        if style_T == 1: # per sample style embeeding
            new_output = new_output + style_emb
        else: # per sample and frame style embedding
            new_output = new_output + style_emb[:,i:i+1,:]
        vertice_emb = torch.cat((vertice_emb, new_output), 1)
        sample["embedded_output"] = vertice_emb
        return sample

    def _style(self, sample, device):
        if self.obj_vector is None:
            return torch.zeros(1,1,self.cfg.feature_dim).to(device)
        
        if isinstance(self.obj_vector, torch.nn.Linear):
            # backwards compatibility (for FaceFormer's original one-hot identity embedding)
            one_hot = sample["one_hot"]
            obj_embedding = self.obj_vector(one_hot)
            style_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            return style_emb 

        if isinstance(self.obj_vector, StyleConditioning):
            style_emb = self.obj_vector(sample)
            return style_emb

        raise ValueError(f"Unsupported style type '{self.style_type}'")

    def _teacher_forced_step(self, sample): 
        vertices = sample["gt_vertices"]
        template = sample["template"].unsqueeze(1)
        hidden_states = sample["hidden_feature"]
        # one_hot = sample["one_hot"]
        # obj_embedding = self.obj_vector(one_hot)

        # vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
        # style_emb = vertice_emb  
        style_emb = self._style(sample, hidden_states.device)

        vertices_input = torch.cat((template, vertices[:,:-1]), 1) # shift one position
        vertices_input = vertices_input - template
        vertices_input = self.vertice_map(vertices_input)
        vertices_input = vertices_input + style_emb
        if self.PE is not None:
            vertices_input = self.PE(vertices_input)
        # else: 
        #     vertices_input = vertices_input

        vertices_out = self._decode(sample, vertices_input, hidden_states)
        sample["predicted_vertices"] = vertices_out

        return sample

    def _decode(self, sample, vertices_input, hidden_states):
        dev = vertices_input.device
        if self.biased_mask is not None:
            tgt_mask = self.biased_mask[:, :vertices_input.shape[1], :vertices_input.shape[1]].clone().detach().to(device=dev)
            if tgt_mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                tgt_mask = tgt_mask.repeat(vertices_input.shape[0], 1, 1)
        else: 
            tgt_mask = None
        if self.use_alignment_bias:
            # attends to diagonals only
            memory_mask = enc_dec_mask(dev, vertices_input.shape[1], hidden_states.shape[1])
        else:
            # attends to everything
            memory_mask = torch.zeros(vertices_input.shape[1], hidden_states.shape[1], device=dev, dtype=torch.bool)
        transformer_out = self.transformer_decoder(vertices_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)

        vertices_out = self._decode_vertices(sample, transformer_out)
        return vertices_out

    def _post_prediction(self, sample):
        template = sample["template"]
        vertices_out = sample["predicted_vertices"]
        vertices_out = vertices_out + template[:, None, ...]
        sample["predicted_vertices"] = vertices_out
        return sample

    def _decode_vertices(self):
        raise NotImplementedError()



class FaceFormerDecoder(FaceFormerDecoderBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.vertex_map = nn.Linear(cfg.feature_dim, cfg.vertices_dim)

        # the init is done this way in the paper for some reason
        nn.init.constant_(self.vertex_map.weight, 0)
        nn.init.constant_(self.vertex_map.bias, 0)

    def _decode_vertices(self, sample, transformer_out): 
        vertice_out = self.vertex_map(transformer_out)
        return vertice_out


def rotation_rep_size(rep):
    if rep in ["quat", "quaternion"]:
        return 4
    elif rep in ["aa", "axis-angle"]:
        return 3
    elif rep in ["matrix", "mat"]:
        return 9
    elif rep in ["euler", "euler-angles"]:
        return 3
    elif rep in ["rot6d", "rot6dof", "6d"]:
        return 6
    else:
        raise ValueError("Unknown rotation representation: {}".format(rep))


class FlameFormerDecoder(FaceFormerDecoderBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        from gdl.models.DecaFLAME import FLAME
        # from munch import Munch
        # self.flame_config = Munch()
        # self.flame_config.flame_model_path = "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl" 
        # self.flame_config.n_shape = 100 
        # self.flame_config.n_exp = 50
        # self.flame_config.flame_lmk_embedding_path = "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy"
        self.flame_config = cfg.flame
        self.flame = FLAME(self.flame_config)
        pred_dim = 0
        self.predict_exp = cfg.predict_exp
        self.predict_jaw = cfg.predict_jaw
        self.rotation_representation = cfg.get("rotation_representation", "aa")
        self.flame_rotation_rep = "aa"
        if self.predict_exp: 
            pred_dim += self.flame_config.n_exp
        if self.predict_jaw:
            pred_dim += rotation_rep_size(self.rotation_representation)

        self.post_transformer = nn.Linear(cfg.feature_dim, pred_dim)
        # initialize the weights to zero to prevent the model from overshooting in the beginning 
        # and to prevent convergence issues
        nn.init.constant_(self.post_transformer.weight, 0)
        nn.init.constant_(self.post_transformer.bias, 0)

        self.flame_space_loss = cfg.flame_space_loss
        # self.rotation_loss_space = cfg.rotation_loss_space

    def get_shape_model(self):
        return self.flame

    def _rotation_representation(self):
        return self.rotation_representation


    def _decode_vertices(self, sample, transformer_out): 
        template = sample["template"]
        # jaw = sample["gt_jaw"]
        # exp = sample["exp"]

        # self.flame.v_template = template.squeeze(1).view(-1, 3)
        transformer_out = self.post_transformer(transformer_out)
        batch_size = transformer_out.shape[0]
        T_size = transformer_out.shape[1]
        pose_params = self.flame.eye_pose.expand(batch_size*T_size, -1)

        vector_idx = 0
        if self.predict_jaw:
            rotation_rep_size_ = rotation_rep_size(self.rotation_representation)
            jaw_pose = transformer_out[..., :rotation_rep_size_].view(batch_size*T_size, -1)
            vector_idx += rotation_rep_size_
            if self.rotation_representation in ['mat', 'matrix']:
                jaw_pose = jaw_pose.view(-1, 3, 3) 
                U, S, Vh = torch.linalg.svd(jaw_pose) # do SVD to ensure orthogonality
                jaw_pose = U.contiguous()
            
        else: 
            jaw = sample["gt_jaw"][:, :T_size, ...]
            jaw_pose = jaw.view(batch_size*T_size, -1)
        if self.predict_exp: 
            expression_params = transformer_out[..., vector_idx:].view(batch_size*T_size, -1)
            vector_idx += self.flame_config.n_exp
        else: 
            exp = sample["gt_exp"][:, :T_size, ...]
            expression_params = exp.view(batch_size*T_size, -1)
        
        sample["predicted_exp"] = expression_params.view(batch_size,T_size, -1)
        sample["predicted_jaw"] = jaw_pose.view(batch_size,T_size, -1)
        
        assert vector_idx == transformer_out.shape[-1]

        B = transformer_out.shape[0]
        if "gt_shape" not in sample.keys():
            template = sample["template"]
            assert B == 1, "Batch size must be 1 if we want to change the template inside FLAME"
            self.flame.v_template = template.squeeze(1).view(-1, 3)
            # shape_params = torch.zeros((B*T_size, self.flame_config.n_shape), device=transformer_out.device)
            shape_params = torch.zeros((B, T_size, self.flame_config.n_shape), device=transformer_out.device)
        else: 
            # sample["gt_shape"].shape = (B, n_shape) -> add T dimension and repeat
            shape_params = sample["gt_shape"][:, None, ...].repeat(1, T_size, 1)

        flame_jaw_pose = convert_rot(jaw_pose, self.rotation_representation, self.flame_rotation_rep)
        pose_params = torch.cat([ pose_params[..., :3], flame_jaw_pose], dim=-1)

        with torch.no_grad():
            # vertice_neutral, _, _ = self.flame.forward(shape_params[:, 0, ...], torch.zeros_like(expression_params[0:1, ...])) # compute neutral shape
            zero_exp = torch.zeros((B, expression_params.shape[1]), dtype=shape_params.dtype, device=shape_params.device)
            # compute neutral shape for each batch (but not each frame, unnecessary)
            vertices_neutral, _, _ = self.flame.forward(shape_params[:, 0, ...], zero_exp) # compute neutral shape
            vertices_neutral = vertices_neutral.contiguous().view(vertices_neutral.shape[0], -1)[:, None, ...]

        shape_params = shape_params.view(B * T_size, -1)
        vertices_out, _, _ = self.flame(shape_params, expression_params, pose_params)
        vertices_out = vertices_out.contiguous().view(batch_size, T_size, -1)
        vertices_out = vertices_out - vertices_neutral # compute the offset that is then added to the template shape
        
        return vertices_out


# Alignment Bias
def enc_dec_mask(device, T, S, dataset="vocaset"):
    mask = torch.ones(T, S)
    smaller_dim = min(T, S)
    # smaller_dim = T
    if dataset == "BIWI":
        for i in range(smaller_dim):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(smaller_dim):
            mask[i, i] = 0
    else:
        raise NotImplementedError("Unknown dataset")
    return (mask==1).to(device=device)


class FeedForwardDecoder(nn.Module): 

    def __init__(self, cfg) -> None:
        super().__init__()
        self.style_type = cfg.get('style_embedding', 'onehot_linear')
        self.obj_vector = style_from_cfg(cfg)
        self.style_op = cfg.get('style_op', 'add')
        self.PE = positional_encoding_from_cfg(cfg.positional_encoding, cfg.feature_dim )
        self.cfg = cfg

    def forward(self, sample, train=False, teacher_forcing=True): 
        sample["hidden_feature"] = sample["seq_encoder_output"]
        hidden_states = self._positional_enc(sample)

        styled_hidden_states = self._style(sample, hidden_states)

        decoded_offsets = self._decode(sample, styled_hidden_states)

        sample["predicted_vertices"] = decoded_offsets
        sample = self._post_prediction(sample)
        return sample

    def _positional_enc(self, sample): 
        hidden_states = sample["hidden_feature"] 
        if self.PE is not None:
            hidden_states = self.PE(hidden_states)
        return hidden_states

    def _style_dim_factor(self): 
        if self.obj_vector is None: 
            return 1
        elif self.style_op == "cat":
            return 2
        elif self.style_op in ["add", "none", "style_only"]:
            return 1
        raise ValueError(f"Invalid operation: '{self.style_op}'")
       
    def _pe_dim_factor(self):
        dim_factor = 1
        if self.PE is not None: 
            dim_factor = self.PE.output_size_factor()
        return dim_factor

    def _total_dim_factor(self): 
        return self._style_dim_factor() * self._pe_dim_factor()

    def _style(self, sample, hidden_states): 
        if self.obj_vector is None:
            return hidden_states
        if isinstance(self.obj_vector, StyleConditioning):
            # the new way
            style_emb = self.obj_vector(sample)
            # return style_emb
        else:
            # backwards compatibility
            one_hot = sample["one_hot"]
            obj_embedding = self.obj_vector(one_hot)
            style_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
        if self.style_op == "add":
            styled_hidden_states = hidden_states + style_emb
        elif self.style_op == "cat":
            if style_emb.ndim == 2:
                style_emb = style_emb.unsqueeze(1)
            if style_emb.shape[1] == 1:
                style_emb = style_emb.repeat(1, hidden_states.shape[1], 1)
            styled_hidden_states = torch.cat([hidden_states, style_emb], dim=-1)
        elif self.style_op == "none": # no style, for debugging purposes only
            styled_hidden_states = hidden_states
        elif self.style_op == "style_only": # no hidden features, only style. for debugging purposes only
            styled_hidden_states = style_emb
        else: 
            raise ValueError(f"Invalid operation: '{self.style_op}'")
        
        return styled_hidden_states

    def _decode(self, sample, hidden_states) :
        """
        Hiddent states are the ouptut of the encoder
        Returns a tensor of vertex offsets.
        """
        raise NotImplementedError("The subdclass must implement this")

    def _post_prediction(self, sample):
        """
        Adds the template vertices to the predicted offsets
        """
        template = sample["template"]
        vertices_out = sample["predicted_vertices"]
        vertices_out = vertices_out + template[:, None, ...]
        sample["predicted_vertices"] = vertices_out
        return sample

    def get_trainable_parameters(self): 
        return [p for p in self.parameters() if p.requires_grad]


class LinearDecoder(FeedForwardDecoder):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        dim_factor = self._total_dim_factor()
        self.decoder = nn.Linear(dim_factor*cfg.feature_dim, cfg.vertices_dim)
        nn.init.constant_(self.decoder.weight, 0)
        nn.init.constant_(self.decoder.bias, 0)

    def _decode(self, sample, styled_hidden_states) :
        B, T = styled_hidden_states.shape[:2]
        styled_hidden_states = styled_hidden_states.view(B*T, -1)
        decoded_offsets = self.decoder(styled_hidden_states)
        decoded_offsets = decoded_offsets.view(B, T, -1)
        return decoded_offsets


class MLPDecoder(FeedForwardDecoder):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        dim_factor = self._total_dim_factor()
        dim = dim_factor*cfg.feature_dim
        hidden_layer_sizes = cfg.num_layers*[dim]
        self.decoder = MLP(in_size = dim, out_size=cfg.vertices_dim, 
            hidden_layer_sizes=hidden_layer_sizes, hidden_activation=nn.LeakyReLU(), batch_norm=None)
        nn.init.constant_(self.decoder.model[-1].weight, 0)
        nn.init.constant_(self.decoder.model[-1].bias, 0)

    def _decode(self, sample, styled_hidden_states) :
        B, T = styled_hidden_states.shape[:2]
        styled_hidden_states = styled_hidden_states.view(B*T, -1)
        decoded_offsets = self.decoder(styled_hidden_states)
        decoded_offsets = decoded_offsets.view(B, T, -1)
        return decoded_offsets


class BertDecoder(FeedForwardDecoder):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        dim_factor = self._total_dim_factor()
        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.feature_dim * dim_factor, 
                    nhead=cfg.nhead, dim_feedforward=dim_factor*cfg.feature_dim, 
                    activation=cfg.activation,
                    dropout=cfg.dropout, batch_first=True
        )        
        self.bert_decoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.decoder = nn.Linear(dim_factor*cfg.feature_dim, self.decoder_output_dim())
        self.post_bug_fix = cfg.get('post_bug_fix', False)
        self.temporal_bias_type = cfg.get('temporal_bias_type', 'none')
        if self.temporal_bias_type == 'faceformer':
            self.biased_mask = init_faceformer_biased_mask(num_heads = cfg.nhead, max_seq_len = cfg.max_len, period=cfg.period)
        elif self.temporal_bias_type == 'faceformer_future':
            self.biased_mask = init_faceformer_biased_mask_future(num_heads = cfg.nhead, max_seq_len = cfg.max_len, period=cfg.period)
        elif self.temporal_bias_type == 'classic':
            self.biased_mask = init_mask(num_heads = cfg.nhead, max_seq_len = cfg.max_len)
        elif self.temporal_bias_type == 'classic_future':
            self.biased_mask = init_mask_future(num_heads = cfg.nhead, max_seq_len = cfg.max_len)
        elif self.temporal_bias_type == 'none':
            self.biased_mask = None
        else:
            raise ValueError(f"Unsupported temporal bias type '{self.temporal_bias_type}'")

        # trying init to prevent the loss from exploding in the beginning
        nn.init.constant_(self.decoder.weight, 0)
        nn.init.constant_(self.decoder.bias, 0)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.biased_mask is not None:
            self.biased_mask = self.biased_mask.to(*args, **kwargs)
        return self

    def get_shape_model(self):
        return None

    def decoder_output_dim(self):
        return self.cfg.vertices_dim

    def _decode(self, sample, styled_hidden_states):
        if self.biased_mask is not None: 
            mask = self.biased_mask[:, :styled_hidden_states.shape[1], :styled_hidden_states.shape[1]].clone() \
                .detach().to(device=styled_hidden_states.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(styled_hidden_states.shape[0], 1, 1)
        else: 
            mask = None
        decoded_offsets = self.bert_decoder(styled_hidden_states, mask=mask)
        B, T = decoded_offsets.shape[:2]
        decoded_offsets = decoded_offsets.view(B*T, -1)
        ## INSANE BUG WARNING (passing styled_hidden_states instead of decoded_offsets)
        if not self.post_bug_fix:
            decoded_offsets = self.decoder(styled_hidden_states)
        ## END INSANE BUG WARNING
        ## BUG FIX
        else:
            decoded_offsets = self.decoder(decoded_offsets)
        ## END BUG FIX
        decoded_offsets = decoded_offsets.view(B, T, -1)
        return decoded_offsets

class FlameBertDecoder(BertDecoder):

    def __init__(self, cfg) -> None:
        from gdl.models.DecaFLAME import FLAME
        self.flame_config = cfg.flame
        self.pred_dim = 0
        self.predict_exp = cfg.predict_exp
        self.predict_jaw = cfg.predict_jaw
        self.rotation_representation = cfg.get("rotation_representation", "aa")
        self.flame_rotation_rep = "aa"
        if self.predict_exp: 
            self.pred_dim += self.flame_config.n_exp
        if self.predict_jaw:
            self.pred_dim += rotation_rep_size(self.rotation_representation)
        assert self.pred_dim > 0, "No parameters to predict"
        super().__init__(cfg)
        self.flame = FLAME(self.flame_config)
        # trying init to prevent the loss from exploding in the beginning
        nn.init.constant_(self.decoder.weight, 0)
        nn.init.constant_(self.decoder.bias, 0)

        # nn.init.normal_(self.decoder.weight, mean=0.0, std=0.00001)
        # nn.init.normal_(self.decoder.bias, mean=0.0, std=0.00001)

    def get_shape_model(self):
        return self.flame

    def decoder_output_dim(self):
        return self.pred_dim

    def _rotation_representation(self): 
        return self.rotation_representation

    def _decode(self, sample, styled_hidden_states): 
        transformer_out = super()._decode(sample, styled_hidden_states)
        B, T = transformer_out.shape[:2]

        vector_idx = 0
        if self.predict_jaw:
            rotation_rep_size_ = rotation_rep_size(self.rotation_representation)
            jaw_pose = transformer_out[..., :rotation_rep_size_].view(B*T, -1)
            # ensure orthonogonality of the 6D rotation
            if self._rotation_representation() in ["6d", "6dof"]:
                jaw_pose = matrix_to_rotation_6d(rotation_6d_to_matrix(jaw_pose))
            elif self._rotation_representation() in ['mat', 'matrix']:
                jaw_pose = jaw_pose.view(-1, 3, 3) 
                U, S, Vh = torch.linalg.svd(jaw_pose) # do SVD to ensure orthogonality
                jaw_pose = U.contiguous()
            vector_idx += rotation_rep_size_
            
        else: 
            jaw = sample["gt_jaw"][:, :T, ...]
            jaw_pose = jaw.view(B*T, -1)
        if self.predict_exp: 
            expression_params = transformer_out[..., vector_idx:].view(B*T, -1)
            vector_idx += self.flame_config.n_exp
        else: 
            exp = sample["gt_exp"][:, :T, ...]
            expression_params = exp.view(B*T, -1)
        
        assert vector_idx == transformer_out.shape[-1]

        sample["predicted_exp"] = expression_params.view(B,T, -1)
        sample["predicted_jaw"] = jaw_pose.view(B,T, -1)
        if "gt_shape" not in sample.keys():
            template = sample["template"]
            assert B == 1, "Batch size must be 1 if we want to change the template inside FLAME"
            self.flame.v_template = template.squeeze(1).view(-1, 3)
            shape_params = torch.zeros((B,T, self.flame_config.n_shape), device=transformer_out.device)
        else: 
            # sample["gt_shape"].shape = (B, n_shape) -> add T dimension and repeat
            shape_params = sample["gt_shape"][:, None, ...].repeat(1, T, 1)

        flame_jaw_pose = convert_rot(jaw_pose, self.rotation_representation, self.flame_rotation_rep)
        pose_params = self.flame.eye_pose.expand(B*T, -1)
        pose_params = torch.cat([ pose_params[..., :3], flame_jaw_pose], dim=-1)

        # with torch.no_grad():
        #     # vertice_neutral, _, _ = self.flame.forward(shape_params[0:1, ...], torch.zeros_like(expression_params[0:1, ...])) # compute neutral shape
        #                 # vertice_neutral, _, _ = self.flame.forward(shape_params[:, 0, ...], torch.zeros_like(expression_params[0:1, ...])) # compute neutral shape
        #     zero_exp = torch.zeros((B, expression_params.shape[1]), dtype=shape_params.dtype, device=shape_params.device)
        #     # compute neutral shape for each batch (but not each frame, unnecessary)
        #     vertices_neutral, _, _ = self.flame.forward(shape_params[:, 0, ...], zero_exp) # compute neutral shape
        #     vertices_neutral = vertices_neutral.contiguous().view(vertices_neutral.shape[0], -1)[:, None, ...]

        vertices_neutral = self._neutral_shape(B, expression_params.shape[1], shape_params)
        # vertices_neutral = vertices_neutral.expand(B, T, -1, -1)

        shape_params = shape_params.view(B * T, -1)
        vertices_out, _, _ = self.flame(shape_params, expression_params, pose_params)
        vertices_out = vertices_out.contiguous().view(B, T, -1)
        vertex_offsets = vertices_out - vertices_neutral # compute the offset that is then added to the template shape
        vertex_offsets = vertex_offsets.view(B, T, -1)
        return vertex_offsets


    def _neutral_shape(self, B, exp_dim, shape_params): 
        with torch.no_grad():
            # vertice_neutral, _, _ = self.flame.forward(shape_params[0:1, ...], torch.zeros_like(expression_params[0:1, ...])) # compute neutral shape
                        # vertice_neutral, _, _ = self.flame.forward(shape_params[:, 0, ...], torch.zeros_like(expression_params[0:1, ...])) # compute neutral shape
            zero_exp = torch.zeros((B, exp_dim), dtype=shape_params.dtype, device=shape_params.device)
            # compute neutral shape for each batch (but not each frame, unnecessary)
            vertices_neutral, _, _ = self.flame.forward(shape_params[:, 0, ...], zero_exp) # compute neutral shape
            vertices_neutral = vertices_neutral.contiguous().view(vertices_neutral.shape[0], -1)[:, None, ...]
        return vertices_neutral



def load_motion_prior_net(path, trainable=False):

    with open(path + "/cfg.yaml", 'r') as f:
        model_config = OmegaConf.load(f)
    checkpoint_mode = 'best' # resuming in the same stage, we want to pick up where we left of
    
    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(
        model_config, "", 
        checkpoint_mode=checkpoint_mode,
        pattern="val"
        )


    motion_prior_net_class = class_from_str(model_config.model.pl_module_class, sys.modules[__name__])
    motion_prior_net = motion_prior_net_class.instantiate(model_config, "", "", checkpoint, checkpoint_kwargs)
    if not trainable:
        motion_prior_net.eval()
        # freeze model
        for p in motion_prior_net.parameters():
            p.requires_grad = False
    return motion_prior_net


class ConvSquasher(nn.Module): 

    def __init__(self, input_dim, quant_factor, output_dim) -> None:
        super().__init__()
        self.squasher = create_squasher(input_dim, output_dim, quant_factor)

    def forward(self, x):
        # BTF -> BFT 
        x = x.transpose(1, 2)
        x = self.squasher(x)
        # BFT -> BTF
        x = x.transpose(1, 2)
        return x

class StackLinearSquash(nn.Module): 
    def __init__(self, input_dim, latent_frame_size, output_dim): 
        super().__init__()
        self.input_dim = input_dim
        self.latent_frame_size = latent_frame_size
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim * latent_frame_size, output_dim)
        
    def forward(self, x):
        B, T, F = x.shape
        # input B, T, F -> B, T // latent_frame_size, F * latent_frame_size
        assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
        T_latent = T // self.latent_frame_size
        F_stack = F * self.latent_frame_size
        x = x.reshape(B, T_latent, F_stack)
        x = x.view(B * T_latent, F_stack)
        x = self.linear(x)
        x = x.view(B, T_latent, -1)
        return x

class BertPriorDecoder(FeedForwardDecoder):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        dim_factor = self._total_dim_factor()

        if cfg.num_layers > 0:
            encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=cfg.feature_dim * dim_factor, 
                        nhead=cfg.nhead, dim_feedforward=dim_factor*cfg.feature_dim, 
                        activation=cfg.activation,
                        dropout=cfg.dropout, batch_first=True
            )        
            self.bert_decoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        else:
            self.bert_decoder = None
        
        self.post_bug_fix = cfg.get('post_bug_fix', False)

        self.temporal_bias_type = cfg.get('temporal_bias_type', 'none')
        # max_len = cfg.max_len
        max_len = 1200
        if self.temporal_bias_type == 'faceformer':
            self.biased_mask = init_faceformer_biased_mask(num_heads = cfg.nhead, max_seq_len = max_len, period=cfg.period)
        elif self.temporal_bias_type == 'faceformer_future':
            self.biased_mask = init_faceformer_biased_mask_future(num_heads = cfg.nhead, max_seq_len = max_len, period=cfg.period)
        elif self.temporal_bias_type == 'classic':
            self.biased_mask = init_mask(num_heads = cfg.nhead, max_seq_len = max_len)
        elif self.temporal_bias_type == 'classic_future':
            self.biased_mask = init_mask_future(num_heads = cfg.nhead, max_seq_len = max_len)
        elif self.temporal_bias_type == 'none':
            self.biased_mask = None
        else:
            raise ValueError(f"Unsupported temporal bias type '{self.temporal_bias_type}'")

        self.motion_prior : MotionPrior = load_motion_prior_net(cfg.motion_prior.path, cfg.motion_prior.get('trainable', False))
        self.latent_frame_size = self.motion_prior.latent_frame_size()
        quant_factor = self.motion_prior.quant_factor()
        bottleneck_dim = self.motion_prior.bottleneck_dim()
        self.motion_prior.discard_encoder()
        self.flame = self.motion_prior.get_flame()

        # temporal downsampling (if need be) to match the motion prior network latent frame rate
        if cfg.get('squash_before', True):
            self.squasher = self._create_squasher(cfg.get('squash_type', 'conv'), cfg.feature_dim * dim_factor, cfg.feature_dim * dim_factor, quant_factor)
        else:
            self.squasher = None

        self.decoder = nn.Linear(dim_factor*cfg.feature_dim, bottleneck_dim)

        # trying init to prevent the loss from exploding in the beginning
        nn.init.constant_(self.decoder.weight, 0)
        nn.init.constant_(self.decoder.bias, 0)

        # self.bottleneck_proj = nn.Linear(cfg.feature_dim * dim_factor, bottleneck_dim)

        if cfg.get('squash_after', False):
            self.squasher_2 = self._create_squasher(cfg.get('squash_type', 'conv'), bottleneck_dim, bottleneck_dim, quant_factor)
        else:
            self.squasher_2 = None

        assert not (self.squasher is not None and self.squasher_2 is not None), "Cannot have two squashers"

    def _create_squasher(self, type, input_dim, output_dim, quant_factor): 
        if type == "conv": 
            return ConvSquasher(input_dim, quant_factor, output_dim)
        elif type == "stack_linear": 
            return StackLinearSquash(input_dim, self.latent_frame_size, output_dim)
        else: 
            raise ValueError("Unknown squasher type")

    def train(self, mode: bool = True) -> "BertPriorDecoder":
        super().train(mode)
        if self.cfg.motion_prior.get('trainable', False):
            self.motion_prior.train(mode)
        else: 
            self.motion_prior.eval()
        return self

    def to(self, device):
        super().to(device)
        if self.biased_mask is not None:
            self.biased_mask = self.biased_mask.to(device)
        if self.bert_decoder is not None:
            self.bert_decoder.to(device)
        self.motion_prior.to(device)
        if self.squasher is not None:
            self.squasher.to(device)
        if self.squasher_2 is not None:
            self.squasher_2.to(device)
        self.decoder.to(device)
        # self.bottleneck_proj.to(device
        return self

    def _rotation_representation(self):
        return self.motion_prior._rotation_representation()

    def forward(self, sample, train=False, teacher_forcing=True): 
        sample = super().forward(sample, train=train, teacher_forcing=teacher_forcing)
        return sample

    def get_shape_model(self):
        return self.motion_prior.get_flame() 

    def decoder_output_dim(self):
        return self.cfg.vertices_dim

    def _post_prediction(self, sample):
        """
        Overrides the base class method to apply the motion prior network. 
        """
        sample = self._apply_motion_prior(sample)
        super()._post_prediction(sample)
        return sample

    def _apply_motion_prior(self, sample):
        decoded_offsets = sample["predicted_vertices"]
        B,T = decoded_offsets.shape[:2]
        batch = {}
        batch[self.motion_prior.input_key_for_decoding_step()] = decoded_offsets 
        T_real = sample["gt_vertices"].shape[1] if "gt_vertices" in sample.keys() else sample["processed_audio"].shape[1]

        T_motion_prior = (T_real // self.latent_frame_size) * self.latent_frame_size
        if T_motion_prior < T_real:
            T_padded = int(math.ceil(T_real / self.latent_frame_size) * self.latent_frame_size)

            # pad to the nearest multiple of the motion prior latent frame size along the temporal dimension T
            # B, T, C -> B, T_padded, C 
            if self.squasher is not None:
                # we're already working on the latent framize size, the data has already been squashed
                padding_size = T_padded // self.latent_frame_size - T_real // self.latent_frame_size
            elif self.squasher_2 is not None:
                # we're working on the original frame size, the data has not been squashed yet
                padding_size = T_padded - T_real 
            batch[self.motion_prior.input_key_for_decoding_step()] = torch.nn.functional.pad(
                decoded_offsets, (0,0, 0, padding_size), mode='constant', value=0
            )

            if self.squasher is not None:
                assert batch[self.motion_prior.input_key_for_decoding_step()].shape[1] == T_padded // self.latent_frame_size, \
                    f"{batch[self.motion_prior.input_key_for_decoding_step()].shape[1]} != {T_padded // self.latent_frame_size}"
            elif self.squasher_2 is not None:
                assert batch[self.motion_prior.input_key_for_decoding_step()].shape[1] == T_padded, \
                    f"{batch[self.motion_prior.input_key_for_decoding_step()].shape[1]} != {T_padded}"

        if self.squasher_2 is not None:
            batch[self.motion_prior.input_key_for_decoding_step()] = self.squasher_2(batch[self.motion_prior.input_key_for_decoding_step()])

        if "gt_shape" in sample:
            batch["gt_shape"] = sample["gt_shape"]
        if "template" in sample:
            batch["template"] = sample["template"]
        if "gt_tex" in sample:
            batch["gt_tex"] = sample["gt_tex"]
        
        sample["prior_input_sequence"] = batch[self.motion_prior.input_key_for_decoding_step()]

        batch = self.motion_prior.decoding_step(batch)

        # if T_real < T:
        # remove the padding
        for i, key in enumerate(self.motion_prior.cfg.model.sequence_components.keys()):
            batch["reconstructed_" + key] = batch["reconstructed_" + key][:,:T_real]
        batch["reconstructed_vertices"] = batch["reconstructed_vertices"][:,:T_real]
        
        for i, key in enumerate(self.motion_prior.cfg.model.sequence_components.keys()):
            sample["predicted_" + key] = batch["reconstructed_" + key]
        sample["predicted_vertices"] = batch["reconstructed_vertices"]
        

        # compute the offsets from neutral, that's how the output of "predicted_vertices" is expected to be
        if "gt_shape" not in sample.keys():
            template = sample["template"]
            assert B == 1, "Batch size must be 1 if we want to change the template inside FLAME"
            flame_template = self.flame.v_template
            self.flame.v_template = template.squeeze(1).view(-1, 3)
            shape_params = torch.zeros((B,T_real, self.flame.cfg.n_shape), device=sample["predicted_vertices"].device)
        else: 
            flame_template = None
            # sample["gt_shape"].shape = (B, n_shape) -> add T dimension and repeat
            if len(sample["gt_shape"].shape) == 2:
                shape_params = sample["gt_shape"][:, None, ...].repeat(1, T_real, 1)
            elif len(sample["gt_shape"].shape) == 3:
                shape_params = sample["gt_shape"]

        B = sample["predicted_vertices"].shape[0]
        vertices_neutral = self._neutral_shape(B, sample["predicted_exp"].shape[-1], shape_params=shape_params)
        # vertices_neutral = vertices_neutral.expand(B, T_real, -1, -1)
        vertex_offsets = sample["predicted_vertices"] - vertices_neutral # compute the offset that is then added to the template shape
        sample["predicted_vertices"] = vertex_offsets

        if flame_template is not None:  # correct the flame template back if need be
            self.flame.v_template = flame_template
        return sample

    def _neutral_shape(self, B, exp_dim, shape_params): 
        with torch.no_grad():
            # vertice_neutral, _, _ = self.flame.forward(shape_params[0:1, ...], torch.zeros_like(expression_params[0:1, ...])) # compute neutral shape
                        # vertice_neutral, _, _ = self.flame.forward(shape_params[:, 0, ...], torch.zeros_like(expression_params[0:1, ...])) # compute neutral shape
            zero_exp = torch.zeros((B, exp_dim), dtype=shape_params.dtype, device=shape_params.device)
            # compute neutral shape for each batch (but not each frame, unnecessary)
            vertices_neutral, _, _ = self.flame.forward(shape_params[:, 0, ...], zero_exp) # compute neutral shape
            vertices_neutral = vertices_neutral.contiguous().view(vertices_neutral.shape[0], -1)[:, None, ...]
        return vertices_neutral

    def _decode(self, sample, styled_hidden_states):
        B, T, F = styled_hidden_states.shape
        # # BTF to BFT (for 1d conv)
        # if self.cfg.get('squash_before', True):
        if self.squasher is not None:
            styled_hidden_states = self.squasher(styled_hidden_states)

        if self.bert_decoder is not None:
            if self.biased_mask is not None: 
                mask = self.biased_mask[:, :styled_hidden_states.shape[1], :styled_hidden_states.shape[1]].clone() \
                    .detach().to(device=styled_hidden_states.device)
                if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                    mask = mask.repeat(styled_hidden_states.shape[0], 1, 1)
            else: 
                mask = None
            decoded_offsets = self.bert_decoder(styled_hidden_states, mask=mask)
        else: 
            decoded_offsets = styled_hidden_states

        B, T = decoded_offsets.shape[:2]
        decoded_offsets = decoded_offsets.view(B*T, -1)
        ## INSANE BUG WARNING (passing in styled_hidden_states instead of decoded_offsets)
        if not self.post_bug_fix:
            decoded_offsets = self.decoder(styled_hidden_states)
        ## END INSANE BUG WARNING
        ## BUG FIX
        else:
            decoded_offsets = self.decoder(decoded_offsets)
        ## END BUG FIX
        decoded_offsets = decoded_offsets.view(B, T, -1) 
        return decoded_offsets


class LinearAutoRegDecoder(AutoRegressiveDecoder):

    def __init__(self, cfg):
        super().__init__()
        # self.flame_config = cfg.flame
        self.decoder = nn.Linear(2*cfg.feature_dim, cfg.vertices_dim)
        nn.init.constant_(self.decoder.weight, 0)
        nn.init.constant_(self.decoder.bias, 0)
        self.obj_vector = nn.Linear(cfg.num_training_subjects, cfg.feature_dim, bias=False)
        self.vertice_map = nn.Linear(cfg.vertices_dim, cfg.feature_dim)
        nn.init.constant_(self.vertice_map.weight, 0)
        nn.init.constant_(self.vertice_map.bias, 0)
        self.PE = None

    def get_trainable_parameters(self): 
        return [p for p in self.parameters() if p.requires_grad]

    def _max_auto_regressive_steps(self):
        return 2000

    def _autoregressive_step(self, sample, i):
        hidden_states = sample["hidden_feature"]
        if i==0:
            one_hot = sample["one_hot"]
            obj_embedding = self.obj_vector(one_hot)
            vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = vertice_emb
            sample["style_emb"] = style_emb
            # vertices_input = self.PE(style_emb)
            vertices_input = style_emb
        else:
            vertice_emb = sample["embedded_output"]
            style_emb = sample["style_emb"]
            vertices_input = vertice_emb
        if self.PE is not None:
            vertices_input = self.PE(vertices_input)
        vertices_out = self._decode(sample, vertices_input, hidden_states)
        sample["predicted_vertices"] = vertices_out

        new_output = self.vertice_map(vertices_out[:,-1,:]).unsqueeze(1)
        new_output = new_output + style_emb
        vertice_emb = torch.cat((vertice_emb, new_output), 1)
        sample["embedded_output"] = vertice_emb
        return sample

    def _teacher_forced_step(self, sample): 
        vertices = sample["gt_vertices"]
        template = sample["template"].unsqueeze(1)
        hidden_states = sample["hidden_feature"]
        one_hot = sample["one_hot"]
        obj_embedding = self.obj_vector(one_hot)

        vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
        style_emb = vertice_emb  

        vertices_input = torch.cat((template, vertices[:,:-1]), 1) # shift one position
        vertices_input = vertices_input - template
        vertices_input = self.vertice_map(vertices_input) # map to feature dim
        vertices_input = vertices_input + style_emb # add style embedding
        if self.PE is not None:
            vertices_input = self.PE(vertices_input) # add positional encoding

        vertices_out = self._decode(sample, vertices_input, hidden_states)
        sample["predicted_vertices"] = vertices_out

        return sample

    def _decode(self, sample, vertices_input, hidden_states):
        # dev = vertices_input.device
        # tgt_mask = self.biased_mask[:, :vertices_input.shape[1], :vertices_input.shape[1]].clone().detach().to(device=dev)
        # memory_mask = enc_dec_mask(dev, vertices_input.shape[1], hidden_states.shape[1])
        # transformer_out = self.transformer_decoder(vertices_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        #TODO: we can also consider feeding the hidden states of a large perceptive field (further back in the past/future) to the decoder
        decoder_input = torch.cat([vertices_input, hidden_states[:,:vertices_input.shape[1], ...]], dim=-1)
        B, T = decoder_input.shape[:2]
        vertices_out = self.decoder(decoder_input)
        vertices_out = vertices_out.view(B, T, -1)
        return vertices_out

    def _post_prediction(self, sample):
        """
        Adds the template vertices to the predicted offsets
        """
        template = sample["template"]
        vertices_out = sample["predicted_vertices"]
        vertices_out = vertices_out + template
        sample["predicted_vertices"] = vertices_out
        return sample

    def _decode_vertices(self, sample, vertices_input):
        return self.decoder(vertices_input)


def get_rec_dict(batch, rec_type): 
    rec_dict = batch if rec_type is None else batch["reconstruction"][rec_type]