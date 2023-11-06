from torch import batch_norm
from gdl.models.DECA import * 
from gdl.models.temporal.MultiModalTemporalNet import *
from gdl.models.temporal.Bases import *
from gdl.models.temporal.BlockFactory import norm_from_cfg
from gdl.models.MLP import MLP 
import random

from gdl.models.temporal.VideoEncoders import EmocaVideoEncoder 


class TemporalMLP(MLP): 

    def __init__(self, in_dim, out_dim, hidden_layer_sizes, batch_norm=None):
        super().__init__(in_dim, out_dim, hidden_layer_sizes, batch_norm=None)
        self._reset_parameters()

    def _reset_parameters(self): 
        super()._reset_parameters()
        # torch.nn.init.constant_(self.model[-1].weight, 0)
        # torch.nn.init.constant_(self.model[-1].bias, 0)

    def forward(self, x):
        B = x.shape[0]
        T = x.shape[1]
        x = x.reshape(B*T, -1)
        y = super().forward(x) 
        y = y.view(B, T, -1)
        return y


def code_vector_projection_from_cfg(cfg, out_sequence_dim, code_vector_dim, code_size_dict=None, proj_cfg=None):
    projection = proj_cfg or cfg.model.get('code_vector_projection')
    if projection is None or projection.type in ['linear']:
        layer = torch.nn.Linear( out_sequence_dim, code_vector_dim) 
        # torch.nn.init.constant_(layer.weight, 0) # if we initialize to zero, the convergence should be much better
        # torch.nn.init.constant_(layer.bias, 0)
        return layer

    elif projection.type == 'mlp':
        if projection.hidden_layer_sizes is None:
            hidden_layer_sizes = [out_sequence_dim, out_sequence_dim]
        else: 
            hidden_layer_sizes = projection.hidden_layer_sizes
        if projection.norm_layer is None: 
            norm_layer = None
        elif projection.norm_layer == 'batchnorm1d':
            norm_layer = torch.nn.BatchNorm1d 
        elif projection.norm_layer == 'layernorm':
            norm_layer =  torch.nn.LayerNorm
        else: 
            raise ValueError(f"Unknown batch norm type '{projection.norm_layer}'")
        module = TemporalMLP(out_sequence_dim, code_vector_dim, hidden_layer_sizes, batch_norm=norm_layer)
        return module
    elif projection.type == 'parallel':
        return ParallelProjection(cfg, out_sequence_dim, code_vector_dim, code_size_dict)
    raise NotImplementedError(f'Unknown code_vector_projection option: "{projection}"')


class ParallelProjection(torch.nn.Module): 
    
    def __init__(self, cfg, out_sequence_dim, code_vector_dim, code_size_dict):
        super().__init__()
        self.cfg = cfg
        projectors = []
        # self.outputs = []
        # self.input_feature_dim = None
        assert len(cfg.model.code_vector_projection.projectors) > 1, "At least two projectors are required"
        total_dim = 0
        self.names = []
        for proj_cfg in cfg.model.code_vector_projection.projectors:          
            name = list(proj_cfg.keys())[0]
            dim = 0 
            if self.seq_predicts_shape(proj_cfg[name]):
                dim += code_size_dict["shapecode"] 
            if self.seq_predicts_tex(proj_cfg[name]):
                dim += code_size_dict["texcode"]
            if self.seq_predicts_exp(proj_cfg[name]):
                dim += code_size_dict["expcode"]
            if self.seq_predicts_jawpose(proj_cfg[name]):
                dim += code_size_dict["jawpose"]
            if self.seq_predicts_globalpose(proj_cfg[name]):
                dim += code_size_dict["globalpose"]
            if self.seq_predicts_cam(proj_cfg[name]):
                dim += code_size_dict["cam"]
            if self.seq_predicts_light(proj_cfg[name]):
                dim += code_size_dict["lightcode"]
                
            projector = code_vector_projection_from_cfg(cfg, out_sequence_dim, dim, None, proj_cfg[name].cfg)    
            # if self.input_feature_dim is None:
            #     self.input_feature_dim = projector.input_feature_dim()
            # assert self.input_feature_dim == projector.input_feature_dim(), \
            #     "All projectors must have the same input feature dim"
            projectors += [projector]
            # self.outputs += [proj_cfg[name].cfg.output]
            total_dim += dim
            self.names += [name]
        assert total_dim == code_vector_dim, \
            f"Total code vector dim {total_dim} does not match code vector dim {code_vector_dim}"
        self.projectors = torch.nn.ModuleList(projectors)

    def forward(self, x):
        assert isinstance(x, dict)
        assert len(x) == len(self.projectors)
        results = []
        for pi, projector in enumerate(self.projectors):
            results += [projector(x[self.names[pi]])]
        res = torch.cat(results, dim=-1)
        return res

    def seq_encoded_code_size_dict(self):
        if self.seq_predicts_shape(): 
            self._seq_encoded_code_sizes["shapecode"] = self.cfg.model.sizes.n_shape
        if self.seq_predicts_tex():
            self._seq_encoded_code_sizes["texcode"] = self.cfg.model.sizes.n_tex
        if self.seq_predicts_exp():
            self._seq_encoded_code_sizes["expcode"] = self.cfg.model.sizes.n_exp
        if self.seq_predicts_globalpose():
            self._seq_encoded_code_sizes["globalpose"] = self.cfg.model.sizes.n_pose // 2
        if self.seq_predicts_jawpose():
            self._seq_encoded_code_sizes["jawpose"] = self.cfg.model.sizes.n_pose // 2
        if self.seq_predicts_cam():
            self._seq_encoded_code_sizes["cam"] = self.cfg.model.sizes.n_cam
        if self.seq_predicts_light():
            self._seq_encoded_code_sizes["lightcode"] = self.cfg.model.sizes.n_light
        return self._seq_encoded_code_sizes

    def seq_predicts_shape(self, proj_cfg):
        return proj_cfg.output.predict_shapecode

    def seq_predicts_tex(self, proj_cfg):
        return proj_cfg.output.predict_texcode

    def seq_predicts_exp(self, proj_cfg):
        return proj_cfg.output.predict_expcode
    
    # def seq_predicts_pose(self):
    #     raise NotImplementedError()

    def seq_predicts_jawpose(self, proj_cfg):
        return proj_cfg.output.predict_jawpose

    def seq_predicts_globalpose(self, proj_cfg):
        return proj_cfg.output.predict_globalpose

    def seq_predicts_cam(self, proj_cfg):
        return proj_cfg.output.predict_cam

    def seq_predicts_light(self, proj_cfg):
        return proj_cfg.output.predict_light
        
    # def seq_encoded_code_vector_dim(self, code_size_dict): 
    #     dim = 0 
    #     for key, value in self.code_size_dict.items(): 
    #         dim += value
    #     return dim
        

class TemporalFace(MultiModalTemporalNet): 

    def __init__(self, 
                cfg,
                modality_flag: FaceModalities, 
                video_model: Optional[TemporalVideoEncoder] = None,
                audio_model: Optional[TemporalAudioEncoder] = None, 
                sequence_encoder : SequenceEncoder = None,
                sequence_decoder : Optional[SequenceDecoder] = None,
                shape_model: Optional[ShapeModel] = None,
                renderer: Optional[Renderer] = None,
                # post_fusion_norm: Optional = None,
                *args: Any, **kwargs: Any) -> None:
        self.cfg = cfg
        # modality_flag = FaceModalities.AUDIO_VISUAL if modality_flag is None else modality_flag
        super().__init__(modality_flag, *args, **kwargs)  
        self.video_model = video_model
        self.audio_model = audio_model
        self.sequence_encoder = sequence_encoder
        self.sequence_decoder = sequence_decoder
        self.shape_model = shape_model
        self.renderer = renderer

        if self.sequence_decoder is None: 
            self.code_vec_input_name = "seq_encoder_output"
        else:
            self.code_vec_input_name = "seq_decoder_output"

        if self.uses_audio(): 
            assert self.audio_model is not None, "audio model is not specified"

        if self.uses_video():
            assert self.video_model is not None, "face reconstruction model is not specified"

        if self.uses_text(): 
            raise NotImplementedError() 

        self.fusion_type = "cat"
        
        # self.drop_video_features_prob = cfg.model.get('drop_video_features_prob', None)
        self.drop_video_features_prob = cfg.model.get('drop_video_features_prob', None)
        self.drop_audio_features_prob = cfg.model.get('drop_audio_features_prob', None)

        self.forward_masked_video = cfg.model.get('forward_masked_video', False)

        if self.fused_feature_dim() != self.sequence_encoder.input_feature_dim(): 
            self.post_fusion_projection = torch.nn.Linear(self.fused_feature_dim(), 
                                                          self.sequence_encoder.input_feature_dim())
        else: 
            self.post_fusion_projection = None
        self.post_fusion_norm = norm_from_cfg(cfg.model.get('post_fusion_norm', None), 
                                              [-1, -1, self.sequence_encoder.input_feature_dim()] # [batch_size, seq_len, feature_dim]
        )
        # projects the results of the decoder into the code space (shape model, rendering, etc.)
        self.code_vector_projector = code_vector_projection_from_cfg(cfg, self.out_sequence_dim(), self.seq_encodeded_code_vector_dim(), self.seq_encoded_code_size_dict())

    def _get_trainable_parameters(self):
        return []

    def get_trainable_parameters(self):
        trainable_params = []
        trainable_params += self._get_trainable_parameters()

        if self.code_vector_projector is not None:
            trainable_params += list(self.code_vector_projector.parameters())
        
        if self.post_fusion_projection is not None:
            trainable_params += list(self.post_fusion_projection.parameters())
        if self.post_fusion_norm is not None:
            trainable_params += list(self.post_fusion_norm.parameters())

        if self.audio_model is not None:
            trainable_params += self.audio_model.get_trainable_parameters()
        if self.video_model is not None:
            trainable_params += self.video_model.get_trainable_parameters()
        trainable_params += self.sequence_encoder.get_trainable_parameters()
        if self.sequence_decoder is not None:
            trainable_params += self.sequence_decoder.get_trainable_parameters()
        if self.shape_model is not None:
            trainable_params += self.shape_model.get_trainable_parameters()
        if self.renderer is not None:
            trainable_params += self.renderer.get_trainable_parameters()

        return trainable_params

    def fused_feature_dim(self): 
        fused_feature_dim = 0
        if self.uses_audio(): 
            fused_feature_dim += self.audio_model.output_feature_dim()
        if self.uses_video():
            fused_feature_dim += self.video_model.output_feature_dim()
        if self.uses_text(): 
            fused_feature_dim += self.text_model.output_feature_dim()
        return fused_feature_dim

    def out_sequence_dim(self): 
        if self.sequence_decoder is not None: 
            return self.sequence_decoder.output_feature_dim()
        return self.sequence_encoder.output_feature_dim()

    def forward_video(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        sample = self.video_model(sample, train=train, **kwargs)
        if train and sample['visual_feature'].numel() > 0 and self.drop_video_features_prob is not None and \
            self.drop_video_features_prob > 0 and self.drop_video_features_prob <= 1.0:
            drop = random.random() 
            if drop < self.drop_video_features_prob:
                mask = sample["masked_frames"]
                sample['visual_feature'] = sample['visual_feature'] * mask[..., None]
        return sample

    def forward_audio(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        sample = self.audio_model(sample, train=train, **kwargs)     
        if train and  sample['audio_feature'].numel() > 0 and self.drop_audio_features_prob is not None and \
            self.drop_audio_features_prob > 0 and self.drop_audio_features_prob <= 1.0:
            drop = random.random() 
            if drop < self.drop_audio_features_prob:
                mask = sample["masked_frames"]
                sample['audio_feature'] = sample['audio_feature'] * mask[..., None]
        return self.audio_model(sample, train=train, **kwargs)
    
    def signal_fusion(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        if self.uses_audio() and self.uses_video():
            video_feat = sample["visual_feature"] # b, t, fv
            audio_feat = sample["audio_feature"] # b, t, fa 

            if self.fusion_type in ["concat", "cat", "concatenate"]:
                fused_feature = torch.cat([video_feat, audio_feat], dim=2) # b, t, fv + fa
            elif self.fusion_type in ["add", "sum"]:
                fused_feature = video_feat + audio_feat 
            elif self.fusion_type in ["max"]:
                fused_feature = torch.stack(video_feat + audio_feat, dim=0).max(dim=0)
            else: 
                raise ValueError(f"Unknown fusion type {self.fusion_type}")
            sample["fused_feature"] = fused_feature
        elif self.uses_audio():
            sample["fused_feature"] = sample["audio_feature"]
        elif self.uses_video():
            sample["fused_feature"] = sample["visual_feature"]
        else:
            raise ValueError("Does not use audio or video. Nothing to fuse")

        if self.post_fusion_projection is not None: 
            sample["fused_feature"] = self.post_fusion_projection(sample["fused_feature"]) 
        if self.post_fusion_norm is not None: 
            sample["fused_feature"] = self.post_fusion_norm(sample["fused_feature"])
        return sample

    def encode_sequence(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        return self.sequence_encoder(sample, **kwargs)

    def decode_sequence(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        if self.sequence_decoder is not None: 
            return self.sequence_decoder(sample, **kwargs)
        sample["seq_decoder_output"] = sample["seq_encoder_output"]
        return sample

    def code_size_dict(self): 
        if not hasattr(self, '_code_sizes') or self._code_sizes is None:
            self._code_sizes = {
                "shapecode":self.cfg.model.sizes.n_shape, 
                "texcode":  self.cfg.model.sizes.n_tex, 
                "expcode":  self.cfg.model.sizes.n_exp, 
                # "posecode": self.cfg.model.sizes.n_pose, 
                "jawpose": self.cfg.model.sizes.n_pose // 2, 
                "globalpose": self.cfg.model.sizes.n_pose // 2, 
                "cam":      self.cfg.model.sizes.n_cam, 
                "lightcode":self.cfg.model.sizes.n_light
            }
        return self._code_sizes
    
    def total_code_vector_dim(self): 
        dim = 0 
        for key, value in self.code_size_dict().items(): 
            dim += value
        return dim

    def seq_predicts_shape(self):
        raise NotImplementedError()

    def seq_predicts_tex(self):
        raise NotImplementedError()

    def seq_predicts_exp(self):
        raise NotImplementedError()
    
    # def seq_predicts_pose(self):
    #     raise NotImplementedError()

    def seq_predicts_jawpose(self):
        raise NotImplementedError()

    def seq_predicts_globalpose(self):
        raise NotImplementedError()

    def seq_predicts_cam(self):
        raise NotImplementedError()

    def seq_predicts_light(self):
        raise NotImplementedError()


    def seq_encoded_code_size_dict(self):
        if not hasattr(self, '_seq_encoded_code_size') or self._seq_encoded_code_size is None:
            self._seq_encoded_code_sizes = {}      
        if self.seq_predicts_shape(): 
            self._seq_encoded_code_sizes["shapecode"] = self.cfg.model.sizes.n_shape
        if self.seq_predicts_tex():
            self._seq_encoded_code_sizes["texcode"] = self.cfg.model.sizes.n_tex
        if self.seq_predicts_exp():
            self._seq_encoded_code_sizes["expcode"] = self.cfg.model.sizes.n_exp
        if self.seq_predicts_globalpose():
            self._seq_encoded_code_sizes["globalpose"] = self.cfg.model.sizes.n_pose // 2
        if self.seq_predicts_jawpose():
            self._seq_encoded_code_sizes["jawpose"] = self.cfg.model.sizes.n_pose // 2
        if self.seq_predicts_cam():
            self._seq_encoded_code_sizes["cam"] = self.cfg.model.sizes.n_cam
        if self.seq_predicts_light():
            self._seq_encoded_code_sizes["lightcode"] = self.cfg.model.sizes.n_light
        return self._seq_encoded_code_sizes

    def seq_encodeded_code_vector_dim(self): 
        dim = 0 
        for key, value in self.seq_encoded_code_size_dict().items(): 
            dim += value
        return dim

    def code_vector_projection(self, sample: Dict, train=False, **kwargs: Any) -> Dict: 
        if self.code_vector_projector is not None:
            sample["code_vector"] = self.code_vector_projector(sample[self.code_vec_input_name])
        else: 
            sample["code_vector"] = sample[self.code_vec_input_name]
        return sample

    def decompose_code(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        code = sample["code_vector"]
        start = 0
        # for name, value in self.code_size_dict().items():
        for name, value in self.seq_encoded_code_size_dict().items():
            sample[name] = (code[..., start:start + value])
            start = start + value
        assert start == code.shape[-1] # sanity check for size of code vector 
        
        # # todo: this should probably be moved over to the renderer class
        # sample["lightcode"] = sample["lightcode"].reshape(code.shape[0], code.shape[1], 9, 3)
        ## todo: done
        return sample

    def decode_shape(self, sample: Dict, train=False, decode_emoca=True, **kwargs: Any) -> None:
        sample = self.shape_model(sample)
        decode_emoca = isinstance(self.video_model, EmocaVideoEncoder) and decode_emoca # hack
        if decode_emoca:
            emoca_sample = {}
            emoca_sample["shapecode"] = sample["emoca_shapecode"]
            emoca_sample["expcode"] = sample["emoca_expcode"]
            emoca_sample["texcode"] = sample["emoca_texcode"]
            emoca_sample["lightcode"] = sample["emoca_lightcode"] 
            emoca_sample["cam"] = sample["emoca_cam"]
            emoca_sample["jawpose"] = sample["emoca_jawpose"]
            emoca_sample["globalpose"] = sample["emoca_globalpose"]
            with torch.no_grad():
                emoca_sample = self.shape_model(emoca_sample)
            for key, value in emoca_sample.items():
                sample["emoca_" + key] = value

            if self.forward_masked_video: 
                unmasked_sample = {}
                unmasked_sample["shapecode"] = sample["unmasked_shapecode"]
                unmasked_sample["expcode"] = sample["unmasked_expcode"]
                unmasked_sample["texcode"] = sample["unmasked_texcode"]
                unmasked_sample["lightcode"] = sample["unmasked_lightcode"] 
                unmasked_sample["cam"] = sample["unmasked_cam"]
                unmasked_sample["jawpose"] = sample["unmasked_jawpose"]
                unmasked_sample["globalpose"] = sample["unmasked_globalpose"]
                with torch.no_grad():
                    unmasked_sample = self.shape_model(unmasked_sample)
                for key, value in unmasked_sample.items():
                    sample["unmasked_" + key] = value
        return sample

    def rendering_pass(self, sample: Dict, train=False, decode_emoca=True, **kwargs: Any) -> Dict:
        sample = self.renderer(sample)
        decode_emoca = isinstance(self.video_model, EmocaVideoEncoder) and decode_emoca # hack
        if decode_emoca:
            prefix = "emoca_"
            emoca_sample = {k[len(prefix):] : v for k,v in sample.items() if k.startswith(prefix)}
            with torch.no_grad():
                emoca_sample = self.renderer(emoca_sample)
            for key, value in emoca_sample.items():
                sample["emoca_" + key] = value

            if self.forward_masked_video:
                prefix = "unmasked_"
                unmasked_sample = {k[len(prefix):] : v for k,v in sample.items() if k.startswith(prefix)}
                with torch.no_grad():
                    unmasked_sample = self.renderer(unmasked_sample)
                for key, value in unmasked_sample.items():
                    sample["unmasked_" + key] = value


        return sample

    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'TemporalFace':
        raise NotImplementedError("Subclasses of TemporalFace must implement instantiate()")




def instantiate(): 
    pass 