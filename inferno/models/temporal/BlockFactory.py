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

from pathlib import Path
from inferno.models.temporal.Bases import Preprocessor
from inferno.models.temporal.Preprocessors import FlamePreprocessor, EmotionRecognitionPreprocessor, EmocaPreprocessor, SpeechEmotionRecognitionPreprocessor
from inferno.models.temporal.external.SpectrePreprocessor import SpectrePreprocessor
from inferno.models.temporal.SequenceEncoders import *
# from inferno.models.temporal.SequenceDecoders import *
from inferno.models.temporal.TemporalFLAME import FlameShapeModel
from inferno.models.temporal.Renderers import FlameRenderer, FixedViewFlameRenderer
from inferno.models.temporal.AudioEncoders import Wav2Vec2Encoder, Wav2Vec2SER

import omegaconf
from omegaconf import open_dict

import torch.nn.functional as F


def sequence_encoder_from_cfg(cfg):
    if cfg.type == "avhubert": 
        from inferno.models.temporal.external.AvHubertSequenceEncoder import AvHubertSequenceEncoder, load_avhubert_model
        path = Path(cfg.checkpoint_folder) / cfg.model_filename
        models, saved_cfg, task = load_avhubert_model(str(path))
        encoder = AvHubertSequenceEncoder(models[0])
    elif cfg.type in ["simple_transformer", "stf"]: 
        encoder = SimpleTransformerSequenceEncoder(cfg)
    elif cfg.type in ["linear"]: 
        encoder = LinearSequenceEncoder(cfg)
    elif cfg.type in ["mlp"]: 
        encoder = MLPSequenceEncoder(cfg)
    elif cfg.type in ["gru"]:
        encoder = GRUSeqEnc(cfg)
    elif cfg.type in ["temporal_cnn"]:
        encoder = TemporalConvNet(cfg)
    elif cfg.type in ["parallel"]:
        encoder = ParallelSequenceEncoder(cfg)
    else: 
        raise ValueError(f"Unknown sequence encoder model type '{cfg.type}'")
    return encoder 


class ParallelSequenceEncoder(SequenceEncoder): 
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        encoders = []
        self.outputs = []
        self.input_feature_dim_ = None
        assert len(cfg.encoders) > 1, "At least two encoders are required"
        self.names = []
        for encoder_cfg in cfg.encoders:
            name = list(encoder_cfg.keys())[0]
            encoder = sequence_encoder_from_cfg(encoder_cfg[name].cfg)
            if self.input_feature_dim_ is None:
                self.input_feature_dim_ = encoder.input_feature_dim()
            assert self.input_feature_dim_ == encoder.input_feature_dim(), \
                "All encoders must have the same input feature dim"
            encoders += [encoder]
            self.outputs += [encoder_cfg.outputs]
            self.names += [name]
        self.encoders = torch.nn.ModuleList(encoders)

    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        # return self.cfg.feature_dim
        return self.input_feature_dim_

    def output_feature_dim(self):
        # dim = 0 
        # for encoder in self.encoders:
        #     dim += encoder.output_feature_dim() 
        dim = self.encoders[0].output_feature_dim()
        return dim

    def forward(self, sample):
        input_feature = sample["fused_feature"]
        # results = []
        results = {}
        for ei, encoder in enumerate(self.encoders):
            # results += [encoder(x)]
            sample_ = {} 
            sample_["fused_feature"] = input_feature 
            results[self.names[ei]] = encoder(sample_)["seq_encoder_output"]
        # results = torch.cat(results, dim=1)
        sample["seq_encoder_output"] = results
        return sample


    

def face_model_from_cfg(cfg): 
    if cfg.type in ["flame", "flame_mediapipe"]: 
        return FlameShapeModel(cfg)
    else:
        raise ValueError(f"Unknown face model type '{cfg.type}'")


def renderer_from_cfg(cfg):
    if cfg.type in ["deca", "emoca"]: 
        renderer = FlameRenderer(cfg)
    elif cfg.type == "fixed_view":
        renderer = FixedViewFlameRenderer(cfg)
    else: 
        raise ValueError(f"Unknown renderer model type '{cfg.type}'")
    return renderer


def norm_from_cfg(cfg, input_tensor_shape):
    if cfg is None or cfg.type == "none":
        return None
    elif cfg.type == "batchnorm2d":
        assert len(input_tensor_shape) == 4 # B, C, H, W
        return torch.nn.BatchNorm2d(input_tensor_shape[1])
    elif cfg.type == "instancenorm2d":
        assert len(input_tensor_shape) == 4 # B, C, H, W
        return torch.nn.InstanceNorm2d(input_tensor_shape[1]) 
    elif cfg.type == "batchnorm1d":
        assert len(input_tensor_shape) == 2 # B, F
        return torch.nn.BatchNorm1d(input_tensor_shape[1])
    elif cfg.type == "instancenorm1d":
        assert len(input_tensor_shape) == 2 # B, F
        return torch.nn.InstanceNorm1d(input_tensor_shape[1])
    elif cfg.type == "layernorm_temporal":
        assert len(input_tensor_shape) == 3  # B, T, F
        return torch.nn.LayerNorm(input_tensor_shape[2])
    elif cfg.type == "layernorm_image":
        assert len(input_tensor_shape) == 4 # B, C, H, W
        return torch.nn.LayerNorm(input_tensor_shape[1:])
    else:
        raise ValueError(f"Unknown norm type '{cfg.type}'")



def sequence_decoder_from_cfg(cfg):
    decoder_cfg = cfg.model.sequence_decoder
    if decoder_cfg is None or decoder_cfg.type in ["none", "no"]:
        return None
    elif decoder_cfg.type == "avhubert": 
        path = Path(cfg.checkpoint_folder) / cfg.model_filename
        models, saved_cfg, task = load_avhubert_model(str(path))
        decoder = AvHubertSequenceDecoder(models[0])
    elif decoder_cfg.type == "fairseq": 
        output_dim = 0 
        if cfg.model.output.predict_shapecode: 
            output_dim += cfg.model.sizes.n_shape 
        if cfg.model.output.predict_expcode:
            output_dim += cfg.model.sizes.n_exp
        if cfg.model.output.predict_texcode:
            output_dim += cfg.model.sizes.n_tex
        # if cfg.model.output.predict_posecode:
        #     output_dim += cfg.model.sizes.n_pose
        if cfg.model.output.predict_jawpose:
            output_dim += cfg.model.sizes.n_pose // 2
        if cfg.model.output.predict_globalpose:
            output_dim += cfg.model.sizes.n_pose // 2
        if cfg.model.output.predict_cam:
            output_dim += cfg.model.sizes.n_cam
        if cfg.model.output.predict_light:
            output_dim += cfg.model.sizes.n_light
        if cfg.model.output.predict_detail:
            output_dim += cfg.model.sizes.n_detail
        
        with open_dict(decoder_cfg):
            decoder_cfg.output_layer_size = output_dim

        decoder = FairSeqModifiedDecoder(decoder_cfg)

    ## FaceFormer related (TODP: probably should be moved to the inferno_apps part. maybe each project should have a BlockFactory)
    elif decoder_cfg.type == "FaceFormerDecoder":
        from inferno.models.talkinghead.FaceFormerDecoder import FaceFormerDecoder
        with open_dict(decoder_cfg):
            decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
        decoder = FaceFormerDecoder(decoder_cfg)
    elif decoder_cfg.type == "FlameFormerDecoder":
        from inferno.models.talkinghead.FaceFormerDecoder import FlameFormerDecoder
        with open_dict(decoder_cfg):
            decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
            decoder_cfg.predict_exp = cfg.model.output.predict_expcode
            decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
        decoder = FlameFormerDecoder(decoder_cfg)
    elif decoder_cfg.type == "LinearDecoder":
        from inferno.models.talkinghead.FaceFormerDecoder import LinearDecoder
        with open_dict(decoder_cfg):
            decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
            decoder_cfg.predict_exp = cfg.model.output.predict_expcode
            decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
        decoder = LinearDecoder(decoder_cfg)
    elif decoder_cfg.type == "LinearAutoRegDecoder":
        from inferno.models.talkinghead.FaceFormerDecoder import LinearAutoRegDecoder
        with open_dict(decoder_cfg):
            decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
            decoder_cfg.predict_exp = cfg.model.output.predict_expcode
            decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
        decoder = LinearAutoRegDecoder(decoder_cfg)
    elif decoder_cfg.type == "BertDecoder":
        from inferno.models.talkinghead.FaceFormerDecoder import BertDecoder 
        with open_dict(decoder_cfg):
            decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
            decoder_cfg.predict_exp = cfg.model.output.predict_expcode
            decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
        decoder = BertDecoder(decoder_cfg)
    elif decoder_cfg.type == "FlameBertDecoder":
        from inferno.models.talkinghead.FaceFormerDecoder import FlameBertDecoder 
        with open_dict(decoder_cfg):
            decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
            decoder_cfg.predict_exp = cfg.model.output.predict_expcode
            decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
        decoder = FlameBertDecoder(decoder_cfg)
    elif decoder_cfg.type == "MLPDecoder":
        from inferno.models.talkinghead.FaceFormerDecoder import MLPDecoder 
        with open_dict(decoder_cfg):
            decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
            decoder_cfg.predict_exp = cfg.model.output.predict_expcode
            decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
        decoder = MLPDecoder(decoder_cfg)
    elif decoder_cfg.type == "BertPriorDecoder":
        from inferno.models.talkinghead.FaceFormerDecoder import BertPriorDecoder 
        with open_dict(decoder_cfg):
            decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
            decoder_cfg.predict_exp = cfg.model.output.predict_expcode
            decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
        decoder = BertPriorDecoder(decoder_cfg)
    else: 
        raise ValueError(f"Unknown sequence decoder model type '{decoder_cfg.type}'")

    return decoder


class NestedPreprocessor(Preprocessor): 

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.preprocessors = {}
        self.prepend_name = cfg.get('prepend_name', False)
        if isinstance(cfg.preprocessors, (list, omegaconf.listconfig.ListConfig)):
            for p in cfg.preprocessors:
                l = list(p.keys())
                assert len(l) == 1, f"Only one preprocessor per entry is allowed, got {l}"
                name = l[0]
                self.preprocessors[name] = preprocessor_from_cfg(p[name])
        else:
            for name, preprocessor_cfg in cfg.preprocessors:
                preprocessor = preprocessor_from_cfg(preprocessor_cfg)
                self.preprocessors[name] = preprocessor

    def to(self, device):
        for preprocessor_name, preprocessor in self.preprocessors.items():
            preprocessor.to(device)
        return self

    def forward(self, batch, input_key, *args, output_prefix="gt_", test_time=False, **kwargs):
        for preprocessor_name, preprocessor in self.preprocessors.items():
            if self.prepend_name:
                output_prefix = preprocessor_name + "_"
            batch = preprocessor(batch, input_key, output_prefix=output_prefix, test_time = test_time)
        return batch
    
    @property
    def device(self):
        return self.preprocessors[list(self.preprocessors.keys())[0]].device

    @property
    def test_time(self):
        return bool(self.cfg.get('test_time', True))


def preprocessor_from_cfg(cfg):
    if cfg.type is None or cfg.type == "none":
        return None
    elif cfg.type == "nested":
        return NestedPreprocessor(cfg)    
    elif cfg.type == "flame":
        return FlamePreprocessor(cfg)
    elif cfg.type == "emoca":
        return EmocaPreprocessor(cfg)
    elif cfg.type == "emorec":
        return EmotionRecognitionPreprocessor(cfg)
    elif cfg.type == "spectre":
        return SpectrePreprocessor(cfg)
    elif cfg.type == "ser":
        return SpeechEmotionRecognitionPreprocessor(cfg)
    else: 
        raise ValueError(f"Unknown preprocess model type '{cfg.model.preprocess.type}'")


def audio_model_from_cfg(cfg):
    if cfg.type == "none":
        return None
    if cfg.type == "avhubert": 
        from inferno.models.temporal.external.AvHubertAudioEncoder import AvHubertAudioEncoder
        from inferno.models.temporal.external.AvHubertSequenceEncoder import load_avhubert_model
        path = Path(cfg.checkpoint_folder) / cfg.model_filename
        models, saved_cfg, task = load_avhubert_model(str(path))
        if 'audio' not in saved_cfg.task.modalities:
            raise ValueError("This AVHubert model does not support audio")
        encoder = AvHubertAudioEncoder(models[0], cfg.trainable)
    elif cfg.type == "wav2vec2": 
        encoder = Wav2Vec2Encoder(cfg.model_specifier, cfg.trainable, 
            with_processor=cfg.get('with_processor', True), 
            expected_fps=cfg.get('model_expected_fps', 50), # 50 fps is the default for wav2vec2 (but not sure if this holds universally)
            target_fps=cfg.get('target_fps', 25), # 25 fps is the default since we use 25 fps for the videos 
            freeze_feature_extractor=cfg.get('freeze_feature_extractor', True),
            dropout_cfg=cfg.get('dropout_cfg', None),
        )
    elif cfg.type == "wav2vec2SER": 
        encoder = Wav2Vec2SER(cfg.model_specifier, cfg.trainable, cfg.get('with_processor', True), 
            expected_fps=cfg.get('model_expected_fps', 50), # 50 fps is the default for wav2vec2 (but not sure if this holds universally)
            target_fps=cfg.get('target_fps', 25), # 25 fps is the default since we use 25 fps for the videos 
            freeze_feature_extractor=cfg.get('freeze_feature_extractor', False),
            dropout_cfg=cfg.get('dropout_cfg', None),
        )
    else: 
        raise ValueError(f"Unknown audio model type '{cfg.type}'")

    return encoder
