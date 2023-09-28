from torch import Tensor
from gdl.models.temporal.AVFace import *
from gdl.layers.losses import MediaPipeLandmarkLosses as mp_losses
from gdl.layers.losses import FanLandmarkLosses as fan_losses
from gdl.models.temporal.AVFace import Any, Dict
from gdl.models.temporal.Bases import Any, Optional, Renderer, SequenceDecoder, SequenceEncoder, ShapeModel, TemporalAudioEncoder, TemporalVideoEncoder
from gdl.models.temporal.MultiModalTemporalNet import Any, FaceModalities, Optional
from gdl.models.temporal.AudioEncoders import Wav2Vec2Encoder

from gdl.models.FaceReconstruction.FaceRecBase import FaceReconstructionBase
from gdl.models.IO import locate_checkpoint

# path_to_av_hubert = get_path_to_externals() / "av_hubert"
# # path_to_av_hubert = get_path_to_externals() / "av_hubert" / "avhubert"
# # path_to_fairseq = get_path_to_externals() / "av_hubert" / "fairseq"
from gdl.utils.lightning_logging import _fix_image, _log_array_image, _log_wandb_image
from gdl.models.rotation_loss import convert_rot, compute_rotation_loss
from gdl.layers.losses.FaceRecognitionLosses import FaceRecognitionLoss
import math
from gdl.utils.batch import dict_get, check_nan
from munch import Munch, munchify
import pytorch_lightning as pl
import omegaconf

from gdl.models.temporal.Bases import ShapeModel
from gdl.models.temporal.external.LipReadingLoss import LipReadingLoss
from gdl.models.temporal.Renderers import Renderer
from gdl.models.FaceReconstruction.Losses import (FanContourLandmarkLoss, LandmarkLoss,
                     MediaPipeLandmarkLoss, MediaPipeLipDistanceLoss,
                     MediaPipeMouthCornerLoss, MediaPipleEyeDistanceLoss,
                     PhotometricLoss, GaussianRegLoss, LightRegLoss)

from gdl.models.talkinghead.TalkingHeadBase import TalkingHeadBase
from gdl.models.talkinghead.FaceFormer import FaceFormer

from typing import List, Dict

class InputEncoder(torch.nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def get_trainable_parameters(self):
        trainable_params = [] 
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def input_keys(self):
        return 

    def output_keys(self,):
        return
    
    def _assign_prediction_keys(self, batch, encoder_batch):
        if isinstance(self.cfg.predicts, (List, omegaconf.listconfig.ListConfig)):
            for key in self.cfg.predicts:
                assert key not in batch.keys(), f"Key '{key}' already exists in batch - "\
                    "trying to overwrite it is probably not what you want."
                batch[key] = encoder_batch[key]
        elif isinstance(self.cfg.predicts, (Dict, omegaconf.dictconfig.DictConfig)):
            for out_key, in_key in self.cfg.predicts.items():
                assert out_key not in batch.keys(), f"Key '{key}' already exists in batch - "\
                    "trying to overwrite it is probably not what you want."
                batch[out_key] = encoder_batch[in_key]
        else:
            raise ValueError(f"Unsupported type for cfg.predicts: '{type(self.cfg.predicts)}'")
        return batch
    
    def get_dimensionality_dict():
        raise NotImplementedError("Please implement this method in your subclass.")


class FaceReconstructionInputEncoder(InputEncoder):

    def __init__(self, cfg): 
        super().__init__(cfg)
        model_cfg = cfg.model_config
        # load the config 
        model_cfg = omegaconf.OmegaConf.load(model_cfg) 

        checkpoint = locate_checkpoint(model_cfg.coarse, mode = "best")
        self.model = FaceReconstructionBase.instantiate(model_cfg.coarse, checkpoint=checkpoint)

        del self.model.renderer # we don't need the renderer for the input encoder
        self.model.renderer = None

        if not self.cfg.trainable: 
            self.model.requires_grad_(False)


    def get_trainable_parameters(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return trainable_params

    def forward(self, batch: Dict, training=False, validation=False, **kwargs: Any) -> Dict:
        face_rec_batch = {} 

        if "video_masked" in batch.keys():
            face_rec_batch["image"] = batch["video_masked"]
            face_rec_batch["image_original"] = batch["video"]
        else:
            face_rec_batch["image"] = batch["video"]
            face_rec_batch["image_original"] = batch["video"]

        if "mica_video_masked" in batch.keys():
            face_rec_batch["mica_images"] = batch["mica_video_masked"]
            face_rec_batch["mica_images_original"]  = batch["mica_video"]
        elif "mica_video" in batch.keys():
            face_rec_batch["mica_images"] = batch["mica_video"]
                    
        face_rec_batch, ring_size = self.model.unring(face_rec_batch)
        face_rec_batch = self.model.encode(face_rec_batch, return_features=True)
        face_rec_batch = self.model.rering(face_rec_batch, ring_size=ring_size)
        batch = self._assign_prediction_keys(batch, face_rec_batch)
        return batch
    
    def get_dimensionality_dict(self):
        return self.model.face_encoder.get_dimentionality()


class TalkingHeadInputEncoder(InputEncoder):

    def __init__(self, cfg): 
        super().__init__(cfg)

        model_cfg = cfg.model_config
        # load the config 
        model_cfg = omegaconf.OmegaConf.load(model_cfg) 
        checkpoint = locate_checkpoint(model_cfg, mode = "latest")
        pl_module_class = model_cfg.model.pl_module_class
        class_ = class_from_str(pl_module_class, sys.modules[__name__])
        self.model = class_.instantiate(model_cfg, checkpoint=checkpoint)

        if not self.cfg.trainable: 
            self.model.requires_grad_(False)

    def forward(self, batch: Dict, training=False, validation=False, **kwargs: Any) -> Dict:
        talking_head_batch = {} 
        if "processed_audio" in batch.keys():
            talking_head_batch["processed_audio"] = batch["processed_audio"]
            talking_head_batch["samplerate"] = batch["samplerate"]
        else: 
            talking_head_batch["raw_audio"] = batch["raw_audio"]
            talking_head_batch["samplerate"] = batch["samplerate"]

        # talking_head_batch = self.model(talking_head_batch)
        talking_head_batch = self.model.forward_audio(talking_head_batch, train=training, **kwargs)
        
        batch = self._assign_prediction_keys(batch, talking_head_batch)
        return batch
        
    def get_dimensionality_dict(self):
        return {"talking_head_feature" : self.model.audio_model.output_feature_dim()}


class Wav2Vec2InputEncoder(InputEncoder):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.model = Wav2Vec2Encoder(
            cfg.model_specifier, 
            cfg.trainable, 
            with_processor=cfg.get('with_processor', True), 
            expected_fps=cfg.get('model_expected_fps', 50), # 50 fps is the default for wav2vec2 (but not sure if this holds universally)
            target_fps= cfg.get('target_fps', 25), # 25 fps is the default since we use 25 fps for the videos 
            freeze_feature_extractor=   cfg.get('freeze_feature_extractor', True),
            dropout_cfg=cfg.get('dropout_cfg', None),
            )
        
    def get_trainable_parameters(self):
        return self.model.get_trainable_parameters()

    def forward(self, batch, training=False, validation=False):
        w2v_batch = {} 
        if "raw_audio" in batch.keys():
            w2v_batch["raw_audio"] = batch["raw_audio"]
        if "processed_audio" in batch.keys():
            w2v_batch["processed_audio"] = batch["processed_audio"]

        w2v_batch = self.model(w2v_batch)

        batch = self._assign_prediction_keys(batch, w2v_batch)
        return batch

    def get_dimensionality_dict(self):
        return {"wav2vec" : self.model.output_feature_dim()}


def modality_encoder_from_cfg(encoder_cfg) -> torch.nn.Module:
    if encoder_cfg.type == "FaceReconstruction": 
        model = FaceReconstructionInputEncoder(encoder_cfg)
        return model 
    elif encoder_cfg.type == "wav2vec2": 
        encoder = Wav2Vec2InputEncoder(
            encoder_cfg
            )
        return encoder
    elif encoder_cfg.type == "TalkingHead": 
        encoder = TalkingHeadInputEncoder(
            encoder_cfg
            )
        return encoder
    raise NotImplementedError(f"Unsupported modality encoder type: '{encoder_cfg.type}'")


class SequenceEncoder(torch.nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def get_trainable_parameters(self):
        trainable_params = [] 
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params


    def _assign_prediction_keys(self, batch, encoder_batch):
        for key in self.cfg.predicts.keys():
            assert key not in batch.keys(), f"Key '{key}' already exists in batch - "\
                "trying to overwrite it is probably not what you want."
            batch[key] = encoder_batch[key]
        return batch


class ConcatFusion(SequenceEncoder):

    def __init__(self, cfg, input_dims=None) -> None:
        super().__init__(cfg)

        self._input_dims = {} 
        for key in self.cfg.input_keys:
            self._input_dims[key] = input_dims[key]

    def forward(self, batch: Dict, train=False, **kwargs: Any) -> Dict: 
        tensors_to_concat = []
        B,T = None, None
        for key in self.cfg.input_keys:
            B_, T_ = batch[self.cfg.input_keys[0]].shape[:2]
            if B is None:
                B = B_ 
                T = T_
            assert B == B_ and T == T_, "All input tensors must have the same batch size and sequence length."
            tensors_to_concat.append(batch[key])
        
        concatenated_tensor = torch.cat(tensors_to_concat, dim=-1)
        batch[self.cfg.output_keys[0]] = concatenated_tensor
        return batch
    
    def get_dimensionality_dict(self):
        input_dim = 0
        for key in self._input_dims.keys():
            input_dim += self._input_dims[key]
        return {self.cfg.output_keys[0] : input_dim}


class ConcatLinearFusion(ConcatFusion):

    def __init__(self, cfg, input_dims=None) -> None:
        super().__init__(cfg, input_dims)
        self.input_dim = 0 
        for key in self.cfg.input_keys:
            self.input_dim += self._input_dims[key]

        self.linear = torch.nn.Linear(
            in_features=self.input_dim, 
            out_features=self.cfg.output_dim
            )

    def forward(self, batch: Dict, train=False, **kwargs: Any) -> Dict:
        batch = super().forward(batch, train=train, **kwargs)
        concatenated_tensor = batch[self.cfg.output_keys[0]]
        linear_output = self.linear(concatenated_tensor)
        batch[self.cfg.output_keys[0]] = linear_output
        return batch

    def get_dimensionality_dict(self):
        return {self.cfg.output_keys[0] : self.cfg.output_dim}



class OutputDecoder(torch.nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def get_trainable_parameters(self):
        trainable_params = [] 
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def forward(self, batch: Dict, training=False, validation=False, **kwargs: Any) -> Dict:
        raise NotImplementedError("Please implement this method in your subclass.")

    def _assign_prediction_keys(self, batch, decoder_batch):
        for out_key, in_key in self.cfg.predicts.items():
            assert out_key not in batch.keys(), f"Key '{out_key}' already exists in batch - "\
                "trying to overwrite it is probably not what you want."
            batch[out_key] = decoder_batch[in_key]
        return batch



class FlameOutputDecoder(OutputDecoder): 

    def __init__(self, cfg, flame=None) -> None:
        super().__init__(cfg)
        if flame is None:
            from gdl.models.temporal.TemporalFLAME import FlameShapeModel
            self.flame = FlameShapeModel(cfg.flame)
        else: 
            self.flame = flame

    
    def forward(self, batch: Dict, training=False, validation=False, **kwargs: Any) -> Dict:
        flame_batch = {}
        for out_key, in_key in self.cfg.input_keys.items():
            flame_batch[in_key] = batch[out_key]

        flame_batch = self.flame(flame_batch)
        batch = self._assign_prediction_keys(batch, flame_batch)
        return batch



def sequence_encoder_from_cfg(encoder_cfg, dims) -> torch.nn.Module:
    ## - add concat Encoder and other stuff
    # if encoder_
    if encoder_cfg.type == "ConcatFusion":
        encoder = ConcatFusion(encoder_cfg, dims)
        return encoder
    elif encoder_cfg.type == "ConcatLinearFusion":
        encoder = ConcatLinearFusion(encoder_cfg, dims)
        return encoder
    raise NotImplementedError(f"Unsupported sequence encoder type: '{encoder_cfg.type}'")


def sequence_decoder_from_cfg(decoder_cfg) -> torch.nn.Module:
    if decoder_cfg.type == "SimpleTransformer": 
        from gdl.models.temporal.SequenceEncoders import SimpleTransformerSequenceEncoder
        decoder = SimpleTransformerSequenceEncoder(decoder_cfg)
        return decoder
    raise NotImplementedError(f"Unsupported sequence decoder type: '{decoder_cfg.type}'")
    ## add 



def output_decoder_from_cfg(self, decoder_cfg) -> torch.nn.Module:
    if decoder_cfg.type == "FaceRecFlame":
        flame = FlameOutputDecoder(decoder_cfg, flame=self._face_rec_encoder.model.shape_model)
        return flame
    elif decoder_cfg.type == "FlameShapeModel": 
        flame = FlameOutputDecoder(decoder_cfg)
        return flame
    elif decoder_cfg.type == "DecaRenderer":
        from gdl.models.temporal.Renderers import FlameRenderer
        renderer = FlameRenderer(decoder_cfg)
        return renderer
    ## TODO: 
    # - add FLINT 
    raise NotImplementedError(f"Unsupported output decoder type: '{decoder_cfg.type}'")

    


class ConcatFusion(SequenceEncoder):

    def __init__(self, cfg, input_dims=None) -> None:
        super().__init__(cfg)

        self._input_dims = {} 
        for key in self.cfg.input_keys:
            self._input_dims[key] = input_dims[key]

    def forward(self, batch: Dict, train=False, **kwargs: Any) -> Dict: 
        tensors_to_concat = []
        B,T = None, None
        for key in self.cfg.input_keys:
            B_, T_ = batch[self.cfg.input_keys[0]].shape[:2]
            if B is None:
                B = B_ 
                T = T_
            assert B == B_ and T == T_, "All input tensors must have the same batch size and sequence length."
            tensors_to_concat.append(batch[key])
        
        concatenated_tensor = torch.cat(tensors_to_concat, dim=-1)
        batch[self.cfg.output_keys[0]] = concatenated_tensor
        return batch
    
    def get_dimensionality_dict(self):
        input_dim = 0
        for key in self._input_dims.keys():
            input_dim += self._input_dims[key]
        return {self.cfg.output_keys[0] : input_dim}




def losses_from_cfg(losses_cfg, device) -> torch.nn.Module:
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
            from gdl.models.temporal.external.LipReadingLoss import LipReadingLoss
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
        elif loss_type == "MaskedTemporalMSELoss": 
            from gdl.layers.losses.Masked import MaskedTemporalMSELoss 
            loss_func = MaskedTemporalMSELoss(loss_cfg.reduction)
        else: 
            raise NotImplementedError(f"Unsupported loss type: '{loss_type}'")
        loss_functions[loss_name] = loss_func
    return loss_functions


class OutputSeparator(torch.nn.Module): 

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def _separate_output(self, batch, input_vector):
        current_index = 0
        for key in self.cfg.output_keys:
            assert key not in batch.keys(), f"Key '{key}' already exists in batch. Overwriting it is probably not what you want."
            batch[key] = input_vector[..., current_index:current_index+self.cfg.output_dims[key]]
            current_index += self.cfg.output_dims[key]
        assert current_index == input_vector.shape[-1], "The sum of all output dimensions must match the input dimension."
        return batch

    def forward(self, batch, training=False, validation=False, **kwargs: Any) -> Dict:
        input_vector = batch[self.cfg.input_key]
        batch = self._separate_output(batch, input_vector)
        return batch
    
    def get_trainable_parameters(self):
        trainable_params = [] 
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params


class OutputSeparatorLinear(OutputSeparator): 

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.output_dim = 0
        for key in self.cfg.output_keys:
            self.output_dim += self.cfg.output_dims[key]

        self.linear = torch.nn.Linear(
            in_features=self.cfg.input_dim, 
            out_features=self.output_dim
            )
        
        if self.cfg.zero_init: 
            torch.nn.init.constant_(self.linear.weight, 0)
            torch.nn.init.constant_(self.linear.bias, 0)

    
    def forward(self, batch, training=False, validation=False, **kwargs: Any) -> Dict:
        input_vector = batch[self.cfg.input_key]
        input_vector = self.linear(input_vector)
        batch = self._separate_output(batch, input_vector)
        return batch
    


def output_projector_from_cfg(proj_cfg) -> torch.nn.Module:
    if proj_cfg.type == "OutputSeparation":
        proj = OutputSeparator(proj_cfg)
        return proj
    elif proj_cfg.type == "OutputSeparatorLinear":
        proj = OutputSeparatorLinear(proj_cfg)
        return proj
    raise NotImplementedError(f"Unsupported output projector type: '{proj_cfg.type}'")
    


class MultiModalBase(pl.LightningModule): 

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self._dims = {} 

        ## input modality encoders 
        self._initialize_input_encoders()

        ## fuse and encode sequence modules
        self._initialize_sequence_encoders()

        ## decode sequence modules
        self._initialize_sequence_decoders()

        ## decompose the ouput into output modality decoder inputs
        self._initialize_output_code_projectors_decoders()

        ## output decoders
        self._initialize_output_decoders()

        ## losses
        self._initialize_losses()            


    def _initialize_input_encoders(self):
        self.input_encoders = torch.nn.ModuleDict()
        for encoder_name, encoder_cfg in self.cfg.model.input_encoders.items(): 
            self.input_encoders[encoder_name] = modality_encoder_from_cfg(encoder_cfg)
            if type(self.input_encoders[encoder_name]) == FaceReconstructionInputEncoder:
                ## just a handle
                self._face_rec_encoder = self.input_encoders[encoder_name]
            self._dims.update(self.input_encoders[encoder_name].get_dimensionality_dict())


    def _initialize_sequence_encoders(self):
        self.sequence_encoders = torch.nn.ModuleDict()
        for encoder_name, encoder_cfg in self.cfg.model.sequence_encoders.items(): 
            self.sequence_encoders[encoder_name] = sequence_encoder_from_cfg(encoder_cfg, self._dims)  
            self._dims.update(self.sequence_encoders[encoder_name].get_dimensionality_dict())


    def _initialize_sequence_decoders(self):
        self.sequence_decoders = torch.nn.ModuleDict()
        for decoder_name, decoder_cfg in self.cfg.model.sequence_decoders.items(): 
            self.sequence_decoders[decoder_name] = sequence_decoder_from_cfg(decoder_cfg)    
            
    def _initialize_output_code_projectors_decoders(self):
        self.output_latent_projectors = torch.nn.ModuleDict()
        for decoder_name, decoder_cfg in self.cfg.model.output_projectors.items(): 
            self.output_latent_projectors[decoder_name] = output_projector_from_cfg(decoder_cfg)    


    def _initialize_output_decoders(self):
        self.output_decoders = torch.nn.ModuleDict()
        for decoder_name, decoder_cfg in self.cfg.model.output_decoders.items(): 
            self.output_decoders[decoder_name] = output_decoder_from_cfg(self, decoder_cfg)    


    def _initialize_losses(self):
        self.losses = losses_from_cfg(self.cfg.learning.losses, self.device)
        self.metrics = losses_from_cfg(self.cfg.learning.metrics, self.device)


    def get_trainable_parameters(self):
        trainable_params = []
        for name, enc in self.input_encoders.items():
            trainable_params += enc.get_trainable_parameters()
        for name, enc in self.sequence_encoders.items():
            trainable_params += enc.get_trainable_parameters()
        for name, dec in self.sequence_decoders.items():
            trainable_params += dec.get_trainable_parameters()
        for name, proj in self.output_latent_projectors.items():
            trainable_params += proj.get_trainable_parameters()
        for name, dec in self.output_decoders.items():
            trainable_params += dec.get_trainable_parameters()
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

    def to(self, device=None, **kwargs):
        super().to(device=device, **kwargs)
        for key, value in self.losses.items():
            self.losses[key] = value.to(device)
        for key, value in self.metrics.items():
            self.metrics[key] = value.to(device)
        return self

    def _get_dimensionality(self, key):
        return self._dims[key]


    def _forward_modality_encoders(self, batch, training=True, validation=False):
        for name, enc in self.input_encoders.items():
            batch = enc(batch,  training, validation)
        return batch
    
    def _forward_sequence_encoders(self, batch, training=True, validation=False):
        for name, enc in self.sequence_encoders.items():
            batch = enc(batch,  training=training, validation=validation)
        return batch

    def _forward_sequence_decoders(self, batch, training=True, validation=False): 
        for name, enc in self.sequence_decoders.items():
            batch = enc(batch,  training=training, validation=validation)
        return batch
    
    def _forward_output_latent_projectors(self, batch, training=True, validation=False): 
        for name, proj in self.output_latent_projectors.items():
            batch = proj(batch,  training=training, validation=validation)
        return batch

    def _forward_output_decoders(self, batch, training=True, validation=False):
        for name, enc in self.output_decoders.items():
            batch = enc(batch,  training=training, validation=validation)
        return batch

    def forward(self, batch, training=True, validation=False, **kwargs):
        batch = self._forward_modality_encoders(batch, training, validation)
        batch = self._forward_sequence_encoders(batch, training, validation)
        batch = self._forward_sequence_decoders(batch, training, validation)
        batch = self._forward_output_latent_projectors(batch, training, validation)
        batch = self._forward_output_decoders(batch, training, validation)
        return batch


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


    @classmethod
    def instantiate(cls, cfg, stage=None, prefix=None, checkpoint=None, checkpoint_kwargs=None) -> 'MultiModalBase':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = MultiModalBase(cfg)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = MultiModalBase.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=True, 
                **checkpoint_kwargs
            )
        return model

