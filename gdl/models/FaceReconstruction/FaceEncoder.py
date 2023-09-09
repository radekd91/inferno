import torch 
import numpy as np 
import os, sys 
from ..DecaEncoder import ResnetEncoder, SwinEncoder
from pathlib import Path
import copy
from omegaconf import OmegaConf
# import timeit


class FaceEncoderBase(torch.nn.Module):
    
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, batch):
        return self.encode(batch)
    
    def get_trainable_parameters(self):
        raise NotImplementedError("Abstract method")
    
    def encode(self, x):
        raise NotImplementedError("Abstract method")
    
    def _prediction_code_dict(self):
        prediction_code_dict = OmegaConf.to_container(self.cfg.predicts) 
        return prediction_code_dict

    
    def _get_codevector_dim(self):
        prediction_code_dict = self._prediction_code_dict()
        return sum([dim for _, dim in prediction_code_dict.items()])
    


class DecaEncoder(FaceEncoderBase):

    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.encoder = ResnetEncoder(self._get_codevector_dim(), None)
        self.trainable = self.cfg.trainable
        if not self.trainable:
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        if not self.trainable:
            return super().train(False)
        return super().train(mode)

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def encode(self, batch):
        image = batch['image']
        # time = timeit.default_timer()
        code_vec = self.encoder(image)
        # time_enc = timeit.default_timer()
        batch = self._decompose_code(batch, code_vec)
        # time_decomp = timeit.default_timer()
        # print(f"Time encoding:\t\t{time_enc - time:0.05f}")
        # print(f"Time decomposing:\t{time_decomp - time_enc:0.05f}")
        return batch
    
    def _decompose_code(self, batch, code):
        '''
        Decompose the code into the different components based on the prediction_code_dict
        '''
        prediction_code_dict = self._prediction_code_dict()
        start = 0
        for key, dim in prediction_code_dict.items():
            subcode = code[..., start:start + dim]
            if key == 'light':
                subcode = subcode.reshape(subcode.shape[0], 9, 3)
            batch[key] = subcode
            start = start + dim

        return batch

    def _get_num_shape_params(self): 
        return self.config.n_shape
    


class MicaEncoder(FaceEncoderBase): 

    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        from ..mica.config import get_cfg_defaults
        from ..mica.mica import MICA
        from ..mica.MicaInputProcessing import MicaInputProcessor
        self.use_mica_shape_dim = True
        # self.use_mica_shape_dim = False
        self.mica_cfg = get_cfg_defaults()

        if Path(self.cfg.mica_model_path).exists(): 
            mica_path = self.cfg.mica_model_path 
        else:
            from gdl.utils.other import get_path_to_assets
            mica_path = get_path_to_assets() / self.cfg.mica_model_path  
            assert mica_path.exists(), f"MICA model path does not exist: '{mica_path}'"

        self.mica_cfg.pretrained_model_path = str(mica_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.E_mica = MICA(self.mica_cfg, device, str(mica_path), instantiate_flame=False)
        # E_mica should be fixed 
        self.E_mica.requires_grad_(False)
        self.E_mica.testing = True
        self.mica_preprocessor = MicaInputProcessor(self.cfg.get('mica_preprocessing', False))

    def get_trainable_parameters(self):
        # return [p for p in self.model.parameters() if p.requires_grad]
        return [] # MICA training is not supported, we take the pretrained model

    def train(self, mode: bool = True):
        return super().train(False) # always in eval mode

    def encode(self, batch):
        image = batch['image']
        if 'mica_images' in batch.keys():
            mica_image = batch['mica_images']
        else:
            # time = timeit.default_timer()
            fan_landmarks = None
            if "landmarks" in batch.keys():
                if isinstance(batch["landmarks"], dict):
                    if "fan3d" in batch["landmarks"].keys():
                        fan_landmarks = batch["landmarks"]["fan3d"]
                    elif "fan" in batch["landmarks"].keys():
                        fan_landmarks = batch["landmarks"]["fan"]
                elif isinstance(batch["landmarks"], (np.ndarray, torch.Tensor)):
                    if batch["landmarks"].shape[1] == 68:
                        fan_landmarks = batch["landmarks"]
            print("[WARNING] Processing MICA image in forward pass. This is very inefficient for training."\
                  " Please precompute the MICA images in the data loader.")
            mica_image = self.mica_preprocessor(image, fan_landmarks)
            # time_preproc = timeit.default_timer()
            # print(f"Time preprocessing:\t{time_preproc - time:0.05f}")
        mica_code = self.E_mica.encode(image, mica_image) 
        mica_code = self.E_mica.decode(mica_code, predict_vertices=False)
        mica_shapecode = mica_code['pred_shape_code']
        batch['shapecode'] = mica_shapecode
        return batch
    
    def _get_num_shape_params(self): 
        return self.mica_cfg.model.n_shape   


class MicaDecaEncoder(FaceEncoderBase): 

    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.mica_encoder = MicaEncoder(cfg=self.cfg.encoders.mica_encoder)
        self.deca_encoder = DecaEncoder(cfg=self.cfg.encoders.deca_encoder)

    def get_trainable_parameters(self):
        return self.mica_encoder.get_trainable_parameters() + self.deca_encoder.get_trainable_parameters()

    def train(self, mode: bool = True):
        return super().train(mode)

    def encode(self, batch):
        batch = self.mica_encoder.encode(batch)
        batch = self.deca_encoder.encode(batch)
        return batch


class ExpressionEncoder(DecaEncoder): 

    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

    def initialize_from(self, other_encoder):
        other_state_dict = copy.deepcopy(other_encoder.state_dict())
        self.encoder.load_state_dict(other_state_dict)


class EmocaEncoder(FaceEncoderBase):

    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.deca_encoder = DecaEncoder(config=self.config)
        self.expression_encoder = ExpressionEncoder(config=self.config)
        # self.initialize_expression_encoder(self.deca_encoder)

    def initialize_expression_encoder(self, other_encoder):
        self.expression_encoder.initialize_from(other_encoder)

    def encode(self, batch):
        batch = self.deca_encoder.encode(batch)
        batch = self.expression_encoder.encode(batch)
        return batch
    
    def get_trainable_parameters(self):
        return self.deca_encoder.get_trainable_parameters() + self.deca_encoder.get_trainable_parameters()


class EmicaEncoder(FaceEncoderBase): 

    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.mica_deca_encoder = MicaDecaEncoder(cfg=self.cfg.encoders.mica_deca_encoder) 
        self.expression_encoder = ExpressionEncoder(cfg=self.cfg.encoders.expression_encoder)
        # self.initialize_expression_encoder(self.deca_encoder)

    def initialize_expression_encoder(self, other_encoder):
        self.expression_encoder.initialize_from(other_encoder)

    def encode(self, batch):
        batch = self.mica_deca_encoder.encode(batch)
        batch = self.expression_encoder.encode(batch)
        return batch
    
    def get_trainable_parameters(self):
        return self.mica_deca_encoder.get_trainable_parameters() + self.expression_encoder.get_trainable_parameters()
    

def encoder_from_cfg(cfg):
    enc_cfg = cfg.model.face_encoder
    if enc_cfg.type == "DecaEncoder":
        encoder = DecaEncoder(cfg=enc_cfg)
    elif enc_cfg.type == "MicaDecaEncoder":
        encoder = MicaDecaEncoder(cfg=enc_cfg)
    elif enc_cfg.type == "EmocaEncoder": 
        encoder = EmocaEncoder(cfg=enc_cfg)
    elif enc_cfg.type == "EmicaEncoder":
        encoder = EmicaEncoder(cfg=enc_cfg)
    else:
        raise NotImplementedError(f"Encoder type '{enc_cfg.type}' not implemented.")
    return encoder