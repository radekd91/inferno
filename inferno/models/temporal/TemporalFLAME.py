from inferno.models.temporal.Bases import ShapeModel
from inferno.models.DecaFLAME import FLAME, FLAME_mediapipe, FLAMETex 
import torch
from pathlib import Path 
from inferno.utils.other import get_path_to_assets


def check_for_relative_paths(cfg):
    if not Path(cfg.flame_lmk_embedding_path).is_absolute():
        cfg.flame_lmk_embedding_path = str(get_path_to_assets() / cfg.flame_lmk_embedding_path) 
    if not Path(cfg.flame_model_path).is_absolute():
        cfg.flame_model_path = str(get_path_to_assets() / cfg.flame_model_path)
    if 'flame_mediapipe_lmk_embedding_path' in cfg.keys(): 
        if not Path(cfg.flame_mediapipe_lmk_embedding_path).is_absolute():
            cfg.flame_mediapipe_lmk_embedding_path = str(get_path_to_assets() / cfg.flame_mediapipe_lmk_embedding_path)
    if 'tex_path' in cfg.keys():
        if not Path(cfg.tex_path).is_absolute():
            cfg.tex_path = str(get_path_to_assets() / cfg.tex_path)
    return cfg


class FlameShapeModel(ShapeModel): 

    def __init__(self, cfg):
        super().__init__() 
        cfg = check_for_relative_paths(cfg)
        if cfg.type == "flame":
            self.flame = FLAME(cfg)
        elif cfg.type == "flame_mediapipe":
            self.flame = FLAME_mediapipe(cfg)
        else: 
            raise ValueError("Unknown FLAME type: {}".format(self.cfg.type))
        if 'tex_type' in cfg.keys():
            self.flametex = FLAMETex(cfg)
        else: 
            self.flametex = None

    def uses_texture(self): 
        return hasattr(self, "flametex") and self.flametex is not None

    def forward(self, sample):
        shapecode = sample["shapecode"]
        expcode = sample["expcode"]
        if shapecode.shape[-1] < self.flame.cfg.n_shape:
            # pad with zeros 
            missing = self.flame.cfg.n_shape - shapecode.shape[-1]
            shapecode = torch.cat([shapecode, torch.zeros([*shapecode.shape[:-1], missing], device=shapecode.device, dtype=shapecode.dtype)], dim=-1)
        if expcode.shape[-1] < self.flame.cfg.n_exp:
            # pad with zeros 
            missing = self.flame.cfg.n_exp - expcode.shape[-1]
            expcode = torch.cat([expcode, torch.zeros([*expcode.shape[:-1], missing], device=expcode.device, dtype=expcode.dtype)], dim=-1)

        
        if "texcode" not in sample.keys():
            texcode = None
        else:
            texcode = sample["texcode"]
        # posecode = sample["posecode"]
        globpose = sample["globalpose"]
        if "jawpose" in sample.keys():
            jawpose = sample["jawpose"]
        else:
            jawpose = torch.zeros_like(globpose)
        
        posecode = torch.cat([globpose, jawpose], dim=-1)

        if expcode.ndim == 3:
            B = expcode.shape[0]
            T = expcode.shape[1]
        else: 
            B = expcode.shape[0]
            T = None

        if shapecode.ndim != expcode.ndim:
            shapecode = shapecode.unsqueeze(1)
            # repeat T times
            shapecode = shapecode.repeat(1, T, 1)
        assert shapecode.ndim == expcode.ndim, "Shape and expression code must have the same number of dimensions"

        if texcode is not None and texcode.ndim != expcode.ndim:
            texcode = texcode.unsqueeze(1)
            # repeat T times
            texcode = texcode.repeat(1, T, 1)
            assert texcode.ndim == expcode.ndim, "Texture and expression code must have the same number of dimensions"

        # batch-temporal squeeze
        if T is not None:
            shapecode = shapecode.view(B*T, *shapecode.shape[2:])
            expcode = expcode.view(B*T, *expcode.shape[2:])
            if self.uses_texture() and texcode is not None:
                texcode = texcode.view(B*T, *texcode.shape[2:])
            posecode = posecode.view(B*T, *posecode.shape[2:])

        out = self.flame(
                shape_params=shapecode, 
                expression_params=expcode,
                pose_params=posecode)

        if len(out) == 3:
            verts, landmarks2d, landmarks3d = out
            landmarks2d_mediapipe = None
            # landmarks3d_mediapipe = None
        elif len(out) == 4:
            verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = out
        else: 
            raise ValueError(f"Unknown FLAME output shape: {len(out)}")

        if self.uses_texture() and texcode is not None:
            # assert texcode is not None, "Texture code must be provided if using texture"
            albedo = self.flametex(texcode)
        else: 
            # if not using texture, default to gray
            effective_batch_size = shapecode.shape[0]
            # albedo = torch.ones([effective_batch_size, 3, self.deca.config.uv_size, 
            #     self.deca.config.uv_size], device=shapecode.device) * 0.5
            albedo = torch.ones([effective_batch_size, 3, 256, 256], device=shapecode.device) * 0.5

        # batch temporal unsqueeze
        if T is not None:
            verts = verts.view(B, T, *verts.shape[1:])
            landmarks2d = landmarks2d.view(B, T, *landmarks2d.shape[1:])
            landmarks3d = landmarks3d.view(B, T, *landmarks3d.shape[1:])
            if landmarks2d_mediapipe is not None:
                landmarks2d_mediapipe = landmarks2d_mediapipe.view(B, T, *landmarks2d_mediapipe.shape[1:])
            # if landmarks3d_mediapipe is not None:
            #     landmarks3d_mediapipe = landmarks3d_mediapipe.view(B, T, *landmarks3d_mediapipe.shape[1:])
            if self.uses_texture() and texcode is not None:
                albedo = albedo.view(B, T, *albedo.shape[1:])
        
        sample["verts"] = verts
        sample["predicted_landmarks2d_flame_space"] = landmarks2d
        sample["predicted_landmarks3d_flame_space"] = landmarks3d
        if landmarks2d_mediapipe is not None:
            sample["predicted_landmarks2d_mediapipe_flame_space"] = landmarks2d_mediapipe
        # if landmarks3d_mediapipe is not None:
        #     sample["predicted_landmarks3d_mediapipe"] = landmarks3d_mediapipe
        if self.uses_texture() and texcode is not None:
            sample["albedo"] = albedo
        return sample

    def get_landmarks_2d(self, vertices, full_pose):
        B, T = vertices.shape[:2]
        vertices = vertices.view(B*T, *vertices.shape[2:])
        full_pose = full_pose.view(B*T, *full_pose.shape[2:])
        landmarks = self.flame._vertices2landmarks2d(vertices, full_pose)
        landmarks = landmarks.view(B, T, *landmarks.shape[1:])
        return landmarks

    def input_dim(): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self):
        return []