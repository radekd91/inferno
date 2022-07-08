from gdl.models.temporal.Bases import ShapeModel
from gdl.models.DecaFLAME import FLAME, FLAME_mediapipe, FLAMETex 
import torch


class FlameShapeModel(ShapeModel): 

    def __init__(self, cfg):
        super().__init__() 
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
        texcode = sample["texcode"]
        # posecode = sample["posecode"]
        jawpose = sample["jawpose"]
        globpose = sample["globalpose"]
        posecode = torch.cat([globpose, jawpose], dim=-1)

        B = shapecode.shape[0]
        T = shapecode.shape[1]

        # batch-temporal squeeze
        shapecode = shapecode.view(B*T, *shapecode.shape[2:])
        expcode = expcode.view(B*T, *expcode.shape[2:])
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

        if self.uses_texture():
            albedo = self.flametex(texcode)
        else: 
            # if not using texture, default to gray
            effective_batch_size = shapecode.shape[0]
            albedo = torch.ones([effective_batch_size, 3, self.deca.config.uv_size, 
                self.deca.config.uv_size], device=shapecode.device) * 0.5

        # batch temporal unsqueeze
        verts = verts.view(B, T, *verts.shape[1:])
        landmarks2d = landmarks2d.view(B, T, *landmarks2d.shape[1:])
        landmarks3d = landmarks3d.view(B, T, *landmarks3d.shape[1:])
        if landmarks2d_mediapipe is not None:
            landmarks2d_mediapipe = landmarks2d_mediapipe.view(B, T, *landmarks2d_mediapipe.shape[1:])
        # if landmarks3d_mediapipe is not None:
        #     landmarks3d_mediapipe = landmarks3d_mediapipe.view(B, T, *landmarks3d_mediapipe.shape[1:])
        albedo = albedo.view(B, T, *albedo.shape[1:])
        
        sample["verts"] = verts
        sample["predicted_landmarks2d_flame_space"] = landmarks2d
        sample["predicted_landmarks3d_flame_space"] = landmarks3d
        if landmarks2d_mediapipe is not None:
            sample["predicted_landmarks2d_mediapipe_flame_space"] = landmarks2d_mediapipe
        # if landmarks3d_mediapipe is not None:
        #     sample["predicted_landmarks3d_mediapipe"] = landmarks3d_mediapipe
            
        sample["albedo"] = albedo
        return sample

    def input_dim(): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self):
        return []