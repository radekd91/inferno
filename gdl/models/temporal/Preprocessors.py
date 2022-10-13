import omegaconf 
from pathlib import Path
from gdl.utils.other import get_path_to_assets
from gdl.models.temporal.Bases import Preprocessor
import torch.nn.functional as F
import torch 


class EmocaPreprocessor(Preprocessor): 

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        from gdl_apps.EMOCA.utils.io import load_model
        if not cfg.model_path:
            self.model_path = get_path_to_assets() / "EMOCA/models"
        else:
            self.model_path = Path(cfg.model_path)
        self.model_name = cfg.model_name
        self.stage = cfg.stage 
        self.model, self.model_conf = load_model(self.model_path, self.model_name, self.stage)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.eval()

        self.with_global_pose = cfg.get('with_global_pose', False)

    @property
    def device(self):
        return self.model.device

    def to(self, device):
        self.model.to(device)

    def forward(self, batch, input_key, *args, output_prefix="gt_", **kwargs):
        # from gdl_apps.EMOCA.utils.io import test
        images = batch[input_key]

        B, T, C, H, W = images.shape

        batch_ = {} 
        batch_['image'] = images.view(B*T, C, H, W)

        # vals, visdict = decode(deca, batch, vals, training=False)
        values = self.model.encode(batch_, training=False)

        if not self.with_global_pose:
            values['posecode'][..., :3] = 0

        values = self.model.decode(values, training=False)
        
        # compute the the shapecode only from frames where landmarks are valid
        weights = batch["landmarks_validity"]["mediapipe"] / batch["landmarks_validity"]["mediapipe"].sum(axis=1, keepdims=True)
        assert weights.isnan().any() == False, "NaN in weights"
        avg_shapecode = (weights * values['shapecode'].view(B, T, -1)).sum(axis=1, keepdims=False)

        verts, landmarks2d, landmarks3d = self.model.deca.flame(
            shape_params=avg_shapecode, 
            expression_params=torch.zeros(device = avg_shapecode.device, dtype = avg_shapecode.dtype, 
                size = (avg_shapecode.shape[0], values['expcode'].shape[-1])),
            pose_params=None
        )

        batch["template"] = verts.contiguous().view(B, -1)
        # batch["template"] = verts.view(B, T, -1, 3)
        # batch[output_prefix + "vertices"] = values['verts'].view(B, T, -1, 3)
        batch[output_prefix + "vertices"] = values['verts'].contiguous().view(B, T, -1)
        # batch[output_prefix + 'shape'] = values['shapecode'].view(B, T, -1)
        batch[output_prefix + 'shape'] = avg_shapecode
        batch[output_prefix + 'exp'] =  values['expcode'].view(B, T, -1)
        batch[output_prefix + 'jaw'] = values['posecode'][..., 3:].contiguous().view(B, T, -1)
        return batch


class EmotionRecognitionPreprocessor(Preprocessor):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
    
        from gdl_apps.EmotionRecognition.utils.io import load_model
        self.cfg = cfg
        if not cfg.model_path:
            self.model_path = get_path_to_assets() / "EmotionRecognition" / "image_based_networks"
        else:
            self.model_path = Path(cfg.model_path)
        self.model_name = cfg.model_name
        self.model = load_model(self.model_path / self.model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.eval()

    @property
    def device(self):
        return self.model.device

    def to(self, device):
        self.model.to(device)

    def forward(self, batch, input_key, *args, output_prefix="gt_", **kwargs):
        images = batch[input_key]
        B, T, C, H, W = images.shape

        batch_ = {} 
        batch_['image'] = images.view(B*T, C, H, W)
        output = self.model(batch_)

        # output_keys = ["valence", "arousal", "emo_feat_2"]

        # for i, key in enumerate(output_keys):
        for i, key in enumerate(output.keys()):
            if key == "expr_classification": 
                # the expression classification is in log space so it should be softmaxed
                batch[output_prefix + "expression"] = F.softmax(output[key].view(B, T, -1), dim=-1)
            else:
                batch[output_prefix + key] = output[key].view(B, T, -1)

        return batch


