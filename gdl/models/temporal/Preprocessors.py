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
        self.cfg = cfg
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
        self.average_shape_decode = cfg.get('average_shape_decode', True)

        self.max_b = cfg.get('max_b', 32)

    @property
    def device(self):
        return self.model.device

    def to(self, device):
        self.model.to(device)


    @property
    def test_time(self):
        return bool(self.cfg.get('test_time', True))

    def forward(self, batch, input_key, *args, output_prefix="gt_", test_time=False, **kwargs):
        if test_time: # if we are at test time
            if not self.test_time: # and the preprocessor is not needed for test time 
                # just return
                return batch
        # from gdl_apps.EMOCA.utils.io import test
        images = batch[input_key]

        B, T, C, H, W = images.shape
        batch_ = {} 
        BT = B*T

        if BT < self.max_b:
            batch_['image'] = images.view(B*T, C, H, W)
            values = self.model.encode(batch_, training=False)
        else:
            batch_ = {} 
            batch_['image'] = images.view(B*T, C, H, W)

            outputs = []
            for i in range(0, BT, self.max_b):
                batch_['image'] = images.view(B*T, C, H, W)[i:i+self.max_b]
                outputs.append(self.model(batch_))
            
            # combine into a single output
            values = {}
            for k in outputs[0].keys():
                values[k] = torch.cat([o[k] for o in outputs], dim=0)

        # vals, visdict = decode(deca, batch, vals, training=False)
        values = self.model.encode(batch_, training=False)

        if not self.with_global_pose:
            values['posecode'][..., :3] = 0

        # compute the the shapecode only from frames where landmarks are valid
        weights = batch["landmarks_validity"]["mediapipe"] / batch["landmarks_validity"]["mediapipe"].sum(axis=1, keepdims=True)
        assert weights.isnan().any() == False, "NaN in weights"
        avg_shapecode = (weights * values['shapecode'].view(B, T, -1)).sum(axis=1, keepdims=False)

        if self.average_shape_decode:
            # set the shape to be equal to the average shape (so that the shape is not changing over time)
            values['shapecode'] = avg_shapecode.view(B, 1, -1).repeat(1, T, 1).view(B*T, -1)

        values = self.model.decode(values, training=False, render=False)

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
        self.cfg = cfg
        self.max_b = cfg.get('max_b', 100)
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

    @property
    def test_time(self):
        return bool(self.cfg.get('test_time', True))

    def to(self, device):
        self.model.to(device)

    def forward(self, batch, input_key, *args, output_prefix="gt_", test_time = False, **kwargs):
        output_keys = ['expression', 'valeance', 'arousal']
        
        for key in output_keys:
            if output_prefix + key in batch.keys():
                # a key is already present, this means the preprocessor is not necessary (because the key is part of the dataset)
                return batch

        if test_time: # if we are at test time
            if not self.test_time: # and the preprocessor is not needed for test time 
                # just return
                return batch
        images = batch[input_key]
        B, T, C, H, W = images.shape

        batch_ = {} 
        BT = B*T
        if BT < self.max_b:
            batch_['image'] = images.view(B*T, C, H, W)
            output = self.model(batch_)
        else: 
            outputs = []
            for i in range(0, BT, self.max_b):
                batch_['image'] = images.view(B*T, C, H, W)[i:i+self.max_b]
                outputs.append(self.model(batch_))
            
            # combine into a single output
            output = {}
            for k in outputs[0].keys():
                output[k] = torch.cat([o[k] for o in outputs], dim=0)

        # output_keys = ["valence", "arousal", "emo_feat_2"]

        # for i, key in enumerate(output_keys):
        for i, key in enumerate(output.keys()):
            if key == "expr_classification": 
                # the expression classification is in log space so it should be softmaxed
                batch[output_prefix + "expression"] = F.softmax(output[key].view(B, T, -1), dim=-1)
            else:
                batch[output_prefix + key] = output[key].view(B, T, -1)

        return batch


class SpeechEmotionRecognitionPreprocessor(Preprocessor):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
    
        from gdl.models.temporal.AudioEncoders import Wav2Vec2SER
        self.cfg = cfg
        model_specifier = cfg.model_specifier
        trainable = False 
        # with_processor=True, 
        target_fps=cfg.target_fps #25, 
        expected_fps= cfg.expected_fps # 50, 
        freeze_feature_extractor= True
        dropout_cfg=None

        self.model = Wav2Vec2SER( 
                model_specifier, trainable, 
                with_processor=True, 
                target_fps=target_fps, 
                expected_fps=expected_fps, 
                freeze_feature_extractor=freeze_feature_extractor, 
                dropout_cfg=dropout_cfg,
                )
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    @property
    def device(self):
        return list(self.model.parameters())[0].device

    @property
    def test_time(self):
        return bool(self.cfg.get('test_time', False))

    def to(self, device):
        self.model.to(device)

    def forward(self, batch, input_key, *args, output_prefix="gt_", test_time=False, **kwargs):

        output_keys = ["valence", "arousal", "expression"]
        for key in output_keys:
            if output_prefix + key in batch.keys():
                # a key is already present, this means the preprocessor is not necessary (because the key is part of the dataset)
                return batch

        if test_time: # if we are at test time
            if not self.test_time: # and the preprocessor is not needed for test time 
                # just return
                return batch
        batch_ = {} 
        batch_['raw_audio'] = batch['raw_audio']
        batch_['samplerate'] = batch['samplerate']
        output = self.model(batch_)

        # output_keys = ["valence", "arousal", "emo_feat_2"]
        # self.model.model.config.label2id # labels here: 
        self.model.model.config

        B, T = batch['raw_audio'].shape[:2]

        keys = ["valence", "arousal", "expression"]

        # for i, key in enumerate(output_keys):
        output_num = 0
        for key in keys: 
            if key in output.keys():
                val = output[key]
                if val.ndim == 2:
                    val = val.unsqueeze(1)
                    val = val.repeat(1, T, 1)
                batch[output_prefix + key] = val.view(B, T, -1)
                output_num += 1
        
        assert output_num > 0, "No output was used"

        return batch
