import omegaconf 
from pathlib import Path
from gdl.utils.other import get_path_to_assets
from gdl.models.temporal.Bases import Preprocessor
import torch.nn.functional as F
import torch 


class FlamePreprocessor(Preprocessor): 

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        from gdl.models.DecaFLAME import FLAME, FLAMETex
        self.cfg = cfg
        self.flame = FLAME(cfg.flame)

        self.flame_tex = None
        if cfg.use_texture:
            self.flame_tex = FLAMETex(cfg.flame_tex)

    @property
    def device(self):
        return self.flame.shapedirs.device

    def to(self, device):
        self.flame = self.flame.to(device)
        if self.flame_tex is not None:
            self.flame_tex = self.flame_tex.to(device)
        return self

    @property
    def test_time(self):
        return bool(self.cfg.get('test_time', True))

    @torch.no_grad()
    def forward(self, batch, input_key, *args, output_prefix="gt_", test_time=False, **kwargs):
        if test_time: # if we are at test time
            if not self.test_time: # and the preprocessor is not needed for test time 
                # just return
                return batch
        # from gdl_apps.EMOCA.utils.io import test
        B, T = batch['gt_exp'].shape[:2]
        
        exp = batch['gt_exp'].view(B * T, -1)[..., :self.cfg.flame.n_exp]#.contiguous()
        jaw = batch['gt_jaw'].view(B * T, -1)
        global_pose = torch.zeros_like(jaw, device=jaw.device, dtype=jaw.dtype)      
        pose = torch.cat([global_pose, jaw], dim=-1)#.contiguous()


        # exp = torch.zeros((B * T, self.cfg.flame.n_exp))
        # pose = torch.zeros_like((B * T, 6))

        if batch['gt_shape'].ndim == 3:
            template_shape = batch['gt_shape'][:, 0 :self.cfg.flame.n_shape]#.contiguous()
            shape =  batch['gt_shape'][..., :self.cfg.flame.n_shape]#.contiguous()
        else: 
            template_shape = batch['gt_shape'] 
            # shape = batch['gt_shape'].view(B, -1)[:, None, ...].repeat(1, T, 1).contiguous().view(B * T, -1)
            shape = batch['gt_shape'].view(B, -1)[:, None, ...].repeat(1, T, 1).view(B * T, -1)


        # shape = torch.zeros((B * T, self.cfg.flame.n_exp))
        # template_shape = torch.zeros_like(template_shape)

        verts, _, _ = self.flame(
            shape_params=shape, 
            expression_params=exp,
            pose_params=pose
        )

        template_verts, _, _ = self.flame(
            shape_params= template_shape,
            expression_params= torch.zeros((B, self.cfg.flame.n_exp), device=self.device, dtype=shape.dtype),
            pose_params=None
        )

        batch["template"] = template_verts.contiguous().view(B, -1)#.detach().clone()        
        batch[output_prefix + 'vertices'] = verts.contiguous().view(B, T, -1)#.detach().clone()

        if self.flame_tex is not None:
            texcode = batch['gt_tex'] 
            ndim = texcode.ndim
            if ndim == 3:
                texcode = texcode.view(B *T, -1)
            albedo = self.flame_tex(
                texcode = batch['gt_tex']
            )
            if ndim == 3:
                albedo_dims = albedo.shape[2:]
                albedo = albedo.view(B, T, *albedo_dims)
            batch[output_prefix + 'albedo'] = albedo
            batch["albedo"] = albedo

        return batch

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
        self.return_global_pose = cfg.get('return_global_pose', False)
        self.model, self.model_conf = load_model(self.model_path, self.model_name, self.stage)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.eval()

        self.with_global_pose = cfg.get('with_global_pose', False)
        self.average_shape_decode = cfg.get('average_shape_decode', True)
        self.return_appearance = cfg.get('return_appearance', False)
        self.render = cfg.get('render', False)
        self.crash_on_invalid = cfg.get('crash_on_invalid', True)
        self.max_b = cfg.get('max_b', 100)

    @property
    def device(self):
        return self.model.device

    def to(self, device):
        self.model = self.model.to(device)
        return self


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
            # batch_['image'] = images.view(B*T, C, H, W)

            outputs = []
            for i in range(0, BT, self.max_b):
                batch_['image'] = images.view(B*T, C, H, W)[i:i+self.max_b]
                outputs.append(self.model.encode(batch_, training=False))
            
            # combine into a single output
            values = {}
            for k in outputs[0].keys():
                values[k] = torch.cat([o[k] for o in outputs], dim=0)

        # # vals, visdict = decode(deca, batch, vals, training=False)
        # values = self.model.encode(batch_, training=False)

        if not self.with_global_pose:
            values['posecode'][..., :3] = 0

        # compute the the shapecode only from frames where landmarks are valid
        weights = batch["landmarks_validity"]["mediapipe"] / batch["landmarks_validity"]["mediapipe"].sum(axis=1, keepdims=True)
        if self.crash_on_invalid:
            assert weights.isnan().any() == False, "NaN in weights"
        else: 
            if weights.isnan().any():
                print("[WARNING] NaN in weights")
        avg_shapecode = (weights * values['shapecode'].view(B, T, -1)).sum(axis=1, keepdims=False)

        if self.average_shape_decode:
            # set the shape to be equal to the average shape (so that the shape is not changing over time)
            values['shapecode'] = avg_shapecode.view(B, 1, -1).repeat(1, T, 1).view(B*T, -1)

        if BT < self.max_b:
            values = self.model.decode(values, training=False, render=self.render)
        else:
            outputs = []
            used_keys = ['verts', 'shapecode', 'expcode', 'lightcode', 'texcode', 'posecode', 'cam', 'detailcode']
            
            for i in range(0, BT, self.max_b):
                values_ = {}
                for k in values.keys():
                    values_[k] = values[k][i:i+self.max_b]
                outputs_ = self.model.decode(values_, training=False, render=self.render)
                unused_keys = [k for k in outputs_.keys() if k not in used_keys]
                for key in unused_keys:
                    del outputs_[key]
                outputs.append(outputs_)
            
            # combine into a single output
            values = {}
            for k in outputs[0].keys():
                if outputs[0][k] is not None:
                    values[k] = torch.cat([o[k] for o in outputs], dim=0)

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
        if self.average_shape_decode:
            batch[output_prefix + 'shape'] = avg_shapecode
        else:
            batch[output_prefix + 'shape'] = values['shapecode'].view(B, T, -1)
        batch[output_prefix + 'exp'] =  values['expcode'].view(B, T, -1)
        batch[output_prefix + 'jaw'] = values['posecode'][..., 3:].contiguous().view(B, T, -1)
        if self.return_global_pose:
            batch[output_prefix + 'global_pose'] = values['posecode'][..., :3].contiguous().view(B, T, -1)
            batch[output_prefix + 'cam'] = values['cam'].view(B, T, -1)
        if self.return_appearance: 
            batch[output_prefix + 'tex'] = values['texcode'].view(B, T, -1)
            batch[output_prefix + 'light'] = values['lightcode'].view(B, T, -1)
            if 'detailcode' in values:
                batch[output_prefix + 'detail'] = values['detailcode'].view(B, T, -1)
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
        self.return_features = cfg.get('return_features', False)
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
        self.model = self.model.to(device)
        return self

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

        if self.return_features:
            batch[output_prefix + 'feature'] = output['emo_feat_2'].view(B, T, -1)

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
        return bool(self.cfg.get('test_time', True))

    def to(self, device):
        self.model = self.model.to(device)
        return self

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
