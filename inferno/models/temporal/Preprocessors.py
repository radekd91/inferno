import omegaconf 
from pathlib import Path
from inferno.utils.other import get_path_to_assets
from inferno.models.temporal.Bases import Preprocessor
import torch.nn.functional as F
import torch 


def check_flame_paths(flame_cfg):
    flame_model_path = Path(flame_cfg.flame_model_path)
    if not flame_model_path.is_absolute():
        flame_model_path = str(get_path_to_assets() / flame_model_path)
        flame_cfg.flame_model_path = flame_model_path
    flame_lmk_embedding_path = Path(flame_cfg.flame_lmk_embedding_path)
    if not flame_lmk_embedding_path.is_absolute():
        flame_lmk_embedding_path = str(get_path_to_assets() / flame_lmk_embedding_path)
        flame_cfg.flame_lmk_embedding_path = flame_lmk_embedding_path
    return flame_cfg

def check_flametex_paths(flame_cfg):
    tex_path = Path(flame_cfg.tex_path)
    if not tex_path.is_absolute():
        tex_path = str(get_path_to_assets() / tex_path)
        flame_cfg.tex_path = tex_path
    return flame_cfg

class FlamePreprocessor(Preprocessor): 

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        from inferno.models.DecaFLAME import FLAME, FLAMETex
        self.cfg = cfg
        self.cfg.flame = check_flame_paths(cfg.flame)
        self.flame = FLAME(cfg.flame)

        self.flame_tex = None
        if cfg.use_texture:
            self.cfg.flame_tex = check_flametex_paths(cfg.flame_tex)
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

    def forward(self, batch, input_key, *args, output_prefix="gt_", test_time=False, with_grad=False, **kwargs):
        if with_grad:
            return self._forward(batch, input_key, *args, output_prefix=output_prefix, test_time=test_time, **kwargs)
        else:
            with torch.no_grad():
                return self._forward(batch, input_key, *args, output_prefix=output_prefix, test_time=test_time, **kwargs)

    def _forward(self, batch, input_key, *args, output_prefix="gt_", test_time=False, **kwargs):
        if test_time: # if we are at test time
            if not self.test_time: # and the preprocessor is not needed for test time 
                # just return
                return batch
        # from inferno_apps.EMOCA.utils.io import test

        rec_types = []
        if 'gt_exp' in batch:
            rec_types += [None]
        else: 
            rec_types += batch["reconstruction"].keys()

        for rec_type in rec_types:
            rec_dict = batch if rec_type is None else batch["reconstruction"][rec_type]

            B, T = rec_dict['gt_exp'].shape[:2]
            # B, T = nested_dict_access(batch, 'gt_exp', rec_type).shape[:2]
            
            exp = rec_dict['gt_exp'].view(B * T, -1)[..., :self.cfg.flame.n_exp]#.contiguous()
            jaw = rec_dict['gt_jaw'].view(B * T, -1)
            # exp = nested_dict_access(batch, 'gt_exp', rec_type).view(B * T, -1)[..., :self.cfg.flame.n_exp]#.contiguous()
            # jaw = nested_dict_access(batch, 'gt_jaw', rec_type).view(B * T, -1)
            
            global_pose = torch.zeros_like(jaw, device=jaw.device, dtype=jaw.dtype)      
            pose = torch.cat([global_pose, jaw], dim=-1)#.contiguous()


            # exp = torch.zeros((B * T, self.cfg.flame.n_exp))
            # pose = torch.zeros_like((B * T, 6))
            
            if 'gt_shape' not in rec_dict:
                # a little hack -> we use the template mesh (if available) to compute the shape coefficients
                template_mesh = rec_dict['template'].reshape(B,-1,3)
                template_v = template_mesh - self.flame.v_template[None, ...]
                # template_v = -template_mesh + self.flame.v_template[None, ...]
                shape_dirs = self.flame.shapedirs[:, :, :self.cfg.flame.n_shape].view(-1, self.cfg.flame.n_shape)
                norms = shape_dirs.norm(dim=0, keepdim=True)
                s_norm = shape_dirs / norms
                shape_dirs_inv = s_norm.transpose(0, 1) 
                # shape_dirs = self.flame.shapedirs[:, :, :]
                # template_shape_coeffs = torch.einsum('bik,ijk->bj', template_v, shape_dirs.transpose(1, 2))
                # template_shape_coeffs = torch.einsum('bik,ijk->bj', template_v.view(B, -1), shape_dirs.view(-1, 100))
                # template_shape_coeffs2 = template_v.view(B, -1) @ shape_dirs.view(-1, 100).inv()
                # template_shape_coeffs2 = template_v.view(B, -1).contiguous() @ shape_dirs_inv.reshape(-1, 100).contiguous() 
                template_shape_coeffs =  (shape_dirs_inv * (1./ norms.t())) @ template_v.view(B, -1).t()
                template_shape_coeffs = template_shape_coeffs.t()
                rec_dict['gt_shape'] = template_shape_coeffs
                
                # ## sanity check
                # shape_sanity_check_mesh = self.flame(shape_params=template_shape_coeffs, expression_params=torch.zeros((B, self.cfg.flame.n_exp), device=template_v.device, dtype=template_v.dtype), pose_params=None)[0]
                # # rec_err = ((shape_sanity_check_mesh - template_mesh) ** 2).mean().sqrt()
                # import pyvista as pv
                # import numpy as np
                # faces = self.flame.faces_tensor.cpu().numpy() 
                # # concatenate a colums of 3s to the faces array (that's how pyvista wants it)
                # faces = np.concatenate((np.ones((faces.shape[0], 1), dtype=np.int) * 3, faces), axis=1)
                # # create a pyvista mesh for the template 
                # template_mesh_pv = pv.PolyData(template_mesh.cpu().numpy()[0], faces)
                # shape_sanity_check_mesh_pv = pv.PolyData(shape_sanity_check_mesh.cpu().numpy()[0], faces)
                # # plot the template mesh and the mesh reconstructed from the shape coefficients
                # p = pv.Plotter()
                # p.add_mesh(template_mesh_pv, color='red', opacity=0.5, show_edges=True)
                # p.add_mesh(shape_sanity_check_mesh_pv, color='blue', opacity=0.5, show_edges=True)
                # p.show()



            gt_shape = rec_dict['gt_shape'] 
            # gt_shape = nested_dict_access(batch, 'gt_shape', rec_type)

            if gt_shape.ndim == 3:
                template_shape_coeffs = gt_shape[:, 0 :self.cfg.flame.n_shape]#.contiguous()
                shape =  gt_shape[..., :self.cfg.flame.n_shape]#.contiguous()
            else: 
                template_shape_coeffs = gt_shape
                ## shape = batch['gt_shape'].view(B, -1)[:, None, ...].repeat(1, T, 1).contiguous().view(B * T, -1)
                # shape = batch['gt_shape'].view(B, -1)[:, None, ...].repeat(1, T, 1).view(B * T, -1)
                # shape = gt_shape.view(B, -1)[:, None, ...].expand(B, T, template_shape.shape[1]).view(B * T, -1)
                shape = gt_shape.view(B, -1)[:, None, ...].expand(B, T, template_shape_coeffs.shape[1]).reshape(B * T, -1)

            # shape = torch.zeros((B * T, self.cfg.flame.n_exp))
            # template_shape = torch.zeros_like(template_shape)

            verts, landmarks_2D, landmarks_3D = self.flame(
                shape_params=shape, 
                expression_params=exp,
                pose_params=pose
            )

            template_verts, _, _ = self.flame(
                shape_params= template_shape_coeffs,
                expression_params= torch.zeros((B, self.cfg.flame.n_exp), device=self.device, dtype=shape.dtype),
                pose_params=None
            )

            rec_dict["template"] = template_verts.contiguous().view(B, -1)#.detach().clone()        
            rec_dict[output_prefix + 'vertices'] = verts.contiguous().view(B, T, -1)#.detach().clone()

            # nested_dict_set(batch, "template", rec_type, template_verts.contiguous().view(B, -1)) 
            # nested_dict_set(batch, output_prefix+"vertices", rec_type, verts.contiguous().view(B, T, -1)) 

            if self.flame_tex is not None:
                texcode = rec_dict['gt_tex'] 
                # texcode = nested_dict_access(batch, 'gt_tex', rec_type)
                ndim = texcode.ndim
                if ndim == 3:
                    texcode = texcode.view(B *T, -1)
                albedo = self.flame_tex(
                    # texcode = batch['gt_tex']
                    texcode = texcode
                )
                if ndim == 3:
                    albedo_dims = albedo.shape[2:]
                    albedo = albedo.view(B, T, *albedo_dims)
                rec_dict[output_prefix + 'albedo'] = albedo
                ## batch["albedo"] = albedo

                # nested_dict_set(batch, output_prefix + 'albedo', rec_type, albedo) 
                ## nested_dict_set(batch, 'albedo', rec_type, albedo) 

            if rec_type is not None:
                batch["reconstruction"][rec_type] = rec_dict
                
        return batch

    def get_flametex(self):
        return self.flame_tex


def nested_dict_access(dictionary, first_key, key):
    if first_key is not None: 
        return dictionary[first_key][key]
    return dictionary[key]

def nested_dict_set(dictionary, first_key, key, value):
    if first_key is not None: 
        if first_key not in dictionary:
            dictionary[first_key] = {}
        dictionary[first_key][key] = value
    else:
        dictionary[key] = value



class FaceRecPreprocessor(Preprocessor): 

    def __init__(self, cfg, **kwargs):
        from inferno.models.FaceReconstruction.FaceRecBase import FaceReconstructionBase 
        from inferno.models.IO import locate_checkpoint

        self.cfg = cfg
        if not Path(cfg.model_name).is_absolute():
            self.model_name = get_path_to_assets() / "FaceReconstruction/models" / cfg.model_name
        else:
            self.model_name = Path(cfg.model_name)
        self.return_global_pose = cfg.get('return_global_pose', False)
        face_rec_cfg = omegaconf.OmegaConf.load(self.model_name / "cfg.yaml")

        checkpoint = locate_checkpoint(face_rec_cfg, mode = self.cfg.get("checkpoint_mode", "best"))
        face_rec_cfg.learning.losses = {}
        face_rec_cfg.learning.metrics = {}
        self.model = FaceReconstructionBase.instantiate(face_rec_cfg, checkpoint=checkpoint)
        
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.eval()

        self.with_global_pose = cfg.get('with_global_pose', False)
        self.average_shape_decode = cfg.get('average_shape_decode', True)
        self.return_appearance = cfg.get('return_appearance', False)
        self.render = cfg.get('render', False)
        self.crash_on_invalid = cfg.get('crash_on_invalid', True)
        self.max_b = cfg.get('max_b', 100)

    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def forward(self, batch, input_key, *args, output_prefix="gt_", test_time=False, **kwargs):
        if test_time: # if we are at test time
            if not self.test_time: # and the preprocessor is not needed for test time 
                # just return
                return batch
        # from inferno_apps.EMOCA.utils.io import test
        images = batch[input_key]

        B, T, C, H, W = images.shape
        batch_ = {} 
        BT = B*T

        if BT < self.max_b:
            batch_['image'] = images.view(B*T, C, H, W)
            batch_['landmarks'] = {}
            for key in batch['landmarks'].keys():
                batch_['landmarks'][key] = batch['landmarks'][key].view(B*T, -1, 2)
            
            values = self.model(batch_, training=False, validation=False)
        else:
            outputs = []
            for i in range(0, BT, self.max_b):
                batch_ = {} 
                batch_['image'] = images.view(B*T, C, H, W)[i:i+self.max_b]
                batch_['landmarks'] = {}
                for key in batch['landmarks'].keys():
                    batch_['landmarks'][key] = batch['landmarks'][key].view(B*T, -1, 2)[i:i+self.max_b]
                out = self.model(batch_, training=False, validation=False)
                outputs.append(out)
            
            # combine into a single output
            values = cat_tensor_or_dict(outputs, dim=0)


        if not self.with_global_pose:
            values['posecode'][..., :3] = 0

        # # compute the the shapecode only from frames where landmarks are valid
        weights = batch["landmarks_validity"]["mediapipe"] / batch["landmarks_validity"]["mediapipe"].sum(axis=1, keepdims=True)
        if self.crash_on_invalid:
            assert weights.isnan().any() == False, "NaN in weights"
        else: 
            if weights.isnan().any():
                print("[WARNING] NaN in weights")
        avg_shapecode = (weights * values['shapecode'].view(B, T, -1)).sum(axis=1, keepdims=False)

        if self.average_shape_decode:
            # set the shape to be equal to the average shape (so that the shape is not changing over time)
            # values['shapecode'] = avg_shapecode.view(B, 1, -1).repeat(1, T, 1).view(B*T, -1)
            values['shapecode'] = avg_shapecode.view(B, 1, -1)
            values['shapecode'] = values['shapecode'].expand(B, T, values['shapecode'].shape[2]).view(B*T, -1)


        _flame_res = self.model.shape_model.flame(
            shape_params=avg_shapecode, 
            expression_params=torch.zeros(device = avg_shapecode.device, dtype = avg_shapecode.dtype, 
                size = (avg_shapecode.shape[0], values['expcode'].shape[-1])),
            pose_params=None
        )

        if len(_flame_res) == 3:
            verts, landmarks2d, landmarks3d = _flame_res
        elif len(_flame_res) == 4:
            verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = _flame_res
        else:
            raise NotImplementedError("Not implemented for len(_flame_res) = {}".format(len(_flame_res)))

        batch["template"] = verts.contiguous().view(B, -1)

        batch[output_prefix + "vertices"] = values['verts'].contiguous().view(B, T, -1)
        if self.average_shape_decode:
            batch[output_prefix + 'shape'] = avg_shapecode
        else:
            batch[output_prefix + 'shape'] = values['shapecode'].view(B, T, -1)
        batch[output_prefix + 'exp'] =  values['expcode'].view(B, T, -1)
        batch[output_prefix + 'jaw'] = values['jawpose'].view(B, T, -1)
        if self.return_global_pose:
            batch[output_prefix + 'global_pose'] = values['globalpose'].view(B, T, -1)
            batch[output_prefix + 'cam'] = values['cam'].view(B, T, -1)
        if self.return_appearance: 
            batch[output_prefix + 'tex'] = values['texcode'].view(B, T, -1)
            batch[output_prefix + 'light'] = values['lightcode'].view(B, T, -1)
            if 'detailcode' in values:
                batch[output_prefix + 'detail'] = values['detailcode'].view(B, T, -1)
        return batch




class EmocaPreprocessor(Preprocessor): 

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        from inferno_apps.EMOCA.utils.io import load_model
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
        # from inferno_apps.EMOCA.utils.io import test
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
            # values = cat_tensor_or_dict(outputs, dim=0)

            values = {}
            for k in outputs[0].keys():
                if isinstance(outputs[0][k], torch.Tensor):
                    values[k] = torch.cat([o[k] for o in outputs], dim=0)
                elif isinstance(outputs[0][k], dict):
                    values[k] = {}
                    for k2 in outputs[0][k].keys():
                        values[k][k2] = torch.cat([o[k][k2] for o in outputs], dim=0)
                else:
                    raise NotImplementedError("Not implemented for type {}".format(type(outputs[0][k])))

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
            # values['shapecode'] = avg_shapecode.view(B, 1, -1).repeat(1, T, 1).view(B*T, -1)
            values['shapecode'] = avg_shapecode.view(B, 1, -1)
            values['shapecode'] = values['shapecode'].expand(B, T, values['shapecode'].shape[2]).view(B*T, -1)

        if BT < self.max_b:
            values = self.model.decode(values, training=False, render=self.render)
        else:
            outputs = []
            used_keys = ['verts', 'shapecode', 'expcode', 'lightcode', 'texcode', 'posecode', 'cam', 'detailcode']
            
            for i in range(0, BT, self.max_b):
                values_ = {}
                for k in values.keys():
                    if isinstance(values[k], torch.Tensor):
                        values_[k] = values[k][i:i+self.max_b]
                    elif isinstance(values[k], dict):
                        values_[k] = {}
                        for k2 in values[k].keys():
                            values_[k][k2] = values[k][k2][i:i+self.max_b]
                    else:
                        raise NotImplementedError("Not implemented for type {}".format(type(values[k])))
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

        _flame_res = self.model.deca.flame(
            shape_params=avg_shapecode, 
            expression_params=torch.zeros(device = avg_shapecode.device, dtype = avg_shapecode.dtype, 
                size = (avg_shapecode.shape[0], values['expcode'].shape[-1])),
            pose_params=None
        )

        if len(_flame_res) == 3:
            verts, landmarks2d, landmarks3d = _flame_res
        elif len(_flame_res) == 4:
            verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = _flame_res
        else:
            raise NotImplementedError("Not implemented for len(_flame_res) = {}".format(len(_flame_res)))

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
        from inferno_apps.EmotionRecognition.utils.io import load_model
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
    
        from inferno.models.temporal.AudioEncoders import Wav2Vec2SER
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
                    # val = val.repeat(1, T, 1)
                    val = val.expand(val.shape[0], T, val.shape[2])
                assert output_prefix + key not in batch.keys(), f"key {output_prefix + key} already in batch"
                batch[output_prefix + key] = val.view(B, T, -1)
                output_num += 1
        
        assert output_num > 0, "No output was used"

        return batch


def cat_tensor_or_dict(dicts, dim=0):
    """Concatenate a list of tensors or dictionaries along a given dimension.
    Args:
        dicts (list[dict]): List of tensors or dictionaries to concatenate.
        dim (int): Dimension along which to concatenate.
    Returns:

        dict: Concatenated tensor or dictionary.
    """
    outputs = {}
    for k in dicts[0].keys():
        if isinstance(dicts[0][k], torch.Tensor):
            outputs[k] = torch.cat([o[k] for o in dicts], dim=dim)
        elif isinstance(dicts[0][k], dict):
            outputs[k] = cat_tensor_or_dict([o[k] for o in dicts], dim=dim)
        else: 
            raise ValueError(f"Unknown type {type(dicts[0][k])}")
    return outputs
