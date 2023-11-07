from ..Preprocessors import Preprocessor 
from inferno.utils.other import get_path_to_externals
import os, sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

path_to_spectre = get_path_to_externals() #/ "spectre"
# path_to_spectre = path_to_spectre.resolve()

if str(path_to_spectre) not in sys.path:
    sys.path.insert(0, str(path_to_spectre))

try: 
    from spectre.src.spectre import SPECTRE 
    from spectre.config import cfg as spectre_cfg
except ImportError as e:
    import traceback
    print("Could not import SPECTRE. Make sure you pull the repository with submodules to enable SPECTRE.")
    print(traceback.format_exc())


class SpectrePreprocessor(Preprocessor): 

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.max_t = cfg.get('max_t', 100)
        spectre_files = Path(path_to_spectre / "spectre" / "data")
        # flame_path = Path(cfg.flame_path)

        # spectre_cfg.model.fixed_displacement_path = os.path.join(cfg.project_dir, 'data', "FLAME2020", "geometry", 'fixed_uv_displacements', 'fixed_displacement_256.npy')
        # spectre_cfg.model.flame_model_path = os.path.join(cfg.project_dir, 'data', 'FLAME2020', 'geometry', 'generic_model.pkl')
        # spectre_cfg.model.flame_lmk_embedding_path = os.path.join(cfg.project_dir, 'data', 'FLAME2020', 'geometry', 'landmark_embedding.npy')
        # spectre_cfg.model.face_mask_path = os.path.join(cfg.project_dir, 'data', 'FLAME2020', 'mask','uv_face_mask.png')
        # spectre_cfg.model.face_eye_mask_path = os.path.join(cfg.project_dir, 'data', 'FLAME2020', 'mask', 'uv_face_eye_mask.png')

        def resolve_path(path):
            if not Path(path).is_absolute(): # if not absolute, prepend the path to spectre project dir
                return str(Path(spectre_cfg.project_dir, cfg.pretrained_modelpath).resolve()) 
            return path

        spectre_cfg.model.fixed_displacement_path = resolve_path(cfg.fixed_displacement_path)
        spectre_cfg.model.flame_model_path = resolve_path(cfg.flame_model_path)
        spectre_cfg.model.flame_lmk_embedding_path = resolve_path(cfg.flame_lmk_embedding_path)
        spectre_cfg.model.face_mask_path = resolve_path(cfg.face_mask_path)
        spectre_cfg.model.face_eye_mask_path = resolve_path(cfg.face_eye_mask_path)
        spectre_cfg.model.tex_path = resolve_path(cfg.tex_path)
        spectre_cfg.pretrained_modelpath = resolve_path(cfg.pretrained_modelpath)



        self.return_vis = cfg.get('return_vis', False)
        self.render = cfg.get('render', False)
        self.crash_on_invalid = cfg.get('crash_on_invalid', True)
        self.with_global_pose = cfg.get('with_global_pose', False)
        self.return_global_pose = cfg.get('return_global_pose', False)
        self.return_appearance = cfg.get('return_appearance', False)
        self.average_shape_decode = cfg.get('average_shape_decode', True)
        self.slice_off_invalid = cfg.get('slice_off_invalid', True) # whether to slice off the invalid frames at the end of the sequence

        self.spectre = SPECTRE(spectre_cfg)
        self.spectre.eval()

        for p in self.spectre.parameters():
            p.requires_grad = False

        # we need to invalidate the first and last num_invalid_frames frames, because of the temporal convolution in the expression encoder
        assert self.spectre.E_expression.temporal[0].stride[0] == 1, "Stride of temporal convolution in expression encoder must be 1"
        assert  self.spectre.E_expression.temporal[0].kernel_size[0] == 5, "Warning: this is not the default kernel size of 5. Are you sure this is what you want?"
        assert  self.spectre.E_expression.temporal[0].padding[0] == 2, "Warning: this is not the default kernel size of 5. Are you sure this is what you want?"
        
        # self.num_invalid_frames = self.spectre.E_expression.temporal[0].kernel_size[0] - self.spectre.E_expression.temporal[0].padding[0] - 1
        self.num_invalid_frames = (self.spectre.E_expression.temporal[0].kernel_size[0] - 1) // 2
        assert self.max_t > self.num_invalid_frames, "Max t must be larger than the number of invalid frames"

    @property
    def device(self):
        return self.spectre.device

    @property
    def test_time(self):
        return bool(self.cfg.get('test_time', True))

    def to(self, device):
        self.spectre = self.spectre.to(device)
        return self

    def forward(self, batch, input_key, *args, output_prefix="gt_", test_time=False, **kwargs):
        if test_time: # if we are at test time
            if not self.test_time: # and the preprocessor is not needed for test time 
                # just return
                return batch
        images = batch[input_key]

        B, T, C, H, W = images.shape

        # spectre has a temporal convolutional layer on the output

        if T > self.max_t:
            codedicts = []
            break_ = False
            for i in range(0, T, self.max_t):
                start_i = i
                # make sure the middle is not cut off by adjusting the indices
                if i > 0:
                    start_i -= self.num_invalid_frames #*2 
                end_i = i + self.max_t + self.num_invalid_frames #*2
                if end_i + self.num_invalid_frames > T: # make sure that the next  chunk is not too small and if it is, add it to the current chunk
                    end_i = T
                    break_ = True
                elif end_i > T:
                    end_i = T
                # images_ = images[:, i * self.max_t : i * (self.maxt+1), ... ]
                images_ = images[:, start_i : end_i, ... ]
                codedict_, initial_deca_exp_, initial_deca_jaw_ = self.spectre.encode(images_)
                codedict_['exp'] = codedict_['exp'] + initial_deca_exp_
                codedict_['pose'][..., 3:] = codedict_['pose'][..., 3:] + initial_deca_jaw_
                
                # cut off the invalid frames
                if i > 0: 
                    for key in codedict_:
                        codedict_[key] = codedict_[key][:, self.num_invalid_frames:, ...]
                if end_i < T:
                    for key in codedict_:
                        codedict_[key] = codedict_[key][:, :-self.num_invalid_frames, ...]
                codedicts.append(codedict_)
                if break_:
                    break  
            
            codedict = {}
            for key in codedicts[0]:
                codedict[key] = torch.cat([codedict_[key] for codedict_ in codedicts], dim=1)
        else:
            codedict, initial_deca_exp, initial_deca_jaw = self.spectre.encode(images)
            codedict['exp'] = codedict['exp'] + initial_deca_exp
            codedict['pose'][..., 3:] = codedict['pose'][..., 3:] + initial_deca_jaw

        # compute the the shapecode only from frames where landmarks are valid
        weights = batch["landmarks_validity"]["mediapipe"] / batch["landmarks_validity"]["mediapipe"].sum(axis=1, keepdims=True)

        if self.crash_on_invalid:
            assert weights.isnan().any() == False, "NaN in weights"
        else: 
            if weights.isnan().any():
                print("[WARNING] NaN in weights")
        
        avg_shapecode = (weights * codedict['shape'].view(B, T, -1)).sum(axis=1, keepdims=False)

        verts, landmarks2d, landmarks3d = self.spectre.flame(
            shape_params=avg_shapecode, 
            expression_params=torch.zeros(device = avg_shapecode.device, dtype = avg_shapecode.dtype, 
                size = (avg_shapecode.shape[0], codedict['exp'].shape[-1])),
            pose_params=None
        )

        if self.average_shape_decode:
            # set the shape to be equal to the average shape (so that the shape is not changing over time)
            codedict['shape'] = avg_shapecode.view(B, 1, -1).repeat(1, T, 1)

        if not self.with_global_pose:
            codedict['pose'][..., :3] = 0

        # this filtering we probably don't need to do? it's done in spectre demo to handle overlapping chunks
        # for key in codedict.keys():
            # if chunk_id == 0 and chunk_id == len(overlapping_indices) - 1:
            #     pass
            # elif chunk_id == 0:
            #     codedict[key] = codedict[key][:-2]
            # elif chunk_id == len(overlapping_indices) - 1:
            #     codedict[key] = codedict[key][2:]
            # else:
            #     codedict[key] = codedict[key][2:-2]

        if self.return_vis:
            opdict, visdict = self.spectre.decode(codedict, rendering=self.render, vis_lmk=False, return_vis=True)

            # plot the visdict using matplotlib
            # for key in visdict.keys():
            #     if key == 'vis':
            #         visdict[key] = visdict[key].unsqueeze(1)

            # B, T, C, H, W to B, T, H, W, C

            # input_images = images.permute(0, 1, 3, 4, 2).cpu().numpy()
            # shape_images = visdict['shape_images'].permute(0, 1, 3, 4, 2).cpu().numpy()
            # for i in range(B):
            #     input_im = np.concatenate( input_images[i].tolist(), axis=1)
            #     shape_im = np.concatenate( shape_images[i].tolist(), axis=1)
            #     img = np.concatenate((input_im, shape_im), axis=0) 
                # plt.figure() 
                # plt.imshow(img)
                # plt.show()


        else: 
            opdict = self.spectre.decode(codedict, rendering=self.render, vis_lmk=False, return_vis=False)



        batch["template"] = verts.contiguous().view(B, -1)
        # batch["template"] = verts.view(B, T, -1, 3)
        # batch[output_prefix + "vertices"] = values['verts'].view(B, T, -1, 3)
        batch[output_prefix + "vertices"] = opdict['verts'].contiguous().view(B, T, -1)
        # batch[output_prefix + 'shape'] = values['shape'].view(B, T, -1)
        if self.average_shape_decode:
            batch[output_prefix + 'shape'] = avg_shapecode
        else:
            batch[output_prefix + 'shape'] = codedict['shape'].view(B, T, -1)
        batch[output_prefix + 'exp'] =  codedict['exp'].view(B, T, -1)
        batch[output_prefix + 'jaw'] = codedict['pose'][..., 3:].contiguous().view(B, T, -1)
        if self.return_global_pose:
            batch[output_prefix + 'global_pose'] = codedict['pose'][..., :3].contiguous().view(B, T, -1)
            batch[output_prefix + 'cam'] = codedict['cam'].view(B, T, -1)


        if self.return_appearance: 
            batch[output_prefix + 'tex'] = codedict['tex'].view(B, T, -1)
            batch[output_prefix + 'light'] = codedict['light'].view(B, T, -1)
            if 'detail' in codedict:
                batch[output_prefix + 'detail'] = codedict['detail'].view(B, T, -1)

        # TODO: this is a little hacky, we need to keep track of which entries are not per-frame (such as template and one_hot identity thingy)
        non_temporal_keys = ['template', 'one_hot', 'samplerate', output_prefix + 'shape', "filename", "fps", "condition_name"] 

        # invalidate the first and last frames for all the per-frame outputs
        if self.slice_off_invalid:
            batch = slice_off_ends(batch, self.num_invalid_frames, non_temporal_keys)
        # for key in batch: 
        #     # if key.startswith(output_prefix):
        #     if key in non_temporal_keys:
        #         continue
        #     entry = batch[key]
        #     batch[key] = entry[:, self.num_invalid_frames: T - self.num_invalid_frames, ...]
            
        return batch


def slice_off_ends(tensor_or_dict, num, keys_to_skip=None): 
    if keys_to_skip is None: 
        keys_to_skip = []
    if isinstance(tensor_or_dict, dict): 
        for key in tensor_or_dict:
            if key in keys_to_skip:
                continue
            entry = tensor_or_dict[key]
            if isinstance(entry, torch.Tensor): 
                tensor_or_dict[key] = entry[:, num: -num, ...]
            elif isinstance(entry, dict): 
                tensor_or_dict[key] = slice_off_ends(entry, num, keys_to_skip)
            else: 
                raise ValueError(f"Unrecognized entry type: '{key}': {type(key)}")
        return tensor_or_dict
    else: 
        return tensor_or_dict[:, num:tensor_or_dict.shape[1]-num, ...]

