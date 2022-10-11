from ..Preprocessors import Preprocessor 
from gdl.utils.other import get_path_to_externals
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
    print("Could not import SPECTRE. Make sure you pull the repository with submodules to enable SPECTRE.")


class SpectrePreprocessor(Preprocessor): 

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        
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

    @property
    def device(self):
        return self.spectre.device

    def to(self, device):
        self.spectre = self.spectre.to(device)
        return self

    def forward(self, batch, input_key, *args, output_prefix="gt_", **kwargs):
        images = batch[input_key]

        B, T, C, H, W = images.shape

        # spectre has a temporal convolutional layer on the output

        codedict, initial_deca_exp, initial_deca_jaw = self.spectre.encode(images)
        codedict['exp'] = codedict['exp'] + initial_deca_exp
        codedict['pose'][..., 3:] = codedict['pose'][..., 3:] + initial_deca_jaw

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

            input_images = images.permute(0, 1, 3, 4, 2).cpu().numpy()
            shape_images = visdict['shape_images'].permute(0, 1, 3, 4, 2).cpu().numpy()
            for i in range(B):
                input_im = np.concatenate( input_images[i].tolist(), axis=1)
                shape_im = np.concatenate( shape_images[i].tolist(), axis=1)
                img = np.concatenate((input_im, shape_im), axis=0) 

                plt.figure() 
                plt.imshow(img)
                plt.show()

                # for j in range(T):
                #     plt.figure() 
                #     # concatenate input and shape image 
                #     # reshape the sequence to a single image
                #     img = np.concatenate((input_images[i, j], shape_images[i, j]), axis=1)
                #     plt.imshow(img)
                #     plt.show()

            visdict = torch.cat([visdict[key].cpu() for key in visdict.keys()], dim=1)
            visdict = visdict[0].permute(0, 2, 3, 1)
            visdict = visdict.reshape(B, T, H, W, -1)
            # to numpy 
            visdict = visdict.cpu().numpy()
            visdict = visdict.astype(np.uint8)


        else: 
            opdict = self.spectre.decode(codedict, rendering=self.render, vis_lmk=False, return_vis=False)

        # compute the the shapecode only from frames where landmarks are valid
        weights = batch["landmarks_validity"]["mediapipe"] / batch["landmarks_validity"]["mediapipe"].sum(axis=1, keepdims=True)
        assert weights.isnan().any() == False, "NaN in weights"
        avg_shapecode = (weights * codedict['shape'].view(B, T, -1)).sum(axis=1, keepdims=False)


        verts, landmarks2d, landmarks3d = self.spectre.flame(
            shape_params=avg_shapecode, 
            expression_params=torch.zeros(device = avg_shapecode.device, dtype = avg_shapecode.dtype, 
                size = (avg_shapecode.shape[0], codedict['exp'].shape[-1])),
            pose_params=None
        )


        batch["template"] = verts.contiguous().view(B, -1)
        # batch["template"] = verts.view(B, T, -1, 3)
        # batch[output_prefix + "vertices"] = values['verts'].view(B, T, -1, 3)
        batch[output_prefix + "vertices"] = opdict['verts'].contiguous().view(B, T, -1)
        # batch[output_prefix + 'shape'] = values['shape'].view(B, T, -1)
        batch[output_prefix + 'shape'] = avg_shapecode
        batch[output_prefix + 'exp'] =  codedict['exp'].view(B, T, -1)
        batch[output_prefix + 'jaw'] = codedict['pose'][..., 3:].contiguous().view(B, T, -1)

        # TODO: this is a little hacky, we need to keep track of which entries are not per-frame (such as template and one_hot identity thingy)
        non_temporal_keys = ['template', 'one_hot', 'samplerate'] 

        # invalidate the first and last frames for all the per-frame outputs
        for key in batch: 
            # if key.startswith(output_prefix):
            if key in non_temporal_keys:
                continue
            batch[key] = batch[key][:, self.num_invalid_frames: T - self.num_invalid_frames, ...]
            
        return batch