"""
Author: Radek Danecek
Copyright (c) 2023, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emote@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

from inferno.models.temporal.Renderers import FixedViewFlameRenderer
import yaml
from munch import munchify
from inferno_apps.TalkingHead.utils.load import load_model
import torch
import random as rand


def instantiate_fixed_view_renderer(apply_mask):
    renderer_cfg_str = f"""
            type: fixed_view 
            path: /ps/project/EmotionalFacialAnimation/emoca/face_reconstruction_models/new_affectnet_split/final_models/packaged/EMOCA
            face_mask_path: /ps/scratch/rdanecek/data/FLAME/mask/uv_face_mask.png
            face_eye_mask_path: /ps/scratch/rdanecek/data/FLAME/mask/uv_face_eye_mask.png
            uv_size: 256
            image_size: 224
            topology_path: /ps/scratch/rdanecek/data/FLAME/geometry/head_template.obj
            n_light: 27

            # project_landmarks: False
            project_landmarks: True
            cut_out_mouth: True
            mouth_grayscale: False
            apply_mask: {apply_mask}
            cam_names: 
            - front
            # - left_30
            # - right_30

            fixed_cams: 
            - [1.0002211e+01, 1.8661031e-03, 3.0013265e-02]
            # - [9.98946   , 0.02993035, 0.02993339]
            # - [9.953406  , -0.04287567,  0.02838523]

            fixed_poses: 
            - [ 0.04197937,  0.0580671 ,  0.02439648]
            # - [ 0.03752856, -0.50771445,  0.01019305]
            # - [ 0.01612128,  0.6390913 , -0.05253839]

            fixed_light: 
            - [ [ 3.60173798e+00,  3.59747148e+00,  3.60435891e+00],
                [ 3.31493653e-03,  3.47901136e-03,  1.16050988e-03],
                [ 4.95540239e-02,  3.33225392e-02,  2.95860935e-02],
                [-2.23113924e-01, -2.49350727e-01, -2.32230291e-01],
                [ 4.57665697e-02,  4.61878628e-02,  4.63453270e-02],
                [-7.16504231e-02, -7.04488382e-02, -7.28791580e-02],
                [-1.00640759e-01, -1.09357856e-01, -1.12508349e-01],
                [ 2.63972402e-01,  2.63828635e-01,  2.67772079e-01],
                [ 4.96761084e-01,  4.85248387e-01,  4.89809245e-01]]
            """
    # from str to yaml
    renderer_cfg = munchify(yaml.load(renderer_cfg_str, Loader=yaml.FullLoader))
    renderer = FixedViewFlameRenderer(renderer_cfg)
    return renderer


class TalkingHeadWrapper(torch.nn.Module):

    def __init__(self, path_to_model, render_results=True, use_preprocessor=True, apply_mask=True) -> None:
        super().__init__()
        self.talking_head_model, self.cfg = load_model(path_to_model.parent, path_to_model.name, mode='latest', with_losses=False)
        self.talking_head_model.eval()

        self.talking_head_model.renderer = None # delete the renderer, we will use our own
        self.talking_head_model.neural_losses = {} # delete the neural losses, no need to compute them
        # we need to render the faces, if renderer missing
        # self.apply_mask = False
        self.apply_mask = apply_mask
        self.render_results = render_results
        self.use_preprocessor = use_preprocessor
        if self.render_results and self.talking_head_model.renderer is None:
            # instantiate the renderer
            self.renderer = instantiate_fixed_view_renderer(apply_mask)

            # if shape_model is not None and renderer is not None:
            #     self.renderer.set_shape_model(shape_model)
            # elif renderer is not None and self.talking_head_model.sequence_decoder is not None:
            self.renderer.set_shape_model(self.talking_head_model.sequence_decoder.get_shape_model())
            if hasattr(self.talking_head_model.sequence_decoder, 'motion_prior'):
                ## ugly hack. motion priors have been trained with flame without texture 
                ## because the texture is not necessary. But we need it for talking head so we
                ## set it here.
                self.talking_head_model.sequence_decoder.motion_prior.set_flame_tex(self.talking_head_model.preprocessor.get_flametex())
        else:
            self.renderer = None

        if not use_preprocessor:
            self.talking_head_model.preprocessor = None

    def get_num_intensities(self):
        return self.cfg.model.sequence_decoder.style_embedding.n_intensities

    def get_num_emotions(self):
        return self.cfg.model.sequence_decoder.style_embedding.n_expression


    def get_num_identities(self):
        return self.cfg.model.sequence_decoder.style_embedding.n_identities

    def forward(self, sample):
        sample = self.talking_head_model(sample)

        if self.renderer is not None:
            sample = self.renderer(sample, train=False, input_key_prefix='predicted_', output_prefix='predicted_')

            with torch.no_grad():
                if "reconstruction" in sample.keys():
                    for method in self.cfg.data.reconstruction_type:
                        sample["reconstruction"][method] = self.renderer(sample["reconstruction"][method], train=False, input_key_prefix='gt_', output_prefix='gt_') #, **kwargs)
                else:
                    sample = self.renderer(sample, train=False, input_key_prefix='gt_', output_prefix='gt_') #, **kwargs)


        return sample

    def set_neutral_mesh(self, neutral_v):
        # very ugly hack to ensure that the neutral mesh is used (there may be multiple flames by accident in the model)
        self.talking_head_model.sequence_decoder.get_shape_model().v_template[...] = neutral_v
        try:
            self.talking_head_model.sequence_decoder.motion_prior.postprocessor.flame.v_template[...] = neutral_v.clone()
        except AttributeError:
            pass
        try:
            self.talking_head_model.sequence_decoder.motion_prior.preprocessor.flame.v_template[...] = neutral_v.clone()
        except AttributeError:
            pass
        try:
            self.talking_head_model.sequence_decoder.flame.v_template[...] = neutral_v.clone()
        except AttributeError:
            pass
        try:
            self.talking_head_model.preprocessor.flame.v_template[...] = neutral_v.clone()
        except AttributeError:
            pass


    def to(self, device):
        super().to(device)
        self.talking_head_model.to(device)
        if self.renderer is not None:
            self.renderer.to(device)
        return self

    def get_subject_labels(self, train_val_test):
        assert self.cfg.data.data_class == "MEADPseudo3DDM"
        assert "random_by_identityV2" in self.cfg.data.split
        res = self.cfg.data.split.split("_")
        random_or_sorted = res[3]
        assert random_or_sorted in ["random", "sorted"], f"Unknown random_or_sorted value: '{random_or_sorted}'"
        train = float(res[-3])
        val = float(res[-2])
        test = float(res[-1])
        train_ = train / (train + val + test)
        val_ = val / (train + val + test)
        test_ = 1 - train_ - val_
        # indices = np.arange(len(self.video_list), dtype=np.int32)

        identities = \
        """
        M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 
        M027 M028 M029 M030 M031 M032 M033 M034 M035 M037 M039 M040 M041 
        M042 W009 W011 W014 W015 W016 W017 W018 W019 W021 W023 W024 W025 
        W026 W028 W029 W033 W035 W036 W037 W038 W040 
        """
        identities = identities.split()
        # set of identities
        identities = sorted(identities)
        male_identities = [i for i in identities if i.startswith("M")]
        female_identities = [i for i in identities if i.startswith("W")]
        rest = set(identities) - set(male_identities) - set(female_identities)
        assert len(rest) == 0, f"Unexpected identities: {rest}"

        if random_or_sorted == "random":
            seed = 4
            # # get the list of identities
            rand.Random(seed).shuffle(identities)
            # rand.shuffle(identities)

        # training_ids = identities[:int(len(identities) * train_)]
        # validation_ids = identities[int(len(identities) * train_):int(len(identities) * (train_ + val_))]
        # test_ids = identities[int(len(identities) * (train_ + val_)):]

        training_ids = male_identities[:int(len(male_identities) * train_)]
        validation_ids = male_identities[int(len(male_identities) * train_):int(len(male_identities) * (train_ + val_))]
        test_ids = male_identities[int(len(male_identities) * (train_ + val_)):]

        training_ids += female_identities[:int(len(female_identities) * train_)]
        validation_ids += female_identities[int(len(female_identities) * train_):int(len(female_identities) * (train_ + val_))]
        test_ids += female_identities[int(len(female_identities) * (train_ + val_)):]

        # training = []
        # validation = []
        # testing = []
        # for id, indices in identity2idx.items():
        #     if id in training_ids:
        #         training += indices
        #     elif id in validation_ids:
        #         validation += indices
        #     elif id in test_ids:
        #         testing += indices
        #     else:
        #         raise RuntimeError(f"Unassigned identity in training/validation/test split: '{id}'. This should not happen")
        # training.sort()
        # validation.sort()
        # testing.sort()
        if train_val_test == 'training':
            return training_ids
        elif train_val_test == 'validation':
            return validation_ids
        elif train_val_test == 'testing':
            return test_ids
        raise RuntimeError(f"Unknown set_type: '{train_val_test}'")