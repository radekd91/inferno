# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import os

import cv2
import torch
import torch.nn.functional as F
# Needed dependency
# pip install -U insightface
# from insightface.utils import face_align
from loguru import logger
from torch import nn

# from .arcface import ArcfaceMica
# from .decoder import DecoderMica
from .arcface import Arcface
from .generator import Generator
from .base_model import BaseModel
from .config import get_cfg_defaults

input_mean = 127.5
input_std = 127.5


def find_model_using_name(model_dir, model_name):
    # adapted from pix2pix framework: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/__init__.py#L25
    # import "model_dir/modelname.py"
    import importlib
    model_filename = model_dir + "." + model_name
    modellib = importlib.import_module(model_filename, package=model_dir)

    # In the file, the class called ModelName() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        # if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a class with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


class MICA(BaseModel):
    def __init__(self, config=None, device=None, tag='MICA', instantiate_flame=True):
        super(MICA, self).__init__(config, device, tag, instantiate_flame)

        self.initialize()

    def create_model(self, model_cfg):
        mapping_layers = model_cfg.mapping_layers
        pretrained_path = None
        if not model_cfg.use_pretrained:
            pretrained_path = model_cfg.arcface_pretrained_model
        self.arcface = Arcface(pretrained_path=pretrained_path).to(self.device)
        self.arcface_feture_size = self.arcface.features.weight.shape[0]
        self.flameModel = Generator(self.arcface_feture_size, 300, self.cfg.model.n_shape, mapping_layers, model_cfg, self.device, 
            instantiate_flame=self.instantiate_flame)

    def load_model(self):
        model_path = os.path.join(self.cfg.output_dir, 'model.tar')
        if os.path.exists(self.cfg.pretrained_model_path) and self.cfg.model.use_pretrained:
            model_path = self.cfg.pretrained_model_path
        if os.path.exists(model_path):
            logger.info(f'[{self.tag}] Trained model found. Path: {model_path} | GPU: {self.device}')
            checkpoint = torch.load(model_path)
            if 'arcface' in checkpoint:
                self.arcface.load_state_dict(checkpoint['arcface'])
            if 'flameModel' in checkpoint:
                ## strict is set to false in case we are not reloading the flame shape model (but only the MLP that regresses it)
                self.flameModel.load_state_dict(checkpoint['flameModel'])
        else:
            logger.info(f'[{self.tag}] Checkpoint not available starting from scratch!')

    def model_dict(self):
        return {
            'flameModel': self.flameModel.state_dict(),
            'arcface': self.arcface.state_dict()
        }

    def parameters_to_optimize(self):
        return [
            {'params': self.flameModel.parameters(), 'lr': self.cfg.train.lr},
            {'params': self.arcface.parameters(), 'lr': self.cfg.train.arcface_lr},
        ]

    def encode(self, images, arcface_imgs):
        codedict = {}

        codedict['arcface'] = F.normalize(self.arcface(arcface_imgs))
        codedict['images'] = images

        return codedict

    def decode(self, codedict, epoch=0, predict_vertices=True):
        self.epoch = epoch

        flame_verts_shape = None
        shapecode = None

        if not self.testing:
            flame = codedict['flame']
            shapecode = flame['shape_params'].view(-1, flame['shape_params'].shape[2])
            shapecode = shapecode.to(self.device)[:, :self.cfg.model.n_shape]
            if predict_vertices:
                with torch.no_grad():
                    flame_verts_shape, _, _ = self.flame(shape_params=shapecode)
            else: 
                flame_verts_shape

        identity_code = codedict['arcface']
        pred_canonical_vertices, pred_shape_code = self.flameModel(identity_code, decode_verts=predict_vertices)

        output = {
            'flame_verts_shape': flame_verts_shape,
            'flame_shape_code': shapecode,
            'pred_canonical_shape_vertices': pred_canonical_vertices,
            'pred_shape_code': pred_shape_code,
            'faceid': codedict['arcface']
        }

        return output

    def compute_losses(self, input, encoder_output, decoder_output):
        losses = {}

        pred_verts = decoder_output['pred_canonical_shape_vertices']
        gt_verts = decoder_output['flame_verts_shape'].detach()

        pred_verts_shape_canonical_diff = (pred_verts - gt_verts).abs()

        if self.use_mask:
            pred_verts_shape_canonical_diff *= self.vertices_mask

        losses['pred_verts_shape_canonical_diff'] = torch.mean(pred_verts_shape_canonical_diff) * 1000.0

        return losses


    def get_feature_size(self):
        return self.arcface_feture_size

# class Mica(nn.Module):
#     def __init__(self, device='cuda:0', model_path=None, instantiate_flame=True):
#         super(Mica, self).__init__()
#         model_cfg = get_cfg_defaults()
#         self.cfg = model_cfg
#         self.device = device
#         self.encoder = ArcfaceMica().to(self.device)
#         self.decoder = DecoderMica(512, 300, 300, 3, model_cfg.model, self.device, 
#             instantiate_flame=instantiate_flame)
#         self.detector = None
#         self.model_path = model_path
#         self.load_model()

#     def load_model(self):
#         # self.model_path = 'TODO: set the path...'
#         if os.path.exists(self.model_path):
#             logger.info(f'Trained model found. Path: {self.model_path} | GPU: {self.device}')
#             checkpoint = torch.load(self.model_path, map_location=self.device)
#             if 'arcface' in checkpoint:
#                 self.encoder.load_state_dict(checkpoint['arcface'])
#             if 'flameModel' in checkpoint:
#                 self.decoder.load_state_dict(checkpoint['flameModel'])
#         else:
#             logger.error(f'Checkpoint {self.model_path} not available!')
#             exit(-1)

#     def model_dict(self):
#         return {
#             'encoder': self.encoder.state_dict(),
#             'decoder': self.decoder.state_dict()
#         }

#     def get_arcface_input(self, img, lmks):
#         aimg = face_align.norm_crop(img, landmark=lmks)
#         blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
#         return blob[0], aimg

#     def crop_images(self, images, lmks):
#         # Keypoint needed for arcface image processing pipeline:

#         # "right_eye"
#         # "left_eye"
#         # "nose"
#         # "mouth_right"
#         # "mouth_left"

#         # images (H, W, 3)

#         arcface_input = self.get_arcface_input(images, lmks)

#         # Load to cuda and transform to tensor?

#         return arcface_input

#     def run(self, images, lmks, decode_verts=True):
#         """
#         Runs MICA network and return FLAME vertices
#         :param images: input images
#         :return: 3D vertices
#         """
#         identity_code = F.normalize(self.encoder(self.crop_images(images, lmks)))
#         vertices, shape_code = self.decoder(identity_code, decode_verts=decode_verts)

#         return {
#             'identity_code': identity_code,
#             'shape_code': shape_code,
#             'vertices': vertices
#         }

#     def forward(self, image, decode_verts=True): 
#         # here uage are already cropped and resized      
#         identity_code = F.normalize(self.encoder(image))
#         vertices, shape_code = self.decoder(identity_code, decode_verts=decode_verts)
#         return vertices, shape_code
