import os

import cv2
import torch
import torch.nn.functional as F
# Needed dependency
# pip install -U insightface
from insightface.utils import face_align
from loguru import logger
from torch import nn

from .arcface import ArcfaceMica
from .decoder import DecoderMica
from .config import get_cfg_defaults

input_mean = 127.5
input_std = 127.5


class Mica(nn.Module):
    def __init__(self, device='cuda:0', model_path=None, instantiate_flame=True):
        super(Mica, self).__init__()
        model_cfg = get_cfg_defaults()
        self.cfg = model_cfg
        self.device = device
        self.encoder = ArcfaceMica().to(self.device)
        self.decoder = DecoderMica(512, 300, 300, 3, model_cfg.model, self.device, 
            instantiate_flame=instantiate_flame)
        self.detector = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        # self.model_path = 'TODO: set the path...'
        if os.path.exists(self.model_path):
            logger.info(f'Trained model found. Path: {self.model_path} | GPU: {self.device}')
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'arcface' in checkpoint:
                self.encoder.load_state_dict(checkpoint['arcface'])
            if 'flameModel' in checkpoint:
                self.decoder.load_state_dict(checkpoint['flameModel'])
        else:
            logger.error(f'Checkpoint {self.model_path} not available!')
            exit(-1)

    def model_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict()
        }

    def get_arcface_input(self, img, lmks):
        aimg = face_align.norm_crop(img, landmark=lmks)
        blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
        return blob[0], aimg

    def crop_images(self, images, lmks):
        # Keypoint needed for arcface image processing pipeline:

        # "right_eye"
        # "left_eye"
        # "nose"
        # "mouth_right"
        # "mouth_left"

        # images (H, W, 3)

        arcface_input = self.get_arcface_input(images, lmks)

        # Load to cuda and transform to tensor?

        return arcface_input

    def run(self, images, lmks, decode_verts=True):
        """
        Runs MICA network and return FLAME vertices
        :param images: input images
        :return: 3D vertices
        """
        identity_code = F.normalize(self.encoder(self.crop_images(images, lmks)))
        vertices, shape_code = self.decoder(identity_code, decode_verts=decode_verts)

        return {
            'identity_code': identity_code,
            'shape_code': shape_code,
            'vertices': vertices
        }

    def forward(self, image, decode_verts=True): 
        # here uage are already cropped and resized      
        identity_code = F.normalize(self.encoder(image))
        vertices, shape_code = self.decoder(identity_code, decode_verts=decode_verts)
        return vertices, shape_code
