import os

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn


from config import get_cfg_defaults
from mica.arcface import ArcfaceMica
from mica.decoder import DecoderMica


class Mica(nn.Module):
    def __init__(self, device='cuda:0'):
        super(Mica, self).__init__()
        model_cfg = get_cfg_defaults()
        self.cfg = model_cfg
        self.device = device
        self.encoder = ArcfaceMica().to(self.device)
        self.decoder = DecoderMica(512, 300, 300, 3, model_cfg.model, self.device)
        self.detector = None

        self.load_model()

    def load_model(self):
        model_path = 'TODO: set the path...'
        if os.path.exists(model_path):
            logger.info(f'Trained model found. Path: {model_path} | GPU: {self.device}')
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'arcface' in checkpoint:
                self.encoder.load_state_dict(checkpoint['arcface'])
            if 'flameModel' in checkpoint:
                self.decoder.load_state_dict(checkpoint['flameModel'])
        else:
            logger.error(f'Checkpoint {model_path} not available!')
            exit(-1)

    def model_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict()
        }

    def run(self, images):
        """
        Runs MICA network and return FLAME vertices
        :param images: input images
        :return: 3D vertices
        """
        identity_code = F.normalize(self.encoder(images))
        vertices, code = self.decoder(identity_code)

        return {
            'identity_code': identity_code,
            'shape_code': code,
            'vertices': vertices
        }
