import numpy as np
import torch
from inferno.layers.losses.DecaLosses import VGGFace2Loss


class PerceptualLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        raise NotImplementedError()

    def _freeze_layers(self): 
        for param in self.parameters():
            param.requires_grad = False

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class FaceRecognitionLoss(PerceptualLoss):
    
    def __init__(self, cfg):
        super().__init__()
        self.metric = cfg.metric
        self.loss_type = cfg.loss_type
        self.network = cfg.network 
        self.network_checkpoint = cfg.network_checkpoint
        self.trainable = cfg.get('trainable', False)
        if self.network == 'vggface2':
            self.face_loss = VGGFace2Loss( 
                pretrained_checkpoint_path=self.network_checkpoint, 
                metric=self.metric, 
                trainable=self.trainable)
        else:
            raise NotImplementedError(f"Face recognition loss '{self.network}' not implemented")
        if not self.trainable:
            self._freeze_layers()


    def forward(self, input, output):
        return self.face_loss(input, output)

