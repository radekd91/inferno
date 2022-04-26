import os
import shutil

import git
import torch

if not os.path.exists('insightfacegit'):
    git.Git('').clone('https://github.com/deepinsight/insightface.git')
    shutil.move('insightface', 'insightfacegit')

from insightfacegit.recognition.arcface_torch.backbones.iresnet import IResNet, IBasicBlock


# Needs pip install insightface
class ArcfaceMica(IResNet):
    def __init__(self, **kwargs):
        super(ArcfaceMica, self).__init__(IBasicBlock, [3, 13, 30, 3], **kwargs)
        arcweights = '/scratch/is-rg-ncs/models_weights/arcface-torch/backbone100.pth'
        self.load_state_dict(torch.load(arcweights))
        self.freezer([self.layer1, self.layer2, self.layer3, self.conv1, self.bn1, self.prelu])

    def freezer(self, layers):
        for layer in layers:
            for block in layer.parameters():
                block.requires_grad = False

    def forward(self, images):
        x = self.forward_arcface(images)
        return x

    def forward_arcface(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.prelu(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)

            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x
