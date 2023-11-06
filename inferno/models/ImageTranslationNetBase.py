import torch 
from abc import ABC, abstractmethod


class ImageTranslationNetBase(torch.nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, image):
        pass

    @property
    @abstractmethod
    def input_size(self):
        pass

