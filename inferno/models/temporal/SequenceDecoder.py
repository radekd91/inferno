import torch


class SequenceDecoder(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        

    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")
