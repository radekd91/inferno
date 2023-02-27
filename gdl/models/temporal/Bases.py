import torch
from typing import Dict, List, Optional, Tuple, Union, Any


class TemporalFeatureEncoder(torch.nn.Module): 

    def __init__(self):
        super().__init__() 

    def forward(self, sample, train=False, desired_output_length=None, **kwargs): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError()

    def output_feature_dim(self): 
        raise NotImplementedError()


class TemporalAudioEncoder(torch.nn.Module): 

    def __init__(self):
        super().__init__() 

    def forward(self, sample, train=False, desired_output_length=None, **kwargs): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError()

    def output_feature_dim(self): 
        raise NotImplementedError()


class TemporalVideoEncoder(torch.nn.Module): 

    def __init__(self):
        super().__init__()

    def forward(self, sample, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def output_feature_dim(self): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError()


class SequenceEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError("Subclasses must implement this method")

    def input_feature_dim(self):
        raise NotImplementedError("Subclasses must implement this method")

    def output_feature_dim(self):
        raise NotImplementedError("Subclasses must implement this method")


class SequenceDecoder(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        if self.trainable:
            return list(self.parameters())
        return []

    def output_feature_dim(self): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError()

    def get_shape_model(self):
        raise NotImplementedError()


class SequenceClassificationEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        if self.trainable:
            return list(self.parameters())
        return []

    def output_feature_dim(self): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError()


class ClassificationHead(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        if self.trainable:
            return list(self.parameters())
        return []

    def num_classes(self): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError()


class ShapeModel(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def input_dim(): 
        raise NotImplementedError("Subclasses must implement this method")

    def uses_texture(): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError()


class Preprocessor(object):

    def __init__(self):
        super().__init__() 
        
    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def to(self, device):
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def device(self):
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def test_time(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_flametex(self):
        raise NotImplementedError("Subclasses must implement this method")

class Renderer(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self):
        return []

    def render_coarse_shape(self, sample, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def set_shape_model(self, shape_model):
        raise NotImplementedError("Subclasses must implement this method")


class EmptyDummyModule(torch.nn.Module): 

    def __init__(self):
        super().__init__()

    def forward(self, sample):
        raise NotImplementedError("Should never be called")
