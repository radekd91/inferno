import torch


class TemporalAudioEncoder(torch.nn.Module): 

    def __init__(self):
        super().__init__() 

    def forward(self, sample, **kwargs): 
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


class Renderer(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self):
        return []

    def render_coarse_shape(self, sample, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class EmptyDummyModule(torch.nn.Module): 

    def __init__(self):
        super().__init__()

    def forward(self, sample):
        raise NotImplementedError("Should never be called")
