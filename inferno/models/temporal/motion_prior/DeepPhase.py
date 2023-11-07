"""
Code taken over from: https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2022/PyTorch/PAE/PAE.py
and adapted to the INFERNO framework.
"""
from .MotionPrior import * 
from torch import nn
from ..BlockFactory import preprocessor_from_cfg
from omegaconf import open_dict
from inferno.utils.other import class_from_str
import sys
# from inferno.models.temporal.PositionalEncodings import positional_encoding_from_cfg
# from inferno.models.temporal.TransformerMasking import biased_mask_from_cfg
from munch import Munch, munchify
from omegaconf import OmegaConf


class DeepPhase(MotionPrior):
    """
    An attempt to implement the DeepPhase model. Not finished, not tested.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        input_dim = self.get_input_sequence_dim()
        
        encoder_class = class_from_str(cfg.model.sequence_encoder.type, sys.modules[__name__])
        decoder_class = class_from_str(cfg.model.sequence_decoder.type, sys.modules[__name__])

        with open_dict(cfg.model.sequence_encoder):
            cfg.model.sequence_encoder.input_dim = input_dim

        with open_dict(cfg.model.sizes):
            # assert cfg.learning.batching.sequence_length_train % (2 **cfg.model.sizes.quant_factor) == 0, \
            #     "Sequence length must be divisible by quantization factor"
            # cfg.model.sizes.quant_sequence_length = cfg.learning.batching.sequence_length_train // (2 **cfg.model.sizes.quant_factor)
            cfg.model.sizes.sequence_length = cfg.learning.batching.sequence_length_train 
            cfg.model.sizes.fps = cfg.data.get('fps', 25)

        motion_encoder = encoder_class(cfg.model.sequence_encoder, cfg.model.sizes)
        motion_decoder = decoder_class(cfg.model.sequence_decoder, cfg.model.sizes, motion_encoder.get_input_dim())
        # motion_quantizer = VectorQuantizer(cfg.model.quantizer)
        if cfg.model.get('quantizer', None) is not None:
            motion_quantizer = class_from_str(cfg.model.quantizer.type, sys.modules[__name__])(cfg.model.quantizer)
        else:
            motion_quantizer = None
        if cfg.model.get('preprocessor', None) is not None:
            preprocessor = preprocessor_from_cfg(cfg.model.preprocessor)
        else:
            preprocessor = None
        super().__init__(cfg, motion_encoder, motion_decoder, motion_quantizer, preprocessor, preprocessor)

    def get_bottleneck_dim(self):
        return self.motion_encoder.bottleneck_dim()

    def get_codebook_size(self): 
        if self.motion_quantizer is None:
            return None
        else:
            return self.motion_quantizer.codebook_size

    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'DeepPhase':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = DeepPhase(cfg)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = DeepPhase.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=False, 
                **checkpoint_kwargs
            )
        return model


class DeepPhaseEncoder(MotionEncoder): 
    """
    An attempt to implement the DeepPhase model encoder . Not finished, not tested.
    """

    def __init__(self, cfg, sizes):
        super().__init__()
        self.config = cfg
        self.input_channels = self.config.input_dim
        self.embedding_channels = self.config.feature_dim
        self.time_range = sizes.sequence_length 
        self.window = sizes.sequence_length / sizes.fps

        self.tpi = nn.Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = nn.Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = nn.Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window, requires_grad=False) #Remove DC frequency

        intermediate_channels = int(self.input_channels/3)
        
        # padding = int((self.time_range - 1) / 2)
        # padding = int((self.time_range) / 2)
        padding = 'same'

        self.conv1 = nn.Conv1d(self.input_channels, intermediate_channels, self.time_range, stride=1, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.norm1 = LN_v2(self.time_range)
        self.conv2 = nn.Conv1d(intermediate_channels, self.embedding_channels, self.time_range, stride=1, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.fc = torch.nn.ModuleList()
        for i in range(self.embedding_channels):
            self.fc.append(nn.Linear(self.time_range, 2))


    def get_input_dim(self):
        return self.config.input_dim
        # return input_dim

    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        #Offset
        offset = rfft.real[:,:,0] / self.time_range #DC component

        return freq, amp, offset

    def forward(self, batch, input_key="input_sequence", output_key="encoded_features", **kwargs):
        # ## downsample into path-wise length seq before passing into transformer
        # dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        y = batch[input_key]
        
        # y must be in B, C, T
        y = y.permute(0, 2, 1)

        # #Signal Embedding
        # y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.conv1(y)
        y = self.norm1(y)
        y = nn.functional.elu(y)
        y = self.conv2(y)

        latent = y #Save latent for returning

        #Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2) # dim is time dimension

        #Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

        #Parameters    
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)

        batch["encoded_frequencies"] = f
        batch["encoded_amplitudes"] = a
        batch["encoded_offsets"] = b
        batch["encoded_phases"] = p
        batch["encoded_features"] = latent
        
        return batch

    def bottleneck_dim(self):
        return self.config.feature_dim
    
    def latent_temporal_factor(self): 
        return 2 ** self.sizes.quant_factor

    def quant_factor(self): 
        return self.sizes.quant_factor

    
class DeepPhaseDecoder(MotionDecoder): 

    def __init__(self, cfg, sizes, out_dim): 
        super().__init__()
        self.config = cfg
        self.sizes = sizes 
        self.out_dim = out_dim
        self.input_channels = out_dim
        self.embedding_channels = self.config.feature_dim
        self.time_range = sizes.sequence_length 
        # try:
        #     fps = self.config.data.get("fps", 25)
        # except AttributeError: 
        #     fps = 25
        #     print("[WARNING] fps not found in config, using default value of 25")
        self.window = sizes.sequence_length / sizes.fps
        assert self.time_range == self.window * sizes.fps, "Time range must be equal to window length * fps"
        # not trainable constant params
        self.tpi = nn.Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = nn.Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)

        intermediate_channels = int(self.input_channels/3)
        # padding = int((self.time_range - 1) / 2)
        # padding = int((self.time_range) / 2)
        padding = 'same'

        self.deconv1 = nn.Conv1d(self.embedding_channels, intermediate_channels, self.time_range, stride=1, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.denorm1 = LN_v2(self.time_range)
        self.deconv2 = nn.Conv1d(intermediate_channels, self.input_channels, self.time_range, stride=1, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')

    def forward(self, batch, input_key="quantized_features", output_key="decoded_sequence", **kwargs):
        f = batch["encoded_frequencies"]
        a = batch["encoded_amplitudes"]
        b = batch["encoded_offsets"]
        p = batch["encoded_phases"]

        #Latent Reconstruction 
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b
        batch["decoded_latent"] = y

        #Signal Reconstruction
        y = self.deconv1(y)
        y = self.denorm1(y)
        y = nn.functional.elu(y)

        y = self.deconv2(y)

        # y is in B, C, T, transpose to B, T, C
        y = y.permute(0, 2, 1)

        batch[output_key] = y
        return batch


class LN_v2(nn.Module):

    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y
