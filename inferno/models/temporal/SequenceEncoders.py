from inferno.models.temporal.Bases import SequenceEncoder 
import torch 
from torch import Tensor
import math
from omegaconf import OmegaConf
import sys


class ParallelSequenceEncoder(SequenceEncoder): 
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        encoders = []
        self.outputs = []
        for encoder_cfg in cfg.encoders:
            encoder = self.get_encoder(encoder_cfg)
            encoders.append(encoder)
            self.outputs += [encoder_cfg.outputs]

    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        return self.cfg.feature_dim

    def output_feature_dim(self):
        return self.cfg.feature_dim

    def output_feature_dim(self): 
        return self.cfg.d_model * self.output_size_factor()


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, op: str = 'add', **kwargs):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.op = op

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pe = self.pe[:x.size(0), :]
        if self.op in ['add', 'sum']:
            x = x + pe
        elif self.op in ['concat', 'cat', 'concatenate']:
            x = torch.cat([x, pe.repeat(1,x.shape[1],1)], dim=2)
        else: 
            raise ValueError('how must be either add or concat')
        return self.dropout(x)

    def output_size_factor(self): 
        if self.op in ['add', 'sum']:
            return 1
        elif self.op in ['concat', 'cat', 'concatenate']:
            return 2
        else:
            raise ValueError('how must be either add or concat')


class DummyPositionalEncoding(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x

    def output_size_factor(self): 
        return 1 


def pos_enc_from_cfg(cfg): 
    if cfg.type == "none":
        return DummyPositionalEncoding
    if cfg.type == "PositionalEncoding":
        return PositionalEncoding
    if cfg.type == "PeriodicPositionalEncoding":
        from inferno.models.talkinghead.FaceFormerDecoder import PeriodicPositionalEncoding
        return PeriodicPositionalEncoding
    raise ValueError("Unknown positional encoding type: {}".format(cfg.type))


class SimpleTransformerSequenceEncoder(SequenceEncoder): 

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        pe_type = pos_enc_from_cfg(cfg.positional_encoding)
        pe_kwargs = OmegaConf.to_container(cfg.positional_encoding)
        del pe_kwargs["type"]
        # if pe_type is not None:
        # self.pos_encoder = pe_type(d_model=cfg.feature_dim, dropout=cfg.dropout, **pe_kwargs)
        self.pos_encoder = pe_type(d_model=cfg.feature_dim, **pe_kwargs)

        self.input_feature_dim_ = self.cfg.get('input_feature_dim', None) or self.cfg.feature_dim 
        if self.input_feature_dim_ is not None and self.input_feature_dim() != self.cfg.feature_dim:
            self.input_projection = torch.nn.Linear(self.input_feature_dim_, self.cfg.feature_dim)
        else:
            self.input_projection = None

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=cfg.feature_dim * self.pos_encoder.output_size_factor(), 
                    nhead=cfg.nhead, dim_feedforward=cfg.feature_dim, activation=cfg.activation,
                    dropout=cfg.dropout, batch_first=True, )        
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

    def forward(self, sample, input_key=None, output_key=None, training=False, validation=False):
        input_key = input_key or self.cfg.input_key or 'fused_feature'
        output_key = output_key or self.cfg.output_key or "seq_encoder_output"
        feat = sample[input_key] 
        if self.input_projection is not None:
            feat = self.input_projection(feat)
        if self.pos_encoder is not None:
            feat = self.pos_encoder(feat.transpose(1, 0)).transpose(0,1)
        out = self.transformer_encoder.forward(feat) 
        sample[output_key] = out 
        return sample

    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        return self.input_feature_dim_ or self.cfg.feature_dim

    def output_feature_dim(self):
        return self.cfg.feature_dim * self.pos_encoder.output_size_factor()


class MLPSequenceEncoder(SequenceEncoder): 

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        layers = []
        for i in range(self.cfg.num_layers-1): 
            layers += [torch.nn.Linear(self.cfg.feature_dim, self.cfg.feature_dim)]
            layers += [torch.nn.Dropout(p=self.cfg.dropout)]
            layers += [torch.nn.ReLU()]
        layers += [torch.nn.Linear(cfg.feature_dim, cfg.feature_dim)]

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, sample):
        feat = sample["fused_feature"] 
        # B, T, D -> B * T, D 
        out = feat.view(-1, feat.shape[-1])
        out = self.layers(feat) 
        # B * T, D -> B, T, D
        out = out.view(feat.shape[0], feat.shape[1], -1)
        sample["seq_encoder_output"] = out 
        return sample

    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        return self.cfg.feature_dim

    def output_feature_dim(self):
        return self.cfg.feature_dim


class LinearSequenceEncoder(SequenceEncoder): 

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        input_feature_dim = self.cfg.get('input_feature_dim', None) or self.cfg.feature_dim 
        output_feature_dim = self.cfg.feature_dim
        self.linear = torch.nn.Linear(input_feature_dim, output_feature_dim)

    def forward(self, sample, input_key="fused_feature"):
        feat = sample[input_key] 
        # B, T, D -> B * T, D 
        out = feat.view(-1, feat.shape[-1])
        out = self.linear(feat) 
        # B * T, D -> B, T, D
        out = out.view(feat.shape[0], feat.shape[1], -1)
        sample["seq_encoder_output"] = out 
        return sample

    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        return self.cfg.feature_dim

    def output_feature_dim(self):
        return self.cfg.feature_dim


class GRUSeqEnc(SequenceEncoder):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.feature_dim
        self.num_layers = cfg.num_layers
        self.bidirectional = cfg.bidirectional
        self.dropout = cfg.dropout
        self.gru = torch.nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, 
            bidirectional=self.bidirectional, dropout=self.dropout, batch_first=True)

    def forward(self, sample):
        feat = sample["fused_feature"] 
        out, last_hidden = self.gru(feat) 
        sample["seq_encoder_output"] = out 
        return sample

    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        return self.cfg.feature_dim

    def output_feature_dim(self):
        return self.cfg.feature_dim * 2 if self.bidirectional else self.cfg.feature_dim


class TemporalConvNet(SequenceEncoder):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        layers = []
        kernel_size = cfg.kernel_size
        
        for i in range(self.cfg.num_layers):
            layers += [torch.nn.Conv1d(self.cfg.feature_dim, 
                    self.cfg.feature_dim, kernel_size=kernel_size, padding='same',)]
            layers += [torch.nn.ReLU()]
            if self.cfg.dropout > 0:
                layers += [torch.nn.Dropout(p=self.cfg.dropout)] 
            if self.cfg.norm_layer: 
                if self.cfg.norm_layer == "batchnorm1d":
                    layers += [torch.nn.BatchNorm1d(self.cfg.feature_dim)]
                elif self.cfg.norm_layer == "layernorm":
                    layers += [torch.nn.LayerNorm(self.cfg.feature_dim)]
                elif self.cfg.norm_layer == "instancenorm1d":
                    layers += [torch.nn.InstanceNorm1d(self.cfg.feature_dim)]
                else:
                    raise NotImplementedError("Unknown norm layer: {}".format(self.cfg.norm_layer))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, sample):
        feat = sample["fused_feature"].transpose(1, 2)
        out = self.layers(feat).transpose(1, 2) 
        sample["seq_encoder_output"] = out 
        return sample 
        
    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        return self.cfg.feature_dim

    def output_feature_dim(self):
        return self.cfg.feature_dim 


class ResNetBottleneck(torch.nn.Module):
    # expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(planes)
        self.conv2 = torch.nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(planes)
        self.conv3 = torch.nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm1d(planes * 4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class TemporalResNet(SequenceEncoder):

    def __init__(self, cfg, block):
        super().__init__()
        self.cfg = cfg
        layers = cfg.layers
        self.conv1 = torch.nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, sample): 
        return sample

