from .MotionPrior import * 
from .VectorQuantizer import VectorQuantizer
from torch import nn


class L2lVqVae(MotionPrior):

    def __init__(self, cfg) -> None:
        # motion_encoder = motion_decoder_from_cfg(cfg)
        # motion_decoder = motion_decoder_from_cfg(cfg)
        # motion_codebook = motion_codebook_from_cfg(cfg)
        motion_encoder = L2lEncoder(cfg.sequence_encoder, cfg.settings)
        motion_decoder = L2lDecoder(cfg.sequence_decoder, cfg.settings)
        motion_quantizer = VectorQuantizer(cfg.quantizer)
        super().__init__(motion_encoder, motion_decoder, motion_quantizer)

    def _compute_loss(self, sample, training, validation, loss_name, loss_cfg):
        if loss_name == 'reconstruction':
            return sample['reconstruction']
        
        elif loss_name == 'codebook_alignment':
            return sample['codebook_alignment']
            
        elif loss_name == 'codebook_commitment':
            return sample['codebook_commitment']
        
        elif loss_name == 'perplexity':
            return sample['perplexity']
        
        else:
            raise NotImplementedError(f"Loss '{loss_name}' not implemented")
        

class LearnedPositionEmbedding(nn.Module):
    """Learned Postional Embedding Layer"""

    def __init__(self, seq_length, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(seq_length, dim))

    def forward(self, x):
        return x + self.pos_embedding


class L2lEncoder(MotionEncoder): 
    """
    Inspired by by the encoder from Learning to Listen.
    """

    def __init__(self, cfg, settings): 
        super().__init__()
        self.config = cfg
        # size=self.config['transformer_config']['in_dim']
        size = self.config.input_dim
        # dim=self.config['transformer_config']['hidden_size']
        dim = self.config.hidden_size
        layers = [nn.Sequential(
                    nn.Conv1d(size,dim,5,stride=2,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(dim))]
        for _ in range(1, cfg.quant_factor):
            layers += [nn.Sequential(
                        nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(0.2, True),
                        nn.BatchNorm1d(dim),
                        nn.MaxPool1d(2)
                        )]
        self.squasher = nn.Sequential(*layers) 
        # the purpose of the squasher is to reduce the FPS of the input sequence

        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.feature_dim, 
                    nhead=cfg.nhead, 
                    dim_feedforward=cfg.intermediate_size, 
                    activation=cfg.activation, # L2L paper uses gelu
                    dropout=cfg.dropout, 
                    batch_first=True
        )
        self.encoder_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # self.encoder_transformer = Transformer(
        #     in_size=self.config['transformer_config']['hidden_size'],
        #     hidden_size=self.config['transformer_config']['hidden_size'],
        #     num_hidden_layers=\
        #             self.config['transformer_config']['num_hidden_layers'],
        #     num_attention_heads=\
        #             self.config['transformer_config']['num_attention_heads'],
        #     intermediate_size=\
        #             self.config['transformer_config']['intermediate_size'])
        self.encoder_pos_embedding = LearnedPositionEmbedding(
            settings.quant_sequence_length,
            self.config.hidden_size
            )
        self.encoder_linear_embedding = nn.Linear(
            self.config.hidden_size,
            self.config.hidden_size
        )

    def get_input_dim(self):
        return self.config.input_dim

    def forward(self, batch, input_key="input_sequence", output_key="encoded_features", **kwargs):
        # ## downsample into path-wise length seq before passing into transformer
        # dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        inputs = batch[input_key]
        # input is batch first: B, T, D but the convolution expects B, D, T
        # so we need to permute back and forth
        inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1)
        
        encoded_features = self.encoder_linear_embedding(inputs)
        encoded_features = self.encoder_pos_embedding(encoded_features)
        encoded_features = self.encoder_transformer(encoded_features, mask=None)
        batch[output_key] = encoded_features
        return batch



    # def forward(self, inputs):
    #     ## downsample into path-wise length seq before passing into transformer
    #     dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    #     inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1)
        
    #     encoder_features = self.encoder_linear_embedding(inputs)
    #     encoder_features = self.encoder_pos_embedding(encoder_features)
    #     encoder_features = self.encoder_transformer((encoder_features, dummy_mask))
    #     return encoder_features



class L2lDecoder(MotionEncoder): 

    def __init__(self, cfg, settings, out_dim): 
        super().__init__()
        is_audio = False
        self.cfg = cfg
        size = self.cfg.hidden_size
        dim = self.cfg.hidden_size
        self.expander = nn.ModuleList()
        self.expander.append(nn.Sequential(
                    nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                        output_padding=1,
                                        padding_mode='replicate'),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(dim)))
        num_layers = settings.quant_factor + 2 \
            if is_audio else settings.quant_factor
        # TODO: check if we need to keep the sequence length fixed
        seq_len = settings.sequence_length*4 \
            if is_audio else settings.sequence_length
        for _ in range(1, num_layers):
            self.expander.append(nn.Sequential(
                                nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                        padding_mode='replicate'),
                                nn.LeakyReLU(0.2, True),
                                nn.BatchNorm1d(dim),
                                ))
        decoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.feature_dim, 
                    nhead=cfg.nhead, 
                    dim_feedforward=cfg.intermediate_size, 
                    activation=cfg.activation, # L2L paper uses gelu
                    dropout=cfg.dropout, 
                    batch_first=True
        )
        self.decoder_transformer = torch.nn.TransformerEncoder(decoder_layer, num_layers=cfg.num_layers)
        self.decoder_pos_embedding = LearnedPositionEmbedding(
            seq_len,
            self.cfg.hidden_size
            )
        self.decoder_linear_embedding = nn.Linear(
            self.cfg.hidden_size,
            self.cfg.hidden_size
            )
        self.cross_smooth_layer = nn.Conv1d(
                cfg.hidden_size,
                out_dim, 5, 
                padding=2
            )

    def forward(self, batch, input_key="encoded_features", output_key="decoded_sequence", **kwargs):
        # dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        ## upsample to the original length of the sequence before passing into transformer
        inputs = batch[input_key]
        for i, module in enumerate(self.expander):
            # input is batch first: B, T, D but the convolution expects B, D, T
            # so we need to permute back and forth
            inputs = module(inputs.permute(0,2,1)).permute(0,2,1)
            if i > 0:
                inputs = inputs.repeat_interleave(2, dim=1)
        
        decoded_features = self.decoder_linear_embedding(inputs)
        decoded_features = self.decoder_pos_embedding(decoded_features)
        decoded_features = self.decoder_transformer(decoded_features, mask=None)
        decoded_reconstruction = self.cross_smooth_layer(decoded_features.permute(0,2,1)).permute(0,2,1)
        batch[output_key] = decoded_reconstruction
        return batch
