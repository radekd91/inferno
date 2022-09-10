from turtle import forward
import torch 
from torch import nn
import math
from gdl.models.rotation_loss import convert_rot


class AutoRegressiveDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, sample, train=False, teacher_forcing=True): 
        # teacher_forcing = not train
        hidden_states = sample["seq_encoder_output"]
        sample["hidden_feature"] = hidden_states # first "hidden state" is the audio feature
        if teacher_forcing:
            sample = self._teacher_forced_step(sample)
        else:
            num_frames = sample["gt_vertices"].shape[1] if "gt_vertices" in sample.keys() else hidden_states.shape[1]
            num_frames = min(num_frames, self._max_auto_regressive_steps())
            for i in range(num_frames):
                sample = self._autoregressive_step(sample, i)
        sample = self._post_prediction(sample)
        return sample

    def _max_auto_regressive_steps(self):
        raise NotImplementedError("")

    def _teacher_forced_step(self): 
        raise NotImplementedError("")

    def _autoregressive_step(self): 
        raise NotImplementedError("")

    def _post_prediction(self): 
        raise NotImplementedError("")


def positional_encoding_from_cfg(cfg): 
    if cfg.positional_encoding.type == 'PeriodicPositionalEncoding': 
        return PeriodicPositionalEncoding(cfg.feature_dim, **cfg.positional_encoding)
    elif cfg.positional_encoding.type == 'PositionalEncoding':
        return PositionalEncoding(cfg.feature_dim, **cfg.positional_encoding)
    elif not cfg.positional_encoding.type: 
        return None
    raise ValueError("Unsupported positional encoding")


class FaceFormerDecoderBase(AutoRegressiveDecoder):

    def __init__(self, cfg) -> None:
        super().__init__()
        # periodic positional encoding 
        # self.PPE = PeriodicPositionalEncoding(cfg.feature_dim, period = cfg.period)
        self.PE = positional_encoding_from_cfg(cfg)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = cfg.nhead, max_seq_len = cfg.max_len, period=cfg.period)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = cfg.feature_dim, 
            nhead=cfg.nhead, 
            dim_feedforward = cfg.feature_dim, batch_first=True)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_layers)
        self.vertice_map = nn.Linear(cfg.vertices_dim, cfg.feature_dim)
        self.obj_vector = nn.Linear(cfg.num_training_subjects, cfg.feature_dim, bias=False)

    def get_trainable_parameters(self):
        return list(self.parameters())

    def _max_auto_regressive_steps(self):
        return self.biased_mask.shape[1]

    def _autoregressive_step(self, sample, i):
        hidden_states = sample["hidden_feature"]
        if i==0:
            one_hot = sample["one_hot"]
            obj_embedding = self.obj_vector(one_hot)
            vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = vertice_emb
            sample["style_emb"] = style_emb
            vertices_input = self.PE(style_emb)
        else:
            vertice_emb = sample["embedded_output"]
            style_emb = sample["style_emb"]
            vertices_input = self.PE(vertice_emb)
        
        vertices_out = self._decode(sample, vertices_input, hidden_states)
        sample["predicted_vertices"] = vertices_out

        new_output = self.vertice_map(vertices_out[:,-1,:]).unsqueeze(1)
        new_output = new_output + style_emb
        vertice_emb = torch.cat((vertice_emb, new_output), 1)
        sample["embedded_output"] = vertice_emb
        return sample


    def _teacher_forced_step(self, sample): 
        vertices = sample["gt_vertices"]
        template = sample["template"].unsqueeze(1)
        hidden_states = sample["hidden_feature"]
        one_hot = sample["one_hot"]
        obj_embedding = self.obj_vector(one_hot)

        vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
        style_emb = vertice_emb  

        vertices_input = torch.cat((template, vertices[:,:-1]), 1) # shift one position
        vertices_input = vertices_input - template
        vertices_input = self.vertice_map(vertices_input)
        vertices_input = vertices_input + style_emb
        vertices_input = self.PE(vertices_input)

        vertices_out = self._decode(sample, vertices_input, hidden_states)
        sample["predicted_vertices"] = vertices_out

        return sample

    def _decode(self, sample, vertices_input, hidden_states):
        dev = vertices_input.device
        tgt_mask = self.biased_mask[:, :vertices_input.shape[1], :vertices_input.shape[1]].clone().detach().to(device=dev)
        memory_mask = enc_dec_mask(dev, vertices_input.shape[1], hidden_states.shape[1])
        transformer_out = self.transformer_decoder(vertices_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)

        vertices_out = self._decode_vertices(sample, transformer_out)
        return vertices_out

    def _post_prediction(self, sample):
        template = sample["template"]
        vertices_out = sample["predicted_vertices"]
        vertices_out = vertices_out + template
        sample["predicted_vertices"] = vertices_out
        return sample

    def _decode_vertices(self):
        raise NotImplementedError()


class PeriodicPositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600, op: str = 'add', **kwargs):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)        
        self.op = op
        
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pe = self.pe[:, :x.size(1), :]
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


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 600, op: str = 'add', **kwargs):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.op = op

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # pe = self.pe[:x.size(0), :]
        pe = self.pe[:, :x.size(1), :]
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


class FaceFormerDecoder(FaceFormerDecoderBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.vertex_map = nn.Linear(cfg.feature_dim, cfg.vertices_dim)

        # the init is done this way in the paper for some reason
        nn.init.constant_(self.vertex_map.weight, 0)
        nn.init.constant_(self.vertex_map.bias, 0)

    def _decode_vertices(self, sample, transformer_out): 
        vertice_out = self.vertex_map(transformer_out)
        return vertice_out


def rotation_rep_size(rep):
    if rep in ["quat", "quaternion"]:
        return 4
    elif rep in ["aa", "axis-angle"]:
        return 3
    elif rep in ["matrix", "mat"]:
        return 9
    elif rep in ["euler", "euler-angles"]:
        return 3
    elif rep in ["rot6d", "rot6dof", "6d"]:
        return 6
    else:
        raise ValueError("Unknown rotation representation: {}".format(rep))


class FlameFormerDecoder(FaceFormerDecoderBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        from gdl.models.DecaFLAME import FLAME
        # from munch import Munch
        # self.flame_config = Munch()
        # self.flame_config.flame_model_path = "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl" 
        # self.flame_config.n_shape = 100 
        # self.flame_config.n_exp = 50
        # self.flame_config.flame_lmk_embedding_path = "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy"
        self.flame_config = cfg.flame
        self.flame = FLAME(self.flame_config)
        pred_dim = 0
        self.predict_exp = cfg.predict_exp
        self.predict_jaw = cfg.predict_jaw
        self.rotation_representation = cfg.get("rotation_representation", "aa")
        self.flame_rotation_rep = "aa"
        if self.predict_exp: 
            pred_dim += self.flame_config.n_exp
        if self.predict_jaw:
            pred_dim += rotation_rep_size(self.rotation_representation)

        self.post_transformer = nn.Linear(cfg.feature_dim, pred_dim)
        self.flame_space_loss = cfg.flame_space_loss
        # self.rotation_loss_space = cfg.rotation_loss_space


    def _rotation_representation(self):
        return self.rotation_representation


    def _decode_vertices(self, sample, transformer_out): 
        template = sample["template"]
        # jaw = sample["gt_jaw"]
        # exp = sample["exp"]

        self.flame.v_template = template.squeeze(1).view(-1, 3)
        transformer_out = self.post_transformer(transformer_out)
        batch_size = transformer_out.shape[0]
        T_size = transformer_out.shape[1]
        pose_params = self.flame.eye_pose.expand(batch_size*T_size, -1)

        vector_idx = 0
        if self.predict_jaw:
            rotation_rep_size_ = rotation_rep_size(self.rotation_representation)
            jaw_pose = transformer_out[..., :rotation_rep_size_].view(batch_size*T_size, -1)
            vector_idx += rotation_rep_size_
            if self.rotation_representation in ['mat', 'matrix']:
                jaw_pose = jaw_pose.view(-1, 3, 3) 
                U, S, Vh = torch.linalg.svd(jaw_pose) # do SVD to ensure orthogonality
                jaw_pose = U.contiguous()
            
        else: 
            jaw = sample["gt_jaw"][:, :T_size, ...]
            jaw_pose = jaw.view(batch_size*T_size, -1)
        if self.predict_exp: 
            expression_params = transformer_out[..., vector_idx:].view(batch_size*T_size, -1)
            vector_idx += self.flame_config.n_exp
        else: 
            exp = sample["gt_exp"][:, :T_size, ...]
            expression_params = exp.view(batch_size*T_size, -1)
        
        sample["predicted_exp"] = expression_params.view(batch_size,T_size, -1)
        sample["predicted_jaw"] = jaw_pose.view(batch_size,T_size, -1)
        
        assert vector_idx == transformer_out.shape[-1]

        flame_jaw_pose = convert_rot(jaw_pose, self.rotation_representation, self.flame_rotation_rep)
        pose_params = torch.cat([ pose_params[..., :3], flame_jaw_pose], dim=-1)
        shape_params = torch.zeros((batch_size*T_size, self.flame_config.n_shape), device=transformer_out.device)
        with torch.no_grad():
            vertice_neutral, _, _ = self.flame.forward(shape_params[0:1, ...], torch.zeros_like(expression_params[0:1, ...])) # compute neutral shape
        vertice_out, _, _ = self.flame(shape_params, expression_params, pose_params)
        vertice_out = vertice_out - vertice_neutral # compute the offset that is then added to the template shape
        vertice_out = vertice_out.view(batch_size, T_size, -1)
        return vertice_out


# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Alignment Bias
def enc_dec_mask(device, T, S, dataset="vocaset"):
    mask = torch.ones(T, S)
    smaller_dim = min(T, S)
    # smaller_dim = T
    if dataset == "BIWI":
        for i in range(smaller_dim):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(smaller_dim):
            mask[i, i] = 0
    else:
        raise NotImplementedError("Unknown dataset")
    return (mask==1).to(device=device)


class FeedForwardDecoder(nn.Module): 

    def __init__(self, cfg) -> None:
        super().__init__()
        self.obj_vector = nn.Linear(cfg.num_training_subjects, cfg.feature_dim, bias=False)
        self.style_op = cfg.get('style_op', 'add')
        self.PE = positional_encoding_from_cfg(cfg)

    def forward(self, sample, train=False, teacher_forcing=True): 
        one_hot = sample["one_hot"]
        obj_embedding = self.obj_vector(one_hot)
        style_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
        sample["hidden_feature"] = sample["seq_encoder_output"]
        hidden_states = self._positional_enc(sample)

        styled_hidden_states = self._style(sample, hidden_states, style_emb)

        decoded_offsets = self._decode(sample, styled_hidden_states)

        sample["predicted_vertices"] = decoded_offsets
        sample = self._post_prediction(sample)
        return sample

    def _positional_enc(self, sample): 
        hidden_states = sample["hidden_feature"] 
        if self.PE is not None:
            hidden_states = self.PE(hidden_states)
        return hidden_states

    def _style_dim_factor(self): 
        if self.style_op == "add":
            return 1
        elif self.style_op == "cat":
            return 2
        raise ValueError(f"Invalid operation: '{self.style_op}'")
       
    def _pe_dim_factor(self):
        dim_factor = 1
        if self.PE is not None: 
            dim_factor = self.PE.output_size_factor()
        return dim_factor

    def _total_dim_factor(self): 
        return self._style_dim_factor() * self._pe_dim_factor()

    def _style(self, sample, hidden_states, style_emb): 
        if self.style_op == "add":
            styled_hidden_states = hidden_states + style_emb
        elif self.style_op == "cat":
            styled_hidden_states = torch.cat([hidden_states, style_emb], dim=-1)
        else: 
            raise ValueError(f"Invalid operation: '{self.style_op}'")
        
        return styled_hidden_states

    def _decode(self, sample, hidden_states) :
        """
        Hiddent states are the ouptut of the encoder
        Returns a tensor of vertex offsets.
        """
        raise NotImplementedError("The subdclass must implement this")

    def _post_prediction(self, sample):
        template = sample["template"]
        vertices_out = sample["predicted_vertices"]
        vertices_out = vertices_out + template
        sample["predicted_vertices"] = vertices_out
        return sample

    def get_trainable_parameters(self): 
        return [p for p in self.parameters() if p.requires_grad]


class LinearDecoder(FeedForwardDecoder):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        dim_factor = self._total_dim_factor()
        self.decoder = nn.Linear(dim_factor*cfg.feature_dim, cfg.vertices_dim)

    def _decode(self, sample, styled_hidden_states) :
        B, T = styled_hidden_states.shape[:2]
        styled_hidden_states = styled_hidden_states.view(B*T, -1)
        decoded_offsets = self.decoder(styled_hidden_states)
        decoded_offsets = decoded_offsets.view(B, T, -1)
        return decoded_offsets


class BertDecoder(FeedForwardDecoder):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        dim_factor = self._total_dim_factor()
        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.feature_dim * dim_factor, 
                    nhead=cfg.nhead, dim_feedforward=dim_factor*cfg.feature_dim, 
                    activation=cfg.activation,
                    dropout=cfg.dropout, batch_first=True
        )        
        self.bert_decoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.decoder = nn.Linear(dim_factor*cfg.feature_dim, cfg.vertices_dim)

    def _decode(self, sample, styled_hidden_states) :
        decoded_offsets = self.bert_decoder(styled_hidden_states)
        B, T = decoded_offsets.shape[:2]
        decoded_offsets = decoded_offsets.view(B*T, -1)
        decoded_offsets = self.decoder(styled_hidden_states)
        decoded_offsets = decoded_offsets.view(B, T, -1)
        return decoded_offsets


class LinearAutoRegDecoder(AutoRegressiveDecoder):

    def __init__(self, cfg):
        super().__init__()
        # self.flame_config = cfg.flame
        self.decoder = nn.Linear(2*cfg.feature_dim, cfg.vertices_dim)
        self.obj_vector = nn.Linear(cfg.num_training_subjects, cfg.feature_dim, bias=False)
        self.vertice_map = nn.Linear(cfg.vertices_dim, cfg.feature_dim)
        self.PE = None

    def get_trainable_parameters(self): 
        return [p for p in self.parameters() if p.requires_grad]

    def _max_auto_regressive_steps(self):
        return 2000

    def _autoregressive_step(self, sample, i):
        hidden_states = sample["hidden_feature"]
        if i==0:
            one_hot = sample["one_hot"]
            obj_embedding = self.obj_vector(one_hot)
            vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = vertice_emb
            sample["style_emb"] = style_emb
            # vertices_input = self.PE(style_emb)
            vertices_input = style_emb
        else:
            vertice_emb = sample["embedded_output"]
            style_emb = sample["style_emb"]
            vertices_input = vertice_emb
        if self.PE is not None:
            vertices_input = self.PE(vertices_input)
        vertices_out = self._decode(sample, vertices_input, hidden_states)
        sample["predicted_vertices"] = vertices_out

        new_output = self.vertice_map(vertices_out[:,-1,:]).unsqueeze(1)
        new_output = new_output + style_emb
        vertice_emb = torch.cat((vertice_emb, new_output), 1)
        sample["embedded_output"] = vertice_emb
        return sample

    def _teacher_forced_step(self, sample): 
        vertices = sample["gt_vertices"]
        template = sample["template"].unsqueeze(1)
        hidden_states = sample["hidden_feature"]
        one_hot = sample["one_hot"]
        obj_embedding = self.obj_vector(one_hot)

        vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
        style_emb = vertice_emb  

        vertices_input = torch.cat((template, vertices[:,:-1]), 1) # shift one position
        vertices_input = vertices_input - template
        vertices_input = self.vertice_map(vertices_input) # map to feature dim
        vertices_input = vertices_input + style_emb # add style embedding
        if self.PE is not None:
            vertices_input = self.PE(vertices_input) # add positional encoding

        vertices_out = self._decode(sample, vertices_input, hidden_states)
        sample["predicted_vertices"] = vertices_out

        return sample

    def _decode(self, sample, vertices_input, hidden_states):
        # dev = vertices_input.device
        # tgt_mask = self.biased_mask[:, :vertices_input.shape[1], :vertices_input.shape[1]].clone().detach().to(device=dev)
        # memory_mask = enc_dec_mask(dev, vertices_input.shape[1], hidden_states.shape[1])
        # transformer_out = self.transformer_decoder(vertices_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        #TODO: we can also consider feeding the hidden states of a large perceptive field (further back in the past/future) to the decoder
        decoder_input = torch.cat([vertices_input, hidden_states[:,:vertices_input.shape[1], ...]], dim=-1)
        B, T = decoder_input.shape[:2]
        vertices_out = self.decoder(decoder_input)
        vertices_out = vertices_out.view(B, T, -1)
        return vertices_out

    def _post_prediction(self, sample):
        template = sample["template"]
        vertices_out = sample["predicted_vertices"]
        vertices_out = vertices_out + template
        sample["predicted_vertices"] = vertices_out
        return sample

    def _decode_vertices(self, sample, vertices_input):
        return self.decoder(vertices_input)