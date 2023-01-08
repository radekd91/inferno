import pytorch_lightning as pl 
from typing import Any, Optional, Dict
from gdl.models.temporal.Bases import TemporalFeatureEncoder, SequenceClassificationEncoder, Preprocessor, ClassificationHead
from gdl.models.talkinghead.FaceFormerDecoder import PositionalEncoding, init_biased_mask_future, init_biased_mask, init_mask
import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding_from_cfg(cfg, feature_dim): 
    # if cfg.positional_encoding.type == 'PeriodicPositionalEncoding': 
    #     return PeriodicPositionalEncoding(cfg.feature_dim, **cfg.positional_encoding)
    # el
    if cfg.positional_encoding.type == 'PositionalEncoding':
        return PositionalEncoding(feature_dim, **cfg.positional_encoding)
        # return PositionalEncoding(cfg.feature_dim, **cfg.positional_encoding)
    elif not cfg.positional_encoding.type or str(cfg.positional_encoding.type).lower() == 'none':
        return None
    raise ValueError("Unsupported positional encoding")


class TransformerPooler(nn.Module):
    """
    inspired by: 
    https://huggingface.co/transformers/v3.3.1/_modules/transformers/modeling_bert.html#BertPooler 
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, sample, input_key = "encoded_sequence_feature"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = sample[input_key]
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        sample["pooled_sequence_feature"] = pooled_output
        return sample


class TransformerEncoder(torch.nn.Module):

    def __init__(self, cfg, input_dim) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = input_dim
        if self.input_dim == cfg.feature_dim:
            self.bottleneck = None
        else:
            self.bottleneck = nn.Linear(self.input_dim, cfg.feature_dim)
        self.PE = positional_encoding_from_cfg(cfg, cfg.feature_dim)
        dim_factor = self._total_dim_factor()
        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.feature_dim * dim_factor, 
                    nhead=cfg.nhead, 
                    dim_feedforward=dim_factor*cfg.feature_dim, 
                    activation=cfg.activation,
                    dropout=cfg.dropout, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        # self.decoder = nn.Linear(dim_factor*self.input_dim, self.decoder_output_dim())
        
        self.temporal_bias_type = cfg.get('temporal_bias_type', 'none')
        if self.temporal_bias_type == 'faceformer':
            self.biased_mask = init_biased_mask(num_heads = cfg.nhead, max_seq_len = cfg.max_len, period=cfg.period)
        elif self.temporal_bias_type == 'faceformer_future':
            self.biased_mask = init_biased_mask_future(num_heads = cfg.nhead, max_seq_len = cfg.max_len, period=cfg.period)
        elif self.temporal_bias_type == 'classic':
            self.biased_mask = init_mask(num_heads = cfg.nhead, max_seq_len = cfg.max_len)
        elif self.temporal_bias_type == 'classic_future':
            self.biased_mask = init_mask(num_heads = cfg.nhead, max_seq_len = cfg.max_len)
        elif self.temporal_bias_type == 'none':
            self.biased_mask = None
        else:
            raise ValueError(f"Unsupported temporal bias type '{self.temporal_bias_type}'")

    def encoder_input_dim(self):
        return self.input_dim

    def encoder_output_dim(self):
        return self.cfg.feature_dim

    def forward(self, sample, train=False, teacher_forcing=True): 
        if self.bottleneck is not None:
            sample["hidden_feature"] = self.bottleneck(sample["hidden_feature"])
        hidden_states = self._positional_enc(sample)
        encoded_feature = self._encode(sample, hidden_states)
        sample["encoded_sequence_feature"] = encoded_feature
        return sample
       
    def _pe_dim_factor(self):
        dim_factor = 1
        if self.PE is not None: 
            dim_factor = self.PE.output_size_factor()
        return dim_factor

    def _total_dim_factor(self): 
        return  self._pe_dim_factor()

    def _positional_enc(self, sample): 
        hidden_states = sample["hidden_feature"] 
        if self.PE is not None:
            hidden_states = self.PE(hidden_states)
        return hidden_states

    def _encode(self, sample, hidden_states):
        if self.biased_mask is not None: 
            mask = self.biased_mask[:, :hidden_states.shape[1], :hidden_states.shape[1]].clone() \
                .detach().to(device=hidden_states.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(hidden_states.shape[0], 1, 1)
        else: 
            mask = None
        encoded_feature = self.encoder(hidden_states, mask=mask)
        B, T = encoded_feature.shape[:2]
        encoded_feature = encoded_feature.view(B*T, -1)
        encoded_feature = self.encoder(hidden_states)
        encoded_feature = encoded_feature.view(B, T, -1)
        return encoded_feature


def transformer_encoder_from_cfg(cfg, input_dim):
    if cfg.type == 'TransformerEncoder':
        return TransformerEncoder(cfg, input_dim)
    raise ValueError(f"Unsupported encoder type '{cfg.type}'")

def pooler_from_cfg(cfg):
    if cfg.type == 'TransformerEncoder':
        return TransformerEncoder(cfg)
    raise ValueError(f"Unsupported encoder type '{cfg.type}'")





class TransformerSequenceClassifier(SequenceClassificationEncoder):

    def __init__(self, cfg, input_dim, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        # self.num_classes = num_classes

        self.transformer_encoder = transformer_encoder_from_cfg(cfg.encoder, input_dim)
        self.pooler = TransformerPooler(self.transformer_encoder.encoder_output_dim(), self.transformer_encoder.encoder_output_dim())
        # self.classifier = nn.Linear(self.transformer_encoder.encoder_output_dim(), self.num_classes)

    def encoder_output_dim(self):
        return self.transformer_encoder.encoder_output_dim()

    def forward(self, sample):
        sample = self.transformer_encoder(sample)
        sample = self.pooler(sample)
        # sample = self.classifier(sample)
        return sample

    def get_trainable_parameters(self):
        return list(self.parameters())
    

class LinearClassificationHead(ClassificationHead): 
    def __init__(self, cfg, input_dim,  num_classes):
        super().__init__()
        self.cfg = cfg
        # self.input_dim = cfg.input_dim
        self.input_dim = input_dim
        # self.num_classes = cfg.num_classes
        self.num_classes = num_classes

        self.dropout = nn.Dropout(cfg.dropout_prob)
        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, sample, input_key="pooled_sequence_feature", output_key="predicted_logits"):
        sample[output_key] = self.classifier(sample[input_key])
        return sample

    def get_trainable_parameters(self):
        return list(self.parameters())
    

class VideoClassifierBase(pl.LightningModule): 

    def __init__(self, 
                 cfg, 
                 preprocessor: Optional[TemporalFeatureEncoder] = None,
                 feature_model: Optional[TemporalFeatureEncoder] = None,
                 sequence_encoder: Optional[SequenceClassificationEncoder] = None,
                 classification_head: Optional[ClassificationHead] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.feature_model = feature_model
        self.sequence_encoder = sequence_encoder
        self.classification_head = classification_head

    def get_trainable_parameters(self):
        trainable_params = []
        if self.feature_model is not None:
            trainable_params += self.feature_model.get_trainable_parameters()
        if self.sequence_encoder is not None:
            trainable_params += self.sequence_encoder.get_trainable_parameters()
        if self.classification_head is not None:
            trainable_params += self.classification_head.get_trainable_parameters()
        return trainable_params

    @property
    def max_seq_length(self):
        return 5000

    def configure_optimizers(self):
        trainable_params = []
        trainable_params += list(self.get_trainable_parameters())

        if trainable_params is None or len(trainable_params) == 0:
            print("[WARNING] No trainable parameters found.")
            return 

        if self.cfg.learning.optimizer == 'Adam':
            opt = torch.optim.Adam(
                trainable_params,
                lr=self.cfg.learning.learning_rate,
                amsgrad=False)
        elif self.cfg.learning.optimizer == 'SGD':
            opt = torch.optim.SGD(
                trainable_params,
                lr=self.cfg.learning.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: '{self.cfg.learning.optimizer}'")

        optimizers = [opt]
        schedulers = []

        opt_dict = {}
        opt_dict['optimizer'] = opt
        if 'learning_rate_patience' in self.cfg.learning.keys():
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                                   patience=self.cfg.learning.learning_rate_patience,
                                                                   factor=self.cfg.learning.learning_rate_decay,
                                                                   mode=self.cfg.learning.lr_sched_mode)
            schedulers += [scheduler]
            opt_dict['lr_scheduler'] = scheduler
            opt_dict['monitor'] = 'val_loss_total'
        elif 'learning_rate_decay' in self.cfg.learning.keys():
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.cfg.learning.learning_rate_decay)
            opt_dict['lr_scheduler'] = scheduler
            schedulers += [scheduler]
        return opt_dict

    @torch.no_grad()
    def preprocess_input(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        if self.preprocessor is not None:
            if self.device != self.preprocessor.device:
                self.preprocessor.to(self.device)
            sample = self.preprocessor(sample, input_key="video", train=train, test_time=not train, **kwargs)
        # sample = detach_dict(sample)
        return sample 

    def forward(self, sample: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        """
        sample: Dict[str, torch.Tensor]
            - audio: (B, T, F)
            # - masked_audio: (B, T, F)
        """
        input_key = "gt_emo_feature"
        T = sample[input_key].shape[1]
        if self.max_seq_length < T: # truncate
            print("[WARNING] Truncating audio sequence from {} to {}".format(T, self.max_seq_length))
            sample = truncate_sequence_batch(sample, self.max_seq_length)

        # preprocess input (for instance get 3D pseudo-GT )
        sample = self.preprocess_input(sample, train=train, **kwargs)
        check_nan(sample)

        if self.feature_model is not None:
            sample = self.feature_model(sample, train=train, **kwargs)
            check_nan(sample)
        else: 
            sample["hidden_feature"] = sample[input_key]

        if self.sequence_encoder is not None:
            sample = self.sequence_encoder(sample) #, train=train, validation=validation, **kwargs)
            check_nan(sample)

        if self.classification_head is not None:
            sample = self.classification_head(sample)
            check_nan(sample)

        return sample

    def compute_loss(self, sample, training, validation): 
        """
        Compute the loss for the given sample. 
        """
        losses = {}
        metrics = {}

        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            assert loss_name not in losses.keys()
            losses["loss_" + loss_name] = self._compute_loss(sample, loss_name, loss_cfg)
            # losses["loss_" + loss_name] = self._compute_loss(sample, training, validation, loss_name, loss_cfg)

        for metric_name, metric_cfg in self.cfg.learning.metrics.items():
            assert metric_name not in metrics.keys()
            with torch.no_grad():
                metrics["metric_" + metric_name] = self._compute_loss(sample, metric_name, metric_cfg)
                # metrics["metric_" + metric_name] = self._compute_loss(sample, training, validation, metric_name, metric_cfg)

        total_loss = None
        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            term = losses["loss_" + loss_name] 
            if term is not None:
                if isinstance(term, torch.Tensor) and term.isnan().any():
                    print(f"[WARNING]: loss '{loss_name}' is NaN. Skipping this term.")
                    continue
                if total_loss is None: 
                    total_loss = 0.
                weighted_term =  (term * loss_cfg["weight"])
                total_loss = total_loss + weighted_term
                losses["loss_" + loss_name + "_w"] = weighted_term

        losses["loss_total"] = total_loss
        return total_loss, losses, metrics

        # def _compute_loss(self, sample, training, validation, loss_name, loss_cfg): 
        #     raise NotImplementedError("Please implement this method in your child class")

    def _compute_loss(self, sample, loss_name, loss_cfg): 
        # TODO: this could be done nicer (have a dict with name - loss functor)
        loss_type = loss_name if 'loss_type' not in loss_cfg.keys() else loss_cfg['loss_type']

        if loss_type == "cross_entropy":
            loss_value = F.cross_entropy(sample[loss_cfg["input_key"]], sample[loss_cfg["output_key"]])
        else: 
            raise ValueError(f"Unsupported loss type: '{loss_type}'")
        return loss_value

    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        training = True 
        # forward pass
        sample = self.forward(batch, train=training, validation=False, teacher_forcing=False, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(sample, training=training, validation=False, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"train/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss

    def validation_step(self, batch, batch_idx, *args, **kwargs): 
        training = False 

        # forward pass
        sample = self.forward(batch, train=training, validation=True, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(sample, training=training, validation=True, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"val_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"val/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss, losses_and_metrics_to_log

    def test_step(self, batch, batch_idx, *args, **kwargs):
        training = False 

        # forward pass
        sample = self.forward(batch, train=training, teacher_forcing=False, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(sample, training, validation=False, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"test/" + k: v.item() if isinstance(v, (torch.Tensor,)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        if self.logger is not None:
            # self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss

    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'VideoClassifierBase':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = VideoClassifierBase(cfg, prefix)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = VideoClassifierBase.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                strict=False, 
                **checkpoint_kwargs)
            # if stage == 'train':
            #     mode = True
            # else:
            #     mode = False
            # model.reconfigure(cfg, prefix, downgrade_ok=True, train=mode)
        return model

def sequence_encoder_from_cfg(cfg, feature_dim):
    if cfg.type == "TransformerSequenceClassifier":
        return TransformerSequenceClassifier(cfg, feature_dim)
    else:
        raise ValueError(f"Unknown sequence classifier model: {cfg.model}")

def classification_head_from_cfg(cfg, feature_size, num_classes):
    if cfg.type == "LinearClassificationHead":
        return LinearClassificationHead(cfg, feature_size, num_classes)
    else:
        raise ValueError(f"Unknown classification head model: {cfg.model}")


def sequence_feature_from_cfg(cfg):
    if cfg is None or not cfg.type: 
        return None
    # elif cfg.model == "transformer":
    #     return TransformerSequenceFeature(cfg, feature_dim, num_labels)
    else:
        raise ValueError(f"Unknown sequence classifier model: {cfg.model}")


class MostFrequentEmotionClassifier(VideoClassifierBase): 

    def __init__(self, cfg):
        super().__init__(cfg)
    
    def get_trainable_parameters(self):
        return super().get_trainable_parameters()

    def forward(self, batch, train=True, teacher_forcing=True, **kwargs):
        sample = batch
        scores = sample['gt_expression'].mean(dim=1)
        sample['predicted_logits'] = scores
        return sample

    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'VideoEmotionClassifier':
        """
        Function that instantiates the model from checkpoint or config
        """
        model = MostFrequentEmotionClassifier(cfg)
        return model



class VideoEmotionClassifier(VideoClassifierBase): 

    def __init__(self, 
                 cfg
        ):
        self.cfg = cfg
        preprocessor = None
        feature_model = sequence_feature_from_cfg(cfg.model.get('feature_extractor', None))
        feature_size = feature_model.output_size if feature_model is not None else cfg.model.input_feature_size
        sequence_classifier = sequence_encoder_from_cfg(cfg.model.get('sequence_encoder', None), feature_size)
        classification_head = classification_head_from_cfg(cfg.model.get('classification_head', None), 
                                                           sequence_classifier.encoder_output_dim(), 
                                                           cfg.model.output.num_classes)

        super().__init__(cfg,
            preprocessor = preprocessor,
            feature_model = feature_model,
            sequence_encoder = sequence_classifier,  
            classification_head = classification_head,  
        )


    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'VideoEmotionClassifier':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = VideoEmotionClassifier(cfg)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = VideoEmotionClassifier.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=False, 
                **checkpoint_kwargs
            )
            # if stage == 'train':
            #     mode = True
            # else:
            #     mode = False
            # model.reconfigure(cfg, prefix, downgrade_ok=True, train=mode)
        return model


def check_nan(sample: Dict): 
    ok = True
    nans = []
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN found in '{key}'")
                nans.append(key)
                ok = False
                # raise ValueError("Nan found in sample")
    if len(nans) > 0:
        raise ValueError(f"NaN found in {nans}")
    return ok


def truncate_sequence_batch(sample: Dict, max_seq_length: int) -> Dict:
    """
    Truncate the sequence to the given length. 
    """
    # T = sample["audio"].shape[1]
    # if max_seq_length < T: # truncate
    for key in sample.keys():
        if isinstance(sample[key], torch.Tensor): # if temporal element, truncate
            if sample[key].ndim >= 3:
                sample[key] = sample[key][:, :max_seq_length, ...]
        elif isinstance(sample[key], Dict): 
            sample[key] = truncate_sequence_batch(sample[key], max_seq_length)
        elif isinstance(sample[key], List):
            pass
        else: 
            raise ValueError(f"Invalid type '{type(sample[key])}' for key '{key}'")
    return sample
