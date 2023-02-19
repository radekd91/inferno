import pytorch_lightning as pl 
from typing import Any, Optional
from gdl.models.temporal.AudioEncoders import TemporalAudioEncoder
from gdl.models.temporal.MultiModalTemporalNet import *
from gdl.models.temporal.Bases import *
from gdl.models.temporal.BlockFactory import norm_from_cfg
import random
import omegaconf


class TalkingHeadBase(pl.LightningModule): 

    def __init__(self, 
                cfg,
                audio_model: Optional[TemporalAudioEncoder] = None, 
                sequence_encoder : SequenceEncoder = None,
                sequence_decoder : Optional[SequenceDecoder] = None,
                shape_model: Optional[ShapeModel] = None,
                preprocessor: Optional[Preprocessor] = None,
                renderer: Optional[Renderer] = None,
                # post_fusion_norm: Optional = None,
                *args: Any, **kwargs: Any) -> None:
        self.cfg = cfg
        super().__init__(*args, **kwargs)  
        self.audio_model = audio_model
        self.sequence_encoder = sequence_encoder
        self.sequence_decoder = sequence_decoder
        self.shape_model = shape_model
        self.preprocessor = preprocessor
        self.renderer = renderer

        if shape_model is not None and renderer is not None:
            self.renderer.set_shape_model(shape_model)
        elif renderer is not None and self.sequence_decoder is not None:
            self.renderer.set_shape_model(self.sequence_decoder.get_shape_model())
        if self.sequence_decoder is None: 
            self.code_vec_input_name = "seq_encoder_output"
        else:
            self.code_vec_input_name = "seq_decoder_output"

    @property
    def max_seq_length(self):
        return 5000

    def configure_optimizers(self):
        trainable_params = []
        trainable_params += list(self.get_trainable_parameters())

        if self.cfg.learning.optimizer == 'Adam':
            opt = torch.optim.Adam(
                trainable_params,
                lr=self.cfg.learning.learning_rate,
                amsgrad=False)
        elif self.cfg.learning.optimizer == 'AdaBound':
            opt = adabound.AdaBound(
                trainable_params,
                lr=self.cfg.learning.learning_rate,
                final_lr=self.cfg.learning.final_learning_rate
            )

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

    def get_trainable_parameters(self):
        trainable_params = []
        if self.audio_model is not None:
            trainable_params += self.audio_model.get_trainable_parameters()
        if self.sequence_encoder is not None:
            trainable_params += self.sequence_encoder.get_trainable_parameters()
        if self.sequence_decoder is not None:
            trainable_params += self.sequence_decoder.get_trainable_parameters()
        if self.shape_model is not None:
            trainable_params += self.shape_model.get_trainable_parameters()
        return trainable_params

    def _choose_primary_3D_rec_method(self, batch): 
        method = self.cfg.data.get('reconstruction_type', None)
        if method is None:
            return batch
        if isinstance(method, (list, omegaconf.listconfig.ListConfig)):
            # method = random.choice(method)
            method = method[0]
        if "reconstruction" in batch:
            for k in batch["reconstruction"][method].keys():
                assert k not in batch.keys(), f"Key '{k}' already exists in batch. We don't want to overwrite it."
                batch[k] = batch["reconstruction"][method][k]
        return batch


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
            # self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss


    def _validation_step(self, batch, batch_idx, *args, **kwargs): 
        training = False 

        # forward pass
        sample = self.forward(batch, train=training, validation=True, teacher_forcing=True, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(sample, training=training, validation=True, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"val_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"val/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}

       
        return total_loss, losses_and_metrics_to_log

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        # if "one_hot" not in batch.keys():
        if "one_hot" not in batch.keys() or batch["one_hot"].ndim <= 2: #one-hot specified
            total_loss, losses_and_metrics_to_log =  self._validation_step(batch, batch_idx, *args, **kwargs)
            if self.logger is not None:
                self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            return total_loss
        # one hot is not specified, so we iterate over them
        num_subjects = batch["one_hot"].shape[1]
        total_loss = None
        losses_and_metrics_to_log = None
        one_hot = batch["one_hot"]
        for i in range(num_subjects):
            # batch["one_hot"] = one_hot[:, i:i+1, :]
            batch["one_hot"] = one_hot[:, i, :]
            one_subject_loss, subject_losses_and_metrics_to_log =  self._validation_step(batch, batch_idx, *args, **kwargs)
            if total_loss is None:
                total_loss = one_subject_loss
                losses_and_metrics_to_log = subject_losses_and_metrics_to_log.copy()
            else:
                total_loss += one_subject_loss
                for k, v in subject_losses_and_metrics_to_log.items():
                    losses_and_metrics_to_log[k] += v
        total_loss /= num_subjects
        for k, v in losses_and_metrics_to_log.items():
            losses_and_metrics_to_log[k] /= num_subjects

        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended                
        return total_loss

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

        return total_loss

    def forward_audio(self, sample: Dict, train=False, desired_output_length=None, **kwargs: Any) -> Dict:
        return self.audio_model(sample, train=train, desired_output_length=desired_output_length, **kwargs)

    def encode_sequence(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        return self.sequence_encoder(sample, input_key="audio_feature", **kwargs)

    @property
    def disentangle_type(self): 
        return self.cfg.model.get('disentangle_type', None)

    def disentangle_expansion_factor(self, training, validation):
        if self.disentangle_type is None:
            return 1
        if self.disentangle_type == "sample_condition":
            if not training: 
                return 1
            return 2
        if self.disentangle_type == "condition_exchange":
            if not training and not validation:
                return 1
            return 2
        raise ValueError(f"Unknown disentangle type '{self.disentangle_type}'")

    def disentangle(self, sample: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        if not (train or validation):
            return sample

        disentangle_type = self.disentangle_type
        if disentangle_type is None:
            return sample
        
        B, T = sample["audio_feature"].shape[:2]

        sample_shape = self.cfg.model.sequence_decoder.style_embedding.use_shape
        num_shape = self.cfg.model.sequence_decoder.flame.n_shape
        sample_expression = self.cfg.model.sequence_decoder.style_embedding.use_expression
        num_expressions = self.cfg.model.sequence_decoder.style_embedding.n_expression
        sample_valence = self.cfg.model.sequence_decoder.style_embedding.use_valence
        sample_arousal = self.cfg.model.sequence_decoder.style_embedding.use_arousal
            
        if disentangle_type == "sample_condition":

            # torch.distributions.categorical.Categorical(probs=torch.ones(num_expressions) / num_expressions)

            conditions = {}

            if sample_shape: 
                if not hasattr(self, "_shape_distribution"):
                    self._shape_distribution =torch.distributions.Normal(loc=torch.zeros(num_shape, device=self.device), scale=torch.ones(num_shape, device=self.device))
                shape_cond = self._shape_distribution.sample((B,))
                conditions["gt_shape"] = shape_cond

            if sample_expression: 
                if not hasattr(self, "_expression_distribution"):
                    self._expression_distribution = torch.distributions.Uniform(
                        low=torch.zeros(num_expressions, device=self.device), 
                        high=torch.ones(num_expressions, device=self.device))
                    # self._expression_distribution = torch.distributions.categorical.Categorical(probs=torch.ones(num_expressions) / num_expressions)
                expression_cond = self._expression_distribution.sample((B,))
                # normalize 
                expression_cond = expression_cond / expression_cond.sum(dim=1, keepdim=True)
                conditions["gt_expression"] = expression_cond.unsqueeze(1).expand(B, T, num_expressions)

            if sample_valence:
                if not hasattr(self, "_valence_distribution"):
                    self._valence_distribution = torch.distributions.Uniform(low=torch.ones(1,1) * -1, high=torch.ones(1,1))
                valence_cond = self._valence_distribution.sample((B,))
                conditions["gt_valence"] = valence_cond.expand(B, T, 1)

            if sample_arousal:
                if not hasattr(self, "_arousal_distribution"):
                    self._arousal_distribution = torch.distributions.Uniform(low=torch.ones(1,1) * -1, high=torch.ones(1,1))
                arousal_cond = self._arousal_distribution.sample((B,))
                conditions["gt_arousal"] = arousal_cond.expand(B, T, 1)

            for key in sample.keys():
                sample_value = sample[key] 
                if key not in conditions.keys():
                    # expand the batch dimension by a factor of 2   
                    # sample_value_expanded = sample_value.unsqueeze(1).expand(-1, 2, -1, -1, -1).view(B * 2, -1, -1, -1)
                    # sample[key] = sample_value_expanded
                    sample[key] = expand_sequence_batch(sample_value, 2, 'cat')
                    # continue  
                # elif key in conditions:
                else:
                    condition_value = conditions[key]
                    sample_value_expanded = torch.cat([sample_value, condition_value], dim=0)
                    sample[key] = sample_value_expanded
                    # continue

            return sample

        elif disentangle_type == "condition_exchange":
            # assert B == 2, f"Batch size must be 2 for disentangle_type '{disentangle_type}'" 

            keys_to_exchange = [] 
            if sample_shape:
                keys_to_exchange += ["gt_shape"]
            if sample_expression:
                keys_to_exchange += ["gt_expression"] # per-frame pseudo-GT
                if "gt_expression_label" in sample.keys():
                    keys_to_exchange += ["gt_expression_label"] # per sequence emotion
            if sample_valence:
                keys_to_exchange += ["gt_valence"]
            if sample_arousal:
                keys_to_exchange += ["gt_arousal"]

            sample["input_indices"] = torch.arange(B, dtype=torch.int64, device=self.device)
            sample["condition_indices"] = torch.arange(B, dtype=torch.int64, device=self.device)

            keys_to_exchange += ["condition_indices"]
            
            for key in sample.keys(): 
                sample_value = sample[key] 
                if key not in keys_to_exchange:
                    sample[key] = expand_sequence_batch(sample_value, 2, 'cat')
                    continue
                else:
                    # sample_value_exchanged = sample_value[[1, 0]]
                    # sample[key] = torch.cat([sample_value, sample_value_exchanged], dim=0)
                    if B == 2:
                        indexation = 'reverse'
                    else:
                        indexation = 'permute'
                    sample[key] = expand_sequence_batch(sample_value, 2, indexation)
                    continue

            return sample

        raise ValueError(f"Unknown disentangle type: '{disentangle_type}'")            

    def decode_sequence(self, sample: Dict, train=False, teacher_forcing=False, **kwargs: Any) -> Dict:
        return self.sequence_decoder(sample, train=train, teacher_forcing=teacher_forcing, **kwargs)

    def render_sequence(self, sample: Dict, train=False, **kwargs): 
        if self.renderer is None:
            return sample 
        with torch.no_grad():
            if "reconstruction" in sample.keys():
                for method in self.cfg.data.reconstruction_type:
                    sample["reconstruction"][method] = self.renderer(sample["reconstruction"][method], train=train, input_key_prefix='gt_', output_prefix='gt_', **kwargs)
            else:
                sample = self.renderer(sample, train=train, input_key_prefix='gt_', output_prefix='gt_', **kwargs)
        sample = self.renderer(sample, train=train, input_key_prefix='predicted_', output_prefix='predicted_', **kwargs)
        return sample


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
        # T = sample["raw_audio"].shape[1]
        T = sample["processed_audio"].shape[1] if "processed_audio" in sample.keys() else sample["raw_audio"].shape[1]
        if self.max_seq_length < T: # truncate
            print("[WARNING] Truncating audio sequence from {} to {}".format(T, self.max_seq_length))
            sample = truncate_sequence_batch(sample, self.max_seq_length)

        # preprocess input (for instance get 3D pseudo-GT )
        sample = self.preprocess_input(sample, train=train, **kwargs)
        check_nan(sample)

        sample = self._choose_primary_3D_rec_method(sample)
        
        teacher_forcing = kwargs.pop("teacher_forcing", False)
        desired_output_length = sample["gt_vertices"].shape[1] if "gt_vertices" in sample.keys() else None
        sample = self.forward_audio(sample, train=train, desired_output_length=desired_output_length, **kwargs)
        # if self.uses_text():
        #     sample = self.forward_text(sample, **kwargs)
        check_nan(sample)
        # encode the sequence
        sample = self.encode_sequence(sample, train=train, **kwargs)
        check_nan(sample)

        sample = self.disentangle(sample, train=train, validation=validation, **kwargs)
        check_nan(sample)

        # decode the sequence
        sample = self.decode_sequence(sample, train=train, teacher_forcing=teacher_forcing, **kwargs)
        check_nan(sample)

        # render the sequence
        sample = self.render_sequence(sample, train=train, **kwargs)
        check_nan(sample)
        return sample

    def _compute_loss(self, sample, training, validation, loss_name, loss_cfg): 
        raise NotImplementedError("Please implement this method in your child class")

    def compute_loss(self, sample, training, validation): 
        """
        Compute the loss for the given sample. 

        """
        losses = {}
        # loss_weights = {}
        metrics = {}

        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            assert loss_name not in losses.keys()
            losses["loss_" + loss_name] = self._compute_loss(sample, training, validation, loss_name, loss_cfg)

        for metric_name, metric_cfg in self.cfg.learning.metrics.items():
            assert metric_name not in metrics.keys()
            with torch.no_grad():
                metrics["metric_" + metric_name] = self._compute_loss(sample, training, validation, metric_name, metric_cfg)

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


def expand_sequence_batch(sample, factor=2, how='cat') -> Dict:
    if isinstance(sample, torch.Tensor):
        B = sample.shape[0]
        # sample = sample.unsqueeze(1)
        # shape = sample.shape
        # sample_value_expanded = sample.expand(shape[0], factor, *shape[2:]).view(B * factor, *shape[2:])
        indices = torch.arange(B, device=sample.device, dtype=torch.int64)
        if how == 'cat':
            indices = torch.cat([indices, indices], dim=0)
        elif how == 'reverse':
            indices = torch.cat([indices, indices.flip(0)], dim=0)
        elif how == 'permute':
            while True:
                indices_permuted = indices[torch.randperm(B, device=sample.device)]
                if (indices == indices_permuted).sum() == 0:
                    break
            indices = torch.cat([indices, indices_permuted], dim=0)
        else:
            raise ValueError(f"Invalid value '{how}' for argument 'how'")
        sample_value_expanded = sample.index_select(0, indices)
        # sample_value_expanded = sample.expand(shape[0], factor, *shape[2:]).reshape(B * factor, *shape[2:])
    elif isinstance(sample, Dict):
        sample_value_expanded = {}
        for key in sample.keys():
            sample_value_expanded[key] = expand_sequence_batch(sample[key], factor=factor, how=how)
    else: 
        raise ValueError(f"Invalid type '{type(sample)}' for key '{key}'")
    return sample_value_expanded


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


def detach_dict(d): 
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.detach()
        elif isinstance(v, dict):
            d[k] = detach_dict(v)
        else: 
            pass
    return d