import pytorch_lightning as pl 
from typing import Any, Optional
from gdl.models.temporal.AudioEncoders import TemporalAudioEncoder
from gdl.models.temporal.MultiModalTemporalNet import *
from gdl.models.temporal.Bases import *
from gdl.models.temporal.BlockFactory import norm_from_cfg
import random


class TalkingHeadBase(pl.LightningModule): 

    def __init__(self, 
                cfg,
                audio_model: Optional[TemporalAudioEncoder] = None, 
                sequence_encoder : SequenceEncoder = None,
                sequence_decoder : Optional[SequenceDecoder] = None,
                shape_model: Optional[ShapeModel] = None,
                preprocessor: Optional[Preprocessor] = None,
                # renderer: Optional[Renderer] = None,
                # post_fusion_norm: Optional = None,
                *args: Any, **kwargs: Any) -> None:
        self.cfg = cfg
        super().__init__(*args, **kwargs)  
        self.audio_model = audio_model
        self.sequence_encoder = sequence_encoder
        self.sequence_decoder = sequence_decoder
        self.shape_model = shape_model
        self.preprocessor = preprocessor
        # self.renderer = renderer

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

    def training_step(self, batch, batch_idx, *args, **kwargs):
        training = False 

        # forward pass
        sample = self.forward(batch, train=training, teacher_forcing=False, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(sample, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"train/" + k: v.item() if isinstance(v, (torch.Tensor, float)) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            # self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss


    def _validation_step(self, batch, batch_idx, *args, **kwargs): 
        training = False 

        # forward pass
        sample = self.forward(batch, train=training, teacher_forcing=True, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(sample, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"val_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"val/" + k: v.item() if isinstance(v, (torch.Tensor, float)) else 0. for k, v in losses_and_metrics_to_log.items()}

       
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
        total_loss, losses, metrics = self.compute_loss(sample, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"test/" + k: v.item() if isinstance(v, (torch.Tensor, float)) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss

    def forward_audio(self, sample: Dict, train=False, desired_output_length=None, **kwargs: Any) -> Dict:
        return self.audio_model(sample, train=train, desired_output_length=desired_output_length, **kwargs)

    def encode_sequence(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        return self.sequence_encoder(sample, input_key="audio_feature", **kwargs)

    def decode_sequence(self, sample: Dict, train=False, teacher_forcing=False, **kwargs: Any) -> Dict:
        return self.sequence_decoder(sample, train=train, teacher_forcing=teacher_forcing, **kwargs)

    @torch.no_grad()
    def preprocess_input(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        if self.preprocessor is not None:
            if self.device != self.preprocessor.device:
                self.preprocessor.to(self.device)
            sample = self.preprocessor(sample, input_key="video", train=train, test_time=not train, **kwargs)
        return sample 

    def forward(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        """
        sample: Dict[str, torch.Tensor]
            - audio: (B, T, F)
            # - masked_audio: (B, T, F)
        """
        T = sample["raw_audio"].shape[1]
        if self.max_seq_length < T: # truncate
            print("[WARNING] Truncating audio sequence from {} to {}".format(T, self.max_seq_length))
            sample = truncate_sequence_batch(sample, self.max_seq_length)

        # preprocess input (for instance get 3D pseudo-GT )
        sample = self.preprocess_input(sample, train=train, **kwargs)
        check_nan(sample)
        
        teacher_forcing = kwargs.pop("teacher_forcing", False)
        desired_output_length = sample["gt_vertices"].shape[1] if "gt_vertices" in sample.keys() else None
        sample = self.forward_audio(sample, train=train, desired_output_length=desired_output_length, **kwargs)
        # if self.uses_text():
        #     sample = self.forward_text(sample, **kwargs)
        check_nan(sample)
        # encode the sequence
        sample = self.encode_sequence(sample, train=train, **kwargs)
        check_nan(sample)
        # decode the sequence
        sample = self.decode_sequence(sample, train=train, teacher_forcing=teacher_forcing, **kwargs)
        check_nan(sample)
        return sample



    def compute_loss(self, sample): 
        """
        Compute the loss for the given sample. 

        """
        losses = {}
        # loss_weights = {}
        metrics = {}

        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            assert loss_name not in losses.keys()
            losses["loss_" + loss_name] = self._compute_loss(sample, loss_name, loss_cfg)

        for metric_name, metric_cfg in self.cfg.learning.metrics.items():
            assert metric_name not in metrics.keys()
            with torch.no_grad():
                metrics["metric_" + metric_name] = self._compute_loss(sample, metric_name, metric_cfg)

        total_loss = None
        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            term = losses["loss_" + loss_name] 
            if term is not None:
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