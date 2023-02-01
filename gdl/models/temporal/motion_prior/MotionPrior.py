import pytorch_lightning as pl 
import torch
from torch import nn
from typing import Any, Optional 
from gdl.models.temporal.Bases import ShapeModel, Preprocessor


class MotionEncoder(torch.nn.Module): 

    def forward(self, input):
        raise NotImplementedError()

    def get_trainable_parameters(self): 
        return [p for p in self.parameters() if p.requires_grad]


class MotionDecoder(torch.nn.Module):

    def forward(self, input):
        raise NotImplementedError()

    def get_trainable_parameters(self): 
        return [p for p in self.parameters() if p.requires_grad]


class MotionQuantizer(torch.nn.Module):
    
    def forward(self, input):
        raise NotImplementedError()

    def get_trainable_parameters(self): 
        return [p for p in self.parameters() if p.requires_grad]



# def motion_encoder_from_cfg(cfg) -> MotionEncoder:
#     if cfg.type == "L2lEncoder":
#         return L2lEncoder(cfg)
#     else:
#         raise NotImplementedError(f"Motion encoder type '{cfg.type}' not implemented")


# def motion_decoder_from_cfg(cfg) -> MotionDecoder:
#     if cfg.type == "L2lDecoder":
#         return L2lDecoder(cfg)
#     else:
#         raise NotImplementedError(f"Motion decoder type '{cfg.type}' not implemented")
        

# def motion_codebook_from_cfg(cfg) -> MotionDecoder:
#     if cfg.type == "L2lCodebook":
#         return L2lCodebook(cfg)
#     else:
#         raise NotImplementedError(f"Motion decoder type '{cfg.type}' not implemented")
        

class MotionPrior(pl.LightningModule):

    def __init__(self, 
        motion_encoder : MotionEncoder,
        motion_decoder : MotionDecoder,
        motion_quantizer: Optional[MotionQuantizer] = None,
        shape_model: Optional[ShapeModel] = None,
        preprocessor: Optional[Preprocessor] = None,
        ) -> None:
        super().__init__()
        self.motion_encoder = motion_encoder
        self.motion_decoder = motion_decoder
        self.motion_quantizer = motion_quantizer
        self.shape_model = shape_model
        self.preprocessor = preprocessor


    @property
    def max_seq_length(self):
        return 5000

    def get_trainable_parameters(self):
        trainable_params = []
        trainable_params += self.motion_encoder.get_trainable_parameters()
        trainable_params += self.motion_decoder.get_trainable_parameters()
        if self.motion_quantizer is not None:
            trainable_params += self.motion_quantizer.get_trainable_parameters()
        if self.shape_model is not None:
            trainable_params += self.shape_model.get_trainable_parameters()
        return trainable_params

    def configure_optimizers(self):
        trainable_params = []
        trainable_params += list(self.get_trainable_parameters())

        if self.cfg.learning.optimizer == 'Adam':
            opt = torch.optim.Adam(
                trainable_params,
                lr=self.cfg.learning.learning_rate,
                amsgrad=False)
        # elif self.cfg.learning.optimizer == 'AdaBound':
        #     opt = adabound.AdaBound(
        #         trainable_params,
        #         lr=self.cfg.learning.learning_rate,
        #         final_lr=self.cfg.learning.final_learning_rate
        #     )
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

    def preprocess(self, batch):
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        return batch

    def encode(self, batch):
        batch = self.motion_encoder(batch)
        return batch

    def quantize(self, batch):
        if self.motion_quantizer is not None:
            batch = self.motion_quantizer(batch)
        return batch

    def decode(self, batch):
        batch = self.motion_decoder(batch)
        return batch

    def forward(self, batch):
        batch = self.preprocess(batch)
        batch = self.encode(batch)
        batch = self.quantize(batch)
        batch = self.decode(batch)
        return batch

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


    def training_step(self, batch, batch_idx, *args, **kwargs):
        training = True 
        # forward pass
        sample = self.forward(batch, train=training, validation=False, **kwargs)
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
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"val/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss


