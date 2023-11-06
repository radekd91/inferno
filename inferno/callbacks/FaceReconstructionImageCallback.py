from typing import Any, Optional
import pytorch_lightning as pl 
from pathlib import Path
from pytorch_lightning.utilities.types import STEP_OUTPUT
from skimage.io import imsave
import os, sys
from wandb import Image
import pickle as pkl
import numpy as np
import torch
from inferno.models.FaceReconstruction.FaceRecBase import FaceReconstructionBase
from pytorch_lightning.loggers import WandbLogger
from inferno.utils.lightning_logging import _log_array_image, _log_wandb_image, _torch_image2np


def nested_dict_access(d, keys):
    if len(keys) == 0:
        return d
    if keys[0] not in d.keys():
        return None
    return nested_dict_access(d[keys[0]], keys[1:])


class FaceReconstructionImageCallback(pl.Callback):

    def __init__(self,  
                 training_frequency=1000, 
                 validation_frequency=100
                 ):
        self.training_frequency = training_frequency
        self.validation_frequency = validation_frequency
        self.training_step_counter = 0
        self.validation_step_counter = 0

    def on_train_batch_end(self, trainer, 
                           pl_module : FaceReconstructionBase, 
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ):
        if self.training_step_counter % self.training_frequency != 0:
            self.training_step_counter += 1
            return 
        
        self.training_step_counter += 1
        # prefix = f"train/{dataloader_idx}/"
        prefix = f"train_vis/"
        visdict = pl_module.visualize_batch(batch, batch_idx, prefix, in_batch_idx=0)
        self._log_image_dict(pl_module, visdict, prefix, self.training_step_counter)

    def on_validation_batch_end(self, 
                                trainer, 
                                pl_module : FaceReconstructionBase, 
                                outputs, 
                                batch: Any, 
                                batch_idx: int, 
                                dataloader_idx: int) -> None:

        if self.validation_step_counter % self.validation_frequency != 0:
            self.validation_step_counter += 1
            return 
        
        self.validation_step_counter += 1
        prefix = f"val_vis/{dataloader_idx}/"
        visdict = pl_module.visualize_batch(batch, batch_idx, prefix, in_batch_idx=0)
        self._log_image_dict(pl_module, visdict, prefix, step=self.validation_step_counter)


    def _log_image_dict(self, pl_module : FaceReconstructionBase, visdict, prefix, step): 
        output_dir = pl_module.cfg.inout.full_run_dir
        log_dict = {}
        for key, image_list in visdict.items():
            for i in range(len(image_list)):
                savepath = Path(f'{output_dir}/{prefix}/{key}/{step:07d}_{i:02d}.png')
                savepath.parent.mkdir(parents=True, exist_ok=True)
                image = image_list[i]
                if image.dtype != np.uint8: 
                    image = (image * 255.).astype(np.uint8)
                caption = None
                if isinstance(pl_module.logger, WandbLogger):
                    im2log = _log_wandb_image(savepath, image, caption)
                elif pl_module.logger is not None:
                    im2log = _log_array_image(savepath, image, caption)
                else:
                    im2log = _log_array_image(None, image, caption)

                log_dict[prefix + key] = im2log 

        if isinstance(pl_module.logger, WandbLogger):
            pl_module.logger.experiment.log(log_dict)
    
    # def on_validation_epoch_end(self, trainer, pl_module : FaceReconstructionBase):
    #     self.validation_step_counter = 0