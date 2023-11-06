import pytorch_lightning as pl 
from pathlib import Path
from skimage.io import imsave
import os, sys
from wandb import Image
import pickle as pkl
import numpy as np
import torch


def nested_dict_access(d, keys):
    if len(keys) == 0:
        return d
    if keys[0] not in d.keys():
        return None
    return nested_dict_access(d[keys[0]], keys[1:])


class ImageSavingCallback(pl.Callback):

    def __init__(self, image_keys_to_save, frequency=100, num_images_in_seq=25):
        self.image_keys_to_save = image_keys_to_save
        self.frequency = frequency
        self.num_images_in_seq = num_images_in_seq


    def on_train_batch_end(self, trainer, pl_module, 
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        
        # save images
        if trainer.global_step % self.frequency != 0:
            return 

        self.save_image(trainer, pl_module, batch, "train/")

    def on_validation_batch_end(self, trainer, pl_module, 
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        
        # save images
        if trainer.global_step % self.frequency != 0:
            return 

        self.save_image(trainer, pl_module, batch, "val/")

    def save_image(self, trainer, pl_module, batch, prefix):
        for image_key in self.image_keys_to_save: 

            image = nested_dict_access(batch, image_key.split("."))
            if image is None:
                continue

            if image.ndim == 5:
                image = image[0]

            assert image.ndim == 4, f"Image must be 4D, got {image.ndim}D"

            image = image[:self.num_images_in_seq]
            T, C, H, W = image.shape

            # image is now T x C x H x W 
            # concatenate the temporal dimension into width 
            image = image.permute(0, 2, 3, 1).contiguous().reshape(H*T, W, C)
            image = torch.cat( image.split(H, dim=0), dim=1)

            # image is now H x W*T x C
            # to numpy
            image = (image.detach().cpu().numpy() * 255.).astype(np.uint8)

            # save image
            image_path = Path(pl_module.cfg.inout.full_run_dir) / "images" / prefix / image_key / f"{trainer.global_step:09d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            
            imsave(image_path, image)

            # log image
            self._log_image(image_path, pl_module.logger, prefix + image_key)


    def _log_image(self, image_path, logger, name):
        if logger is not None: 
            if isinstance(logger, pl.loggers.WandbLogger):
                # name = "test_video" 
                # dl_name = self.dl_names[image_path.parent]
                # if dl_name is not None:
                #     name += f"/{dl_name}"
                # condition = self.video_conditions[image_path.parent]
                # if condition is not None:
                #     name += "/" + condition 
                # name += "/" + str(self._path_chunk(image_path.parent))

                logger.experiment.log({name: Image(str(image_path), caption=str(image_path))}) #, step=epoch)