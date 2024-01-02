"""
Author: Radek Danecek
Copyright (c) 2023, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emote@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

from pathlib import Path
from omegaconf import OmegaConf
import os, sys

from inferno.models.FaceReconstruction.FaceRecBase import FaceReconstructionBase
from inferno.models.IO import locate_checkpoint
from inferno.utils.other import get_path_to_assets

def load_model(path_to_models,
              run_name,
            #   mode='best',
              with_losses=True,
              ):
    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        cfg = OmegaConf.load(f)
    if not with_losses:
        cfg.learning.losses = {}
        cfg.learning.metrics = {}

    face_rec_model = load_face_reconstruction(cfg, 
                                            #   mode,
                                              )
    return face_rec_model, cfg


def load_face_reconstruction(
              cfg,
            #   mode
              ):
    path = Path(cfg.inout.full_run_dir)
    if not path.is_absolute(): 
        path = Path(get_path_to_assets()) / "FaceReconstruction/models" / path / "checkpoints"

    checkpoint = locate_checkpoint(path, mode = cfg.get("checkpoint_mode", "best"))
    assert checkpoint is not None, "No checkpoint found. Check the paths in the config file."
    cfg.learning.losses = {}
    cfg.learning.metrics = {}
    face_rec_model = FaceReconstructionBase.instantiate(cfg, checkpoint=checkpoint, strict=False)
        
    return face_rec_model