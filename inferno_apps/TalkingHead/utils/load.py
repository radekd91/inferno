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

from inferno.models.talkinghead.FaceFormer import FaceFormer
from inferno.models.IO import locate_checkpoint


def load_model(path_to_models,
              run_name,
              mode='latest',
              with_losses=True,
              ):
    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        cfg = OmegaConf.load(f)
    if not with_losses:
        cfg.learning.losses = {}
        cfg.learning.metrics = {}

    faceformer = load_faceformer(cfg, mode,)
    return faceformer, cfg


def load_faceformer(
              cfg,
              mode,
              relative_to_path=None,
              replace_root_path=None,
              terminate_on_failure=True,
              ):
    checkpoint = locate_checkpoint(cfg, replace_root_path, relative_to_path, mode=mode)
    if checkpoint is None:
        if terminate_on_failure:
            sys.exit(0)
        else:
            return None
    print(f"Loading checkpoint '{checkpoint}'")
    checkpoint_kwargs = {
        "cfg": cfg
    }
    deca = FaceFormer.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
    return deca