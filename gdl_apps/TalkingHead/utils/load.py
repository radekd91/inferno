from pathlib import Path
from omegaconf import OmegaConf
import os, sys

from gdl.models.talkinghead.FaceFormer import FaceFormer
from gdl.models.IO import locate_checkpoint


def load_model(path_to_models,
              run_name,
            #   stage,
            #   relative_to_path=None,
            #   replace_root_path=None,
              mode='best',
            #   allow_stage_revert=False, # allows to load coarse if detail checkpoint not found
              ):
    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        cfg = OmegaConf.load(f)

    # conf = replace_asset_dirs(conf, Path(path_to_models) / run_name)
    # conf.coarse.checkpoint_dir = str(Path(path_to_models) / run_name / "coarse" / "checkpoints")
    # conf.coarse.full_run_dir = str(Path(path_to_models) / run_name / "coarse" )
    # conf.coarse.output_dir = str(Path(path_to_models) )
    # conf.detail.checkpoint_dir = str(Path(path_to_models) / run_name / "detail" / "checkpoints")
    # conf.detail.full_run_dir = str(Path(path_to_models) / run_name / "detail" )
    # conf.detail.output_dir = str(Path(path_to_models) )
    faceformer = load_faceformer(cfg,
              mode,
            #   relative_to_path,
            #   replace_root_path,
            #   terminate_on_failure= not allow_stage_revert
              )
    # if faceformer is None and allow_stage_revert:
    #     faceformer = load_faceformer(conf,
    #                      mode,
    #                     #  relative_to_path,
    #                     #  replace_root_path,
    #                      )

    return faceformer, cfg



def load_faceformer(
              cfg,
            #   stage,
              mode,
              relative_to_path=None,
              replace_root_path=None,
              terminate_on_failure=True,
              ):
    # print(f"Taking config of stage '{stage}'")
    # print(conf.keys())
    # if stage is not None:
    #     cfg = conf[stage]
    # else:
    #     cfg = conf
    # if relative_to_path is not None and replace_root_path is not None:
    #     cfg = hack_paths(cfg, replace_root_path=replace_root_path, relative_to_path=relative_to_path)
    # cfg.model.resume_training = False

    checkpoint = locate_checkpoint(cfg, replace_root_path, relative_to_path, mode=mode)
    if checkpoint is None:
        if terminate_on_failure:
            sys.exit(0)
        else:
            return None
    print(f"Loading checkpoint '{checkpoint}'")
    # if relative_to_path is not None and replace_root_path is not None:
    #     cfg = hack_paths(cfg, replace_root_path=replace_root_path, relative_to_path=relative_to_path)

    checkpoint_kwargs = {
        "cfg": cfg
    }
    deca = FaceFormer.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
    return deca