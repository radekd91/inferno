"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
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
import os, sys
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from inferno.utils.condor import execute_on_cluster
from pathlib import Path
import inferno_apps.TalkingHead.training.train_talking_head as script
import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
import time as t
from inferno.utils.other import get_path_to_assets
import sys
import random

submit_ = False
# submit_ = True

if submit_ or __name__ != "__main__":
    config_path = Path(__file__).parent / "submission_settings.yaml"
    if not config_path.exists():
        cfg = DictConfig({})
        cfg.cluster_repo_path = "todo"
        cfg.submission_dir_local_mount = "todo"
        cfg.submission_dir_cluster_side = "todo"
        cfg.python_bin = "todo"
        cfg.username = "todo"
        OmegaConf.save(config=cfg, f=config_path)
        
    user_config = OmegaConf.load(config_path)
    for key, value in user_config.items():
        if value == 'todo': 
            print("Please fill in the settings.yaml file")
            sys.exit(0)


def submit(cfg , bid=10, 
    stage=None, 
    resume_from_previous=None, 
    force_new_location=None,
    max_price = None
):
    cluster_repo_path = user_config.cluster_repo_path
    submission_dir_local_mount = user_config.submission_dir_local_mount
    submission_dir_cluster_side = user_config.submission_dir_cluster_side

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(time) + random.randint(-100000, 100000)) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(script.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[2].name / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    config_file = submission_folder_local / "config.yaml"

    with open(config_file, 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)

    python_bin = user_config.python_bin
    username = user_config.username
    gpu_mem_requirement_mb = cfg.learning.batching.gpu_memory_min_gb * 1024
    gpu_mem_requirement_mb_max = cfg.learning.batching.get('gpu_memory_max_gb' , None)
    if gpu_mem_requirement_mb_max is not None:
        gpu_mem_requirement_mb_max *= 1024
    # gpu_mem_requirement_mb = None
    cpus = cfg.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.learning.batching.num_gpus
    num_jobs = 1
    max_time_h = 36
    job_name = "train_talking_head"
    # cuda_capability_requirement = 7
    cuda_capability_requirement = 7.5
    mem_gb = 50

    args = f"{config_file.name}"    
    env = user_config.env

    if stage is not None and resume_from_previous is not None and force_new_location is not None:
        args += f" {stage} {int(resume_from_previous)} {int(force_new_location)}"
    elif stage is not None or resume_from_previous is not None or force_new_location is not None:
        raise ValueError("stage, resume_from_previous and force_new_location must be all None or all not None")

    # if env is an absolute path to the conda environment
    if Path(env).exists() and Path(env).resolve() == Path(env):
        python_bin = str(Path(env) / "bin/python")
        assert Path(python_bin).exists(), f"Python binary {python_bin} does not exist"

    execute_on_cluster(str(cluster_script_path),
                       args,
                       str(submission_folder_local),
                       str(submission_folder_cluster),
                       str(cluster_repo_path),
                       python_bin=python_bin,
                       username=username,
                       gpu_mem_requirement_mb=gpu_mem_requirement_mb,
                       gpu_mem_requirement_mb_max=gpu_mem_requirement_mb_max,
                       cpus=cpus,
                       mem_gb=mem_gb,
                       gpus=gpus,
                       num_jobs=num_jobs,
                       bid=bid,
                       max_time_h=max_time_h,
                       job_name=job_name,
                       cuda_capability_requirement=cuda_capability_requirement,
                       env=env,
                       max_price=max_price,
                       )
    # t.sleep(2)


def submit_trainings():
    from hydra.core.global_hydra import GlobalHydra
    ## 0) Set your data paths and result path
    # model_output_dir = None ## default from configs
    model_output_dir = str(get_path_to_assets().absolute() / "TalkingHead/trainings")
    
    ## 1) Base EMOTE config - bert-like model with motion prior
    conf = "bertprior_wild" 
    
    tags = [] 
    
    ## 2) Set hyperparameters
    # use_shape = True
    use_shape = False ## EMOTE does not condition on shape

    # style_operation = 'add'
    style_operation = 'cat' ## concatenation is used in EMOTE to incorporate style

    use_identity = True ## use identity one-hot as part of style
    # use_identity = False

    training_modes = [
        [ 
            f'model.sequence_decoder.style_embedding.use_shape={use_shape}',
            f'model.sequence_decoder.style_embedding.gt_expression_identity={use_identity}',
            f'+model.sequence_decoder.style_op={style_operation}',
        ],
    ]

    ## 3) Dataset, splits, batching
    dataset = "mead_pseudo_gt"
    # reconstruction_type = "EMICA_mead_mp_lr_mse_15" ## old version of data used in EMOTE paper
    reconstruction_type = "EMICA-MEAD_flame2020" ## new version of data with much better reconstructions
    # batching = "fixed_length_bs32_seq32"
    batching = "fixed_length_bs4_45gb"

    preprocessor = "flame_tex"
        
    ### MEAD splits
    ## split = "random_70_15_15"
    ## split = "random_by_sequence_random_70_15_15" 
    # split = "random_by_sequence_sorted_70_15_15" ## identity overlap between train/val/test (not used in paper EMOTE)
    ## split = "random_by_identityV2_random_70_15_15" 
    split = "random_by_identityV2_sorted_70_15_15" ## split used to train EMOTE (no identity overlap between train/val/test)
    ## split = "specific_identity_random_80_20_M003"
    # split = "specific_identity_sorted_80_20_M003" ## specific identity (M003) for quick testing
    ## split = "specific_identity_random_80_20_M005"
    # split = "specific_identity_sorted_80_20_M005"
    
    ## 3b) set paths to the data (where you store MEAD), or use the default paths which are set in the config    
    # mead_input_dir = "/is/cluster/fast/rdanecek/data/mead_25fps/resampled_videos"
    # mead_processed_dir = "/is/cluster/fast/rdanecek/data/mead_25fps/"
    # mead_processed_subfolder = "processed"
    mead_input_dir = None
    mead_processed_dir = None
    mead_processed_subfolder = None
    

    ## 4) Motion prior - Choose your motion prior (aka FLINT)
    # motion_prior_path = get_path_to_assets() / "MotionPrior" / "models"
    motion_prior_path = Path("MotionPrior") / "models"
    # motion_prior_name = "FLINT" ## FLINT in EMOTE paper
    motion_prior_name = "FLINTv2"  ## FLINT of EMOTE v2
    fixed_overrides = []

    ## specify the motion prior for EMOTE 
    fixed_overrides += [f'model.sequence_decoder.motion_prior.path={str(motion_prior_path / motion_prior_name)}']

    # open motion prior 
    motion_prior_cfg = OmegaConf.load(motion_prior_path / motion_prior_name / "cfg.yaml")
    assert split == motion_prior_cfg.data.split, f"Split '{split}' does not match motion prior split '{motion_prior_cfg.data.split}'"
    # assert reconstruction_type == motion_prior_cfg.data.reconstruction_type, \
    #     f"Reconstruction '{reconstruction_type}' does not match motion prior split '{motion_prior_cfg.data.reconstruction_type}'. " \
    #     f"This is probably not what you want."


    ## 5) Submit the training
    fixed_overrides += [f'data/datasets={dataset}']
    fixed_overrides += [f'data.reconstruction_type={reconstruction_type}']
    if batching is not None:
        fixed_overrides += [f'+learning/batching@learning.batching={batching}']
    if preprocessor is not None:
        fixed_overrides += [f'+model/preprocessor@model.preprocessor={preprocessor}']
    if split is not None:
        fixed_overrides += [f'data.split={split}']

    if model_output_dir is not None:
        fixed_overrides += [f'inout.output_dir={model_output_dir}']
        
    # override the paths to the data
    if mead_input_dir is not None:
        fixed_overrides += [f'data.input_dir={mead_input_dir}']
    if mead_processed_dir is not None:
        fixed_overrides += [f'data.output_dir={mead_processed_dir}']
    if mead_processed_subfolder is not None:
        fixed_overrides += [f'data.processed_subfolder={mead_processed_subfolder}']
    

    # bid = 1000
    bid = 250
    max_price = 500


    ## optionally disable logging    
    # if not submit_:
        # fixed_overrides += [
        #     'learning/logging=none',
        # ]

    for fmode in training_modes:
        overrides = fixed_overrides.copy()
        overrides += fmode

        cfg = script.configure(
            conf, overrides,
        )

        GlobalHydra.instance().clear()
        # config_pairs += [cfgs]

        # if not submit_:
        #     with open_dict(cfg) as d:
        #         # if dataset == "vocaset":
        #         #     d.data.debug_mode = True
        #         d.data.num_workers = 0
        #         # d.data.num_workers = 1
        #         tags += ["DEBUG_FROM_WORKSTATION"]
        #         if d.learning.tags is None:
        #             d.learning.tags = tags

        with open_dict(cfg) as d:
            if d.learning.tags is None:
                d.learning.tags = tags
    
            if 'motion_prior' in d.model.sequence_decoder and 'path' in d.model.sequence_decoder.motion_prior:
                # load motion prior config
                d.model.sequence_decoder.motion_prior.cfg = motion_prior_cfg


        if submit_:
            submit(cfg, bid=bid, max_price=max_price)
        else:
            script.train_model(cfg, resume_from_previous=False)






if __name__ == "__main__":
    # default_main()
    submit_trainings()

