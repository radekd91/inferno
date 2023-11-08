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
import os, sys
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from inferno.utils.condor import execute_on_cluster
from pathlib import Path
import inferno_apps.MotionPrior.training.train_motion_prior as script
import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
import time as t
import copy
import random

# submit_ = False
submit_ = True

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
            print("Please fill in the submission_settings.yaml file")
            sys.exit(0)



def submit(cfg , bid=10):
    cluster_repo_path = user_config.cluster_repo_path
    submission_dir_local_mount = user_config.submission_dir_local_mount
    submission_dir_cluster_side = user_config.submission_dir_cluster_side

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(time) + random.randint(-10000, 10000)) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(script.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[2].name / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    config_file = submission_folder_local / "config.yaml"

    with open(config_file, 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)

    # python_bin = 'python'
    python_bin = user_config.python_bin
    username = user_config.username
    gpu_mem_requirement_mb = cfg.learning.batching.gpu_memory_min_gb * 1024
    gpu_mem_requirement_mb_max = cfg.learning.batching.get('gpu_memory_max_gb', None)
    if gpu_mem_requirement_mb_max is not None:
        gpu_mem_requirement_mb_max *= 1024
    # gpu_mem_requirement_mb = None
    cpus = cfg.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.learning.batching.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_motion_prior"
    cuda_capability_requirement = 7.5
    # mem_gb = 40
    mem_gb = 45

    args = f"{config_file.name}"
    env = user_config.env

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
                       max_price=max_price,
                       job_name=job_name,
                       cuda_capability_requirement=cuda_capability_requirement,
                       env=env,
                       )
    # t.sleep(2)


def submit_trainings():
    from hydra.core.global_hydra import GlobalHydra

    ## 1) Base FLINT config
    conf = "l2l-vae_geometry"

    tags = []
    tags += ['EMICA_v2']

    training_modes = [
        # [], # no modifications to default config
        
        [ ## final FLINT setting
            '+model/sequence_decoder@model.sequence_decoder=l2l_decoder_post_proj',  
        ],

        # [ ## this option uses no convolution in FLINT
        #    '+model/sequence_decoder@model.sequence_decoder=l2l_decoder_post_proj_no_conv',  
        # ],
    ]


    ## 2) FLINT config dataset settings
    ## a) dataset config name
    dataset = "mead_pseudo_gt"
    # reconstruction_type = "EMICA_mead_mp_lr_mse_15" ## old version of data used in EMOTE paper
    reconstruction_type = "EMICA-MEAD_flame2020" ## new version of data with much better reconstructions
    
    ## b) batching config name
    # batching = "fixed_length"
    # batching = "fixed_length_bs16_35gb"
    # batching = "fixed_length150_bs16_35gb"
    batching = "fixed_length_bs32_35gb"
    # batching = "fixed_length_bs64_35gb"
    preprocessor = "flame" ## the dataset is saved in FLAME format, so we need to use the FLAME preprocessor

    ### MEAD splits
    ## split = "random_70_15_15"
    ## split = "random_by_sequence_random_70_15_15" 
    # split = "random_by_sequence_sorted_70_15_15" 
    ## split = "random_by_identityV2_random_70_15_15" 
    split = "random_by_identityV2_sorted_70_15_15" 
    ## split = "specific_identity_random_80_20_M003"
    # split = "specific_identity_sorted_80_20_M003"

    fixed_overrides = []

    
    fixed_overrides += [f'data/datasets={dataset}']
    fixed_overrides += [f'data.reconstruction_type={reconstruction_type}']
    
    if batching is not None:
        fixed_overrides += [f'+learning/batching@learning.batching={batching}']
    if preprocessor is not None:
        fixed_overrides += [f'+model/preprocessor@model.preprocessor={preprocessor}']
    if split is not None:
        fixed_overrides += [f'data.split={split}']

    bid = 200
    
    # ## optionally disable logging (wand by default)
    # if not submit_:
        # fixed_overrides += [
        #     '+learning.logging=none',
        # ]

    ## 3) set (or sweep through) important training parameters (e.g. number of layers, feature dimension, kl weight, ...)
    for fmode in training_modes:
        overrides = fixed_overrides.copy()
        overrides += fmode

        # num_layer_list = [None] # default 
        # num_layer_list = [1, 2,  4,  6,  8, 12]
        num_layer_list = [1] ## final setting
        for num_layers in num_layer_list:
            
            if num_layers is not None:
                overrides += ['model.sequence_encoder.num_layers=' + str(num_layers)]
                overrides += ['model.sequence_decoder.num_layers=' + str(num_layers)]
        

            # quant_factor_list = [None] # 
            # quant_factor_list = [1, 2, 3, 4, 5]
            quant_factor_list = [3] ## final setting - one latent frame is 2^3=8 real frames

            # feature_dims = [None] # default
            # feature_dims = [16, 32, 64, 128, 256]
            feature_dims = [128] ## final setting
            for feature_dim in feature_dims:
                if feature_dim is not None:
                    overrides += ['model.sequence_encoder.feature_dim=' + str(feature_dim)]
                    overrides += ['model.sequence_decoder.feature_dim=' + str(feature_dim)]


                for quant_factor in quant_factor_list:
                    if quant_factor is not None:
                        overrides += ['model.sizes.quant_factor=' + str(quant_factor)]

                    # kl_weights = [None]
                    # kl_weights = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
                    kl_weights = [0.001] ## final setting
                    for kl_weight in kl_weights:
                        if kl_weight is not None:
                            overrides += ['learning.losses.kl_divergence.weight=' + str(kl_weight)]

                        cfg = script.configure(
                            conf, overrides,
                        )

                        GlobalHydra.instance().clear()
                        with open_dict(cfg) as d:
                            # if not submit_:
                            #     d.data.debug_mode = True
                            #     tags += ["DEBUG_FROM_WORKSTATION"]
                            if d.learning.tags is None:
                                d.learning.tags = tags
                    
                        if submit_:
                            submit(cfg, bid=bid)
                        else:
                            script.train_model(cfg, resume_from_previous=False)



if __name__ == "__main__":
    submit_trainings()

