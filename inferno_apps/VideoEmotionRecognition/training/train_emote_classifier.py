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
import inferno_apps.VideoEmotionRecognition.training.train_video_emorec as script
import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
import time as t
from inferno.utils.other import get_path_to_assets

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
            print("Please fill in the submission_settings.yaml file")
            sys.exit(0)


def submit(cfg , bid=10, max_price=None):
    cluster_repo_path = user_config.cluster_repo_path
    submission_dir_local_mount = user_config.submission_dir_local_mount
    submission_dir_cluster_side = user_config.submission_dir_cluster_side

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(time)) + "_" + "submission"
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
    # gpu_mem_requirement_mb = None
    cpus = cfg.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.learning.batching.num_gpus
    num_jobs = 1
    max_time_h = 36
    job_name = "train_videoemorec"
    cuda_capability_requirement = 7.5
    mem_gb = 40

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
    ## 0) Set your data paths and result path
    # model_output_dir = None ## default from configs
    model_output_dir = str(get_path_to_assets().absolute() / "VideoEmotionRecognition/trainings")
    

    ## 1) basic classifier for expression and intensity for video emotion feature loss EMOTE
    conf = "videoemorec_transformer"

    ## 2) Choose the training mode (classifying expressions and intensities with two separate heads vs one large head)
    training_modes = [
        # [ ## a single, large prediction head (num expressions * num intensities)
        #     'learning/losses=cross_entropy_exp_with_intensities',
        #     '+model/output@model.output=basic_expressions_with_intensity',
        # ],
        
        [ ## two prediction heads -- used in EMOTE
            'learning/losses=cross_entropy_exp_and_intensities',
            '+model/classification_head@model.classification_head=linear_expression_and_intensity',
            '+model/output@model.output=basic_expressions_and_intensity_separate',
        ],
    ]

    batching = "fixed_length150_bs16_35gb"
    
    
    ## 3) Choose the dataset and split
    dataset= "mead_pseudo_gt"
    
    ### a) MEAD splits
    ## split = "random_70_15_15"
    ## split = "random_by_sequence_random_70_15_15" 
    split = "random_by_sequence_sorted_70_15_15" ## we train on all subjects and split by sequence since we want the emotion to be accurate for all subjects
    ## split = "random_by_identityV2_random_70_15_15" 
    # split = "random_by_identityV2_sorted_70_15_15" 
    ## split = "specific_identity_random_80_20_M003"
    # split = "specific_identity_sorted_80_20_M003"

    ## b) set paths to the data (where you store MEAD), or use the default paths which are set in the config    
    # mead_input_dir = "/is/cluster/fast/rdanecek/data/mead_25fps/resampled_videos"
    # mead_processed_dir = "/is/cluster/fast/rdanecek/data/mead_25fps/"
    # mead_processed_subfolder = "processed"
    mead_input_dir = None
    mead_processed_dir = None
    mead_processed_subfolder = None


    fixed_overrides = []
    if dataset is not None: 
        fixed_overrides += [f'data/datasets={dataset}']
    if batching is not None:
        fixed_overrides += [f'+learning/batching@learning.batching={batching}']
    # if preprocessor is not None:
    #     fixed_overrides += [f'+model/preprocessor@model.preprocessor={preprocessor}']
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
    

    bid = 28
    max_price = 250
    
    for fmode in training_modes:
        overrides = fixed_overrides.copy()
        overrides += fmode

        cfg = script.configure(
            conf, overrides,
        )

        GlobalHydra.instance().clear()

        # if not submit_:
        #     with open_dict(cfg) as d:
        #         d.data.debug_mode = True
        #         tags = ["DEBUG_FROM_WORKSTATION"]
        #         if d.learning.tags is None:
        #             d.learning.tags = tags
        #         d.data.num_workers = 0

        if submit_:
            submit(cfg, bid=bid, max_price=max_price)
        else:
            script.train_model(cfg, resume_from_previous=False)



if __name__ == "__main__":
    submit_trainings()

