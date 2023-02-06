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
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""
from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import gdl_apps.MotionPrior.training.train_motion_prior as script
import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
import time as t
import copy


def submit(cfg , bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"

    submission_dir_local_mount = "/is/cluster/work/rdanecek/motion_prior/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/motion_prior/submission"

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
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg.learning.batching.gpu_memory_min_gb * 1024
    gpu_mem_requirement_mb_max = cfg.learning.batching.get('gpu_mem_requirement_mb_max', None)
    # gpu_mem_requirement_mb = None
    cpus = cfg.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.learning.batching.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_motion_prior"
    cuda_capability_requirement = 7
    mem_gb = 40

    # args = f"{coarse_file.name} {detail_file.name}"
    args = f"{config_file.name}"
    # args = f"{str(Path(result_dir_cluster_side) / resume_folder)} 
    # {stage} {int(resume_from_previous)} {int(force_new_location)}"

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
                       env="work38_clone",
                       )
    # t.sleep(2)


def submit_trainings():
    from hydra.core.global_hydra import GlobalHydra

    conf = "l2lvq-vae"
    # conf = "l2lvq-vae_geometry"

    training_modes = [
        [], # no modifications to defaut config
    ]

    dataset = "vocaset"
    # dataset = "mead_pseudo_gt"
    
    # batching = "fixed_length"
    # batching = "fixed_length_bs16_35gb"
    # batching = "fixed_length150_bs16_35gb"
    batching = "fixed_length_bs32_35gb"
    # batching = "fixed_length_bs64_35gb"

    # preprocessor = "emoca"
    preprocessor = "flame"
    # preprocessor = None

    split = None

    ### MEAD splits
    ## split = "random_70_15_15"
    ## split = "random_by_identity_random_70_15_15" 
    # split = "random_by_identity_sorted_70_15_15" 
    ## split = "random_by_identityV2_random_70_15_15" 
    # split = "random_by_identityV2_sorted_70_15_15" 
    ## split = "specific_identity_random_80_20_M003"
    # split = "specific_identity_sorted_80_20_M003"

    fixed_overrides = [
        # '+model.sequence_decoder.style_embedding=none',
        # '+model.sequence_decoder.temporal_bias_type=faceformer_future',
        # '+model.audio.droput_cfg={type: Dropout, p: 0.2}',
        # '+model.sequence_decoder.use_alignment_bias=False',
    ]

    if dataset is not None: 
        fixed_overrides += [f'data/datasets={dataset}']
    if batching is not None:
        fixed_overrides += [f'+learning/batching@learning.batching={batching}']
    if preprocessor is not None:
        fixed_overrides += [f'+model/preprocessor@model.preprocessor={preprocessor}']
    if split is not None:
        fixed_overrides += [f'data.split={split}']

    bid = 1000
    # submit_ = False
    submit_ = True
    
    # if not submit_:
    #     fixed_overrides += [
    #         '+learning.logging=none',
    #     ]


    # config_pairs = []
    for fmode in training_modes:
        overrides = fixed_overrides.copy()
        overrides += fmode

        cfg = script.configure(
            conf, overrides,
        )

        GlobalHydra.instance().clear()
        # config_pairs += [cfgs]

        # OmegaConf.set_struct(cfgs[0], False)
        if not submit_:
            with open_dict(cfg) as d:
                d.data.debug_mode = True
                tags = ["DEBUG_FROM_WORKSTATION"]
                if d.learning.tags is None:
                    d.learning.tags = tags
      
        if submit_:
            submit(cfg, bid=bid)
        else:
            script.train_model(cfg, resume_from_previous=False)






if __name__ == "__main__":
    # default_main()
    submit_trainings()
