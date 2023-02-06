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
import gdl_apps.MotionPrior.training.resume_motion_prior_training  as script
import datetime
from omegaconf import OmegaConf
import time as t
import random
from omegaconf import DictConfig, OmegaConf, open_dict 
import sys

# submit_ = False
submit_ = True

if submit_:
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


def submit(resume_folder,
           stage = 0,
           resume_from_previous = False,
           force_new_location = False,
           bid=10):
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

    # config_file = Path(submission_folder_local) / resume_folder / "cfg.yaml"
    config_file = Path(resume_folder) / "cfg.yaml"

    with open(config_file, 'r') as f:
        cfg = OmegaConf.load(f)

    python_bin = user_config.python_bin
    username = user_config.username
    gpu_mem_requirement_mb = cfg.learning.batching.gpu_memory_min_gb * 1024
    gpu_mem_requirement_mb_max = cfg.learning.batching.get('gpu_memory_max_gb' , None)
    # gpu_mem_requirement_mb = None
    cpus = cfg.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.learning.batching.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_motion_prior"
    cuda_capability_requirement = 7
    # mem_gb = 16
    mem_gb = 30
    args = f"{str(config_file.parent)}"

    args += f" {stage} {int(resume_from_previous)} {int(force_new_location)}"

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
                    #    env="work38",
                       env="work38_clone",
                       )
    t.sleep(1)


def resume_motion_prior_on_cluster():
    root = "/is/cluster/work/rdanecek/motion_prior/trainings/"
    
    model_names = []
    model_names += ["2023_02_05_19-50-07_-5643053022395030803_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-47-04_-6204082330817104839_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-46-40_-8068289807722994107_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-45-15_1020392668976619472_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-45-15_-8493098869856430542_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-45-07_3708617454039099621_L2lVqVae_Facef"]
    ## model_names += ["2023_02_05_19-44-51_4819075661105074885_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-44-51_-4469641301695294897_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-43-35_3322208019009558883_L2lVqVae_Facef"]

    bid = 1000

    # # continue training
    # stage = 0 
    # resume_from_previous = False
    # force_new_location = False

    # # ## test 
    stage = 1
    resume_from_previous = True
    force_new_location = False

    for model_folder in model_names:
        if submit_:
            submit(root + model_folder, stage, resume_from_previous, force_new_location, bid=bid)
        else: 
            script.resume_training(root + model_folder, stage, resume_from_previous, force_new_location)


if __name__ == "__main__":
    resume_motion_prior_on_cluster()

