
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
import gdl_apps.FaceReconstruction.training.resume_face_reconstruction_training as script
import datetime
from omegaconf import OmegaConf
import time as t
import random
from omegaconf import DictConfig, OmegaConf, open_dict 
import sys
import shutil

submit_ = False
# submit_ = True

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
           bid=10, 
           max_price=None,
           ):
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
    cfg = cfg.coarse

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
    cuda_capability_requirement = 7
    # mem_gb = 16
    if stage == 1:
        mem_gb = 30
    else:
        mem_gb = 60
    args = f"{str(config_file.parent)}"

    args += f" {stage} {int(resume_from_previous)} {int(force_new_location)}"

    #    env="work38",
    #    env="work38_clone",
    # env = "/is/cluster/fast/rdanecek/envs/work38_fast" 
    env = "/is/cluster/fast/rdanecek/envs/work38_fast_clone"
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
    t.sleep(1)


def train_emodeca_on_cluster():

    root = "/is/cluster/work/rdanecek/face_reconstruction/trainings/"
    

    resume_folders = []
    # resume_folders += ['2023_09_14_12-08-02_803026180622496885_FaceReconstructionBase_Celeb_ResNet50_Pe_Aug']
    # bid = 2000
    bid = 150
    # bid = 28
    # max_price = 250
    max_price = 200

    # # # continue training
    stage = 0 
    resume_from_previous = False
    force_new_location = False

    # # ## test 
    # stage = 1
    # resume_from_previous = True
    # force_new_location = False

    # delete_rec_videos = False
    # # delete_cond_videos = True
    # delete_cond_videos = False

    for resume_folder in resume_folders:

        # if stage == 1: 
        #     folders_to_delete = []
        #     if delete_rec_videos:
        #         folders_to_delete += [Path(root) / resume_folder / "videos" / "train", Path(root) / resume_folder / "videos" / "val"]
        #     if delete_cond_videos: 
        #         folders_to_delete += [Path(root) / resume_folder / "videos" / "train_cond", Path(root) / resume_folder / "videos" / "val_cond"]

        #     for folder in folders_to_delete:
        #         if folder.exists():
        #             print(f"deleting {folder}")
        #             shutil.rmtree(folder)                

        if submit_:
            submit(root + resume_folder, stage, resume_from_previous, force_new_location, bid=bid, max_price=max_price)
        else: 
            script.resume_training(root + resume_folder, stage, resume_from_previous, force_new_location)


if __name__ == "__main__":
    train_emodeca_on_cluster()

