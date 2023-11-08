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
import inferno_apps.FaceReconstruction.training.train_face_reconstruction as script
import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
import time as t
import copy
import sys
import torch

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


def submit(cfg , bid=10):
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

    config_filename = submission_folder_local / "config.yaml"

    with open(config_filename, 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)

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
    job_name = "train_face_reconstruction"
    cuda_capability_requirement = 7
    # mem_gb = 60
    mem_gb = 70

    args = f"{config_filename.name}"
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


def submit_trainings():
    from hydra.core.global_hydra import GlobalHydra
    ### THIS IS AN EXAMPLE SCRIPT ON HOW TO TRAIN THE EMICA MODEL ON THE CELEBVTEXT DATASET
    ### PLEASE ADAPT THE SCRIPT TO YOUR NEEDS
    
    ## 0) is jaw rotation predicted by expression or shape branch? default: NO
    jaw = False # predicted by shape branch (more stable)
    # jaw = True
    jaw_str = ""
    if jaw:
        jaw_str = "_jaw"
        

    ## 1) WHICH STAGE are you training?
    
    ## 1a) pretrain stage (training everything (expect for MICA) with landmark losses), START here
    conf_default = f"emica{jaw_str}_pretrain_stage" 
    
    ## 1b) DECA stage (finetuning everything (expect for MICA) with differentiable rendering), train this AFTER pretrain stage
    # coarse_conf = f"emica{jaw_str}_pretrain_stage" 
    
    ## 1c) EMOCA stage (finetuning only expression branch with lip reading and emotion perceptual losses), train this AFTER DECA stage
    # coarse_conf = f"emica{jaw_str}_emoca_stage" 

    finetune_modes = [] 
    
    # 2) Choose your FLAME version
    ## DEFAULT OPTION - FLAME 2020
    flame2023 = False
    finetune_modes += [
        [ 
            [
            ]
        ],
    ]
    
    ## ## FLAME 2023
    # flame2023 = True
    # finetune_modes += [
    #     [ 
    #         [ 
    #             '+model/shape_model@model.shape_model=flametex2023',
    #             'model.face_encoder.encoders.mica_deca_encoder.encoders.mica_encoder.mica_model_path=MICA/model/mica_2023.tar',
    #         ]
    #     ], 
    # ]


    ## batch size
    batch_sizes = [20] 
    ring_size = 8

    new_finetune_modes = []

    if not submit_:
        batch_sizes = [4]
        ring_size = 4

    for mode in finetune_modes: 
        for batch_size in batch_sizes:
            # num_workers = int(batch_size * 1)
            # num_workers = 8
            num_workers = 10
            if not submit_:
                num_workers = 0
            mode = copy.deepcopy(mode)
            mode[0] += [ 
                f'learning.batching.batch_size_train={batch_size}',
                f'learning.batching.batch_size_val={batch_size}',
                f'learning.batching.batch_size_test={batch_size}',
                f'learning.batching.ring_size_train={ring_size}',
                f'learning.batching.ring_size_val={ring_size}',
                f'learning.batching.ring_size_test={ring_size}',
                f'data.num_workers={num_workers}'
            ]
            new_finetune_modes += [mode]
    finetune_modes = new_finetune_modes


    ## 3) DATASET OPTIONS 

    ## CelebV-Text 
    dataset_options = [
        # 'data/datasets=celebvtext',
        'data/datasets=celebvtext_occlusions', ## add artificial occlusions
        'data.split=random_70_15_15',
        'data/augmentations=default',
        # 'data/augmentations=none', ## no image augmentation
    ]

    fixed_overrides_coarse = []
    fixed_overrides_coarse += dataset_options

    if not submit_: 
        fixed_overrides_coarse += [
            'inout.output_dir=/is/cluster/work/rdanecek/face_reconstruction/debug_trainings/',
        ]

    # config_pairs = []
    for fmode in finetune_modes:
        conf_overrides = fixed_overrides_coarse.copy()
        conf_overrides += fmode[0]

        conf = script.configure(
            conf_default, conf_overrides,
        )

        GlobalHydra.instance().clear()
        init_from = None
        
        ## 4) LOAD PRETRAINED MODEL FROM the previous stage (if applicable)
        if "emica_deca_stage" in conf_default:
            if conf.data.data_class == "CelebVTextDataModule":
                ## flame 2020
                if not flame2023:
                    init_from = "<add your pretrained model's config>"
                else:
                    ## flame 2023
                    init_from = "<add your pretrained model's config>"
            else:
                raise ValueError(f"Unknown data class {conf.data.data_class}")
        elif "emica_emoca_stage" in conf_default:
            if conf.data.data_class == "CelebVTextDataModule":
                ## flame 2020
                if not flame2023:
                    init_from = "<add your pretrained model's config>"
                else:
                    ## flame 2023
                    init_from = "<add your pretrained model's config>"
            else:
                raise ValueError(f"Unknown data class {conf.data.data_class}")
            
        elif "emica_jaw_deca_stage" in conf_default:
            if conf.data.data_class == "CelebVTextDataModule":
                ## flame 2020
                if not flame2023:
                    init_from = "<add your pretrained model's config>"
                else:
                    ## flame 2023
                    init_from = "<add your pretrained model's config>"
            else: 
                raise ValueError(f"Unknown data class {conf.data.data_class}")
        elif "emica_jaw_emoca_stage" in conf_default:
            if conf.data.data_class == "MEADDataModule":
                ## flame 2020
                if not flame2023:
                    init_from = "<add your pretrained model's config>"
                else:
                    ## flame 2023
                    init_from = "<add your pretrained model's config>"
            elif conf.data.data_class == "CelebVTextDataModule":
                ## flame 2020
                if not flame2023:
                    init_from = "<add your pretrained model's config>"
                else:
                    ## flame 2023
                    init_from = "<add your pretrained model's config>"
            else: 
                raise ValueError(f"Unknown data class {conf.data.data_class}")


        bid = 150
        OmegaConf.set_struct(conf, False)
        with open_dict(conf) as d:
            tags = []
            tags += [conf_default]
            if not submit_:
                tags += ["DEBUG_FROM_WORKSTATION"]
                
            if '2023' in Path(conf.model.face_encoder.encoders.mica_deca_encoder.encoders.mica_encoder.mica_model_path).name: 
                tags += ["MICA_2023"]

            if d.learning.tags is None:
                d.learning.tags = tags
                
            ## 5) SET THE INIT_FROM PARAMETER (if applicable). This will load the pretrained model from the previous stage
            if init_from is not None:
                d.model.init_from = init_from

        conf = OmegaConf.create(cfg)

        if submit_:
            submit(conf, bid=bid)
        else:
            script.train_model(conf, resume_from_previous=False)


if __name__ == "__main__":
    submit_trainings()

