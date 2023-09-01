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
import gdl_apps.FaceReconstruction.training.train_face_reconstruction as script
import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
import time as t
import copy


def submit(cfg , bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"

    submission_dir_local_mount = "/is/cluster/work/rdanecek/face_reconstruction/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/face_reconstruction/submission"

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(time)) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(script.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[2].name / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    coarse_file = submission_folder_local / "config.yaml"
    # detail_file = submission_folder_local / "submission_detail_config.yaml"

    with open(coarse_file, 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)
    # with open(detail_file, 'w') as outfile:
    #     OmegaConf.save(config=cfg_detail, f=outfile)


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
    job_name = "train_face_reconstruction"
    cuda_capability_requirement = 7
    mem_gb = 40

    # args = f"{coarse_file.name} {detail_file.name}"
    args = f"{coarse_file.name}"
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
                       env="work38",
                       )
    # t.sleep(2)


def submit_trainings():
    from hydra.core.global_hydra import GlobalHydra

    coarse_conf = "mica_deca_pretrain"


    finetune_modes = [
        [ 
            [
            ]
        ],
    ]
    
    # QUESTIONS TO ANSWER 
    # 1. What batch size/sequence length is optimal for 32GB, 40GB and 80GB GPUs? 
    #  - 4 for 32GB, 6 for 40GB,  10 or maybe 12 for 80GB (these numbers are for 20-frame sequence length)
    # batch_sizes = [4, 6, 8, 10]
    batch_sizes = [12]
    new_finetune_modes = []

    for mode in finetune_modes: 
        for batch_size in batch_sizes:
            # num_workers = int(batch_size * 1)
            num_workers = 0
            mode = copy.deepcopy(mode)
            mode[0] += [ 
                f'learning.batching.batch_size_train={batch_size}',
                f'learning.batching.batch_size_val={batch_size}',
                f'learning.batching.batch_size_test={batch_size}',
                f'data.num_workers={num_workers}'
            ]
            new_finetune_modes += [mode]
    finetune_modes = new_finetune_modes

    # # 2. What occlusions probability is optimal? 
    # mouth_occlusions_probabilities = [0.2, 0.4, 0.6, 0.8, 1.0]
    # new_finetune_modes = []
    # for mode in finetune_modes:
    #     for mouth_occlusions_probability in mouth_occlusions_probabilities:
    #         mode = copy.deepcopy(mode)
    #         mode[0] += [ 
    #             F'data.occlusion_settings_train.occlusion_probability_mouth={mouth_occlusions_probability}'
    #         ]
    #         new_finetune_modes += [mode]
    # finetune_modes = new_finetune_modes

    # 3.
   

    fixed_overrides_coarse = [
        ## LRS3
        # # 'data.split=random_by_identity_pretrain_80_20',
        # 'data.split=specific_identity_80_20_pretrain/0af00UcTOSc', # training on a single identity 
        
        ## MEAD 
        'data/datasets=mead', 
        'data.split=specific_identity_sorted_80_20_M003',

        ## CelebV-HQ 
        # 'data/datasets=celebvhq_no_occlusion', # training on a single video (and therefore identity)
        # # 'data.split=specific_video_temporal_eknCAJ0ik8c_0_0_80_10_10',
        # 'data.split=specific_video_temporal_6jRVZQMKlxw_1_0_80_10_10',
        # 'data.preload_videos=true',
        # 'data.inflate_by_video_size=true',
    ]

    # config_pairs = []
    for fmode in finetune_modes:
        coarse_overrides = fixed_overrides_coarse.copy()
        coarse_overrides += fmode[0]

        # coarse_overrides += [emonet_weight_override]
        # detail_overrides += [emonet_weight_override]

        cfgs = script.configure(
            coarse_conf, coarse_overrides,
            # detail_conf, detail_overrides
        )
        cfgs = list(cfgs)

        GlobalHydra.instance().clear()
        # config_pairs += [cfgs]


        bid = 1000
        submit_ = False
        # submit_ = True
        if not submit_: 
            bs = 2
            seq_len = 10
        else:
            # bs = 4
            # seq_len = 20
            bs = 1
            # seq_len = 100
            seq_len = 80
        # cfgs[0].learning.batching.batch_size_train = bs
        # cfgs[0].learning.batching.batch_size_val = bs
        # cfgs[0].learning.batching.batch_size_test = bs
        # cfgs[0].learning.batching.sequence_length_train = seq_len
        # cfgs[0].learning.batching.sequence_length_test = seq_len
        # cfgs[0].learning.batching.sequence_length_val = seq_len
        # # cfgs[0].learning.batching.sequence_length_train = 10
        # # cfgs[0].learning.batching.sequence_length_test = 10
        # # cfgs[0].learning.batching.sequence_length_val = 10
        # cfgs[0].model.max_epochs = 200    
        # cfgs[0].model.val_check_interval = 1.0
        # cfgs[0].model.train_vis_frequency = 20
        # cfgs[0].model.val_vis_frequency = 10
        OmegaConf.set_struct(cfgs[0], False)
        with open_dict(cfgs[0]) as d:
            tags = ["INITIAL_SMALL_TESTS"]
            tags += ["EMOCA_LIKE"]
            # tags += ["EMOCA_LIKE_NO_EMO"]
            # tags = ["PREDICT_EJ"]
            # tags += ["PREDICT_ALL"]
            # tags += ["PREDICT_EJG"]
            # tags += ["PREDICT_EJGC"]
            # tags += ["PREDICT_G"]
            # tags += ["PREDICT_C"]
            # tags += ["PREDICT_GC"]
            # tags += ["EMOCA_REG"]
            if not submit_:
                tags += ["DEBUG_FROM_WORKSTATION"]
            if d.learning.tags is None:
                d.learning.tags = tags
        cfg = OmegaConf.to_container(cfgs[0])
        # if not cfgs[0].model.output.predict_shapecode: 
        #     if 'shape_reg' in cfg["learning"]["losses"]: 
        #         del cfg["learning"]["losses"]["shape_reg"]
        # if not cfgs[0].model.output.predict_expcode: 
        #     if 'expression_reg' in cfg["learning"]["losses"]: 
        #         del cfg["learning"]["losses"]["expression_reg"]
        # if not cfgs[0].model.output.predict_texcode: 
        #     if 'tex_reg' in cfg["learning"]["losses"]:
        #         del cfg["learning"]["losses"]["tex_reg"]
        # if not cfgs[0].model.output.predict_light: 
        #     if 'light_reg' in cfg["learning"]["losses"]:
        #         del cfg["learning"]["losses"]["light_reg"]

        cfgs[0] = OmegaConf.create(cfg)

        if submit_:
            submit(cfgs[0], bid=bid)
        else:
            script.train_model(cfgs[0], resume_from_previous=False)






if __name__ == "__main__":
    # default_main()
    submit_trainings()

