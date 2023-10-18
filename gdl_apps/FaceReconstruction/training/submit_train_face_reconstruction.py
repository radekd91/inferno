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
from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import gdl_apps.FaceReconstruction.training.train_face_reconstruction as script
import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
import time as t
import copy
import sys
import torch

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

    coarse_file = submission_folder_local / "config.yaml"
    # detail_file = submission_folder_local / "submission_detail_config.yaml"

    with open(coarse_file, 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)
    # with open(detail_file, 'w') as outfile:
    #     OmegaConf.save(config=cfg_detail, f=outfile)

    # python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    python_bin = user_config.python_bin
    # username = 'rdanecek'
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

    # args = f"{coarse_file.name} {detail_file.name}"
    args = f"{coarse_file.name}"
    # args = f"{str(Path(result_dir_cluster_side) / resume_folder)} 
    # {stage} {int(resume_from_previous)} {int(force_new_location)}"

    env="/is/cluster/fast/rdanecek/envs/work38_fast_clone" 

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


    # coarse_conf = "emica_jaw_pretrain_stage" 
    # coarse_conf = "emica_jaw_deca_stage"
    # coarse_conf = "emica_jaw_emoca_stage"

    # coarse_conf = "emica_pretrain_stage" 
    # coarse_conf = "emica_deca_stage"
    # coarse_conf = "emica_emoca_stage"

    # coarse_conf = "emica_jaw_pretrain_stage_swin"
    # coarse_conf = "emica_jaw_deca_stage_swin"
    # coarse_conf = "emica_jaw_emoca_stage_swin"

    # coarse_conf = "emica_pretrain_stage_swin"
    # coarse_conf = "emica_deca_stage_swin"
    # coarse_conf = "emica_emoca_stage_swin"

    coarse_conf = "emica_pretrain_stage_swintoken"
    # coarse_conf = "emica_deca_stage_swintoken"
    # coarse_conf = "emica_emoca_stage_swintoken"

    ## FLAME 2023, no jaw
    # coarse_conf = "emica_pretrain_stage_swin_no_jaw"
    # coarse_conf = "emica_deca_stage_swin_no_jaw"
    # coarse_conf = "emica_emoca_stage_swin_no_jaw"
    
    jaw = 'jaw' in coarse_conf
    swin = 'swin' in coarse_conf

    finetune_modes = [] 

    # finetune_modes += [
    #     [ 
    #         [
    #         ]
    #     ],
    # ]

    # # # ## FLAME 2023
    finetune_modes += [
        [ 
            [ 
                '+model/shape_model@model.shape_model=flametex2023',
                'model.face_encoder.encoders.mica_deca_encoder.encoders.mica_encoder.mica_model_path=MICA/model/mica_2023.tar',
            ]
        ], 
    ]

    #  ## FLAME 2023, no jaw
    # if 'emoca_stage' in coarse_conf: 
    #     encoder = 'emica_encoder_pretrain_swin_no_jaw' if swin else 'emica_encoder_pretrain_no_jaw'
    # else: 
    #     encoder = 'emica_encoder_train_swin_no_jaw' if swin else 'emica_encoder_train_no_jaw'
    # finetune_modes += [
    #     [ 
    #         [
    #             '+model/shape_model@model.shape_model=flametex2023_no_jaw',
    #             f'+model/face_encoder@model.face_encoder={encoder}',
    #             # 'model.face_encoder.encoders.mica_deca_encoder.encoders.deca_encoder.predicts.jawpose=-',
    #             'model.face_encoder.encoders.mica_deca_encoder.encoders.mica_encoder.mica_model_path=MICA/model/mica_2023.tar',
    #         ]
    #     ],
    # ]


    # good for resnet

    if not swin:
        batch_sizes = [20] 
        ring_size = 8
    else:
        # ## good for swin
        batch_sizes = [18] 
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

    ## DATASET OPTIONS 
    
    # # LRS3
    # dataset_options = [
    #     'data.split=random_by_identity_pretrain_80_20',
    #     'data.split=specific_identity_80_20_pretrain/0af00UcTOSc', # training on a single identity 
    # ]
   
    # ### MEAD 
    # dataset_options = [
    #     'data/datasets=mead', 
    #     # 'data/datasets=mead_occlusions', 
    #     # 'data.split=specific_identity_sorted_80_20_M003',
    #     'data.split=random_by_sequence_sorted_70_15_15',
    #     # 'data/augmentations=default',
    #     # 'data/augmentations=default_no_jpeg',
    #     'data/augmentations=none',
    # ]

    # # # CelebV-Text 
    dataset_options = [
        # 'data/datasets=celebvtext',
        'data/datasets=celebvtext_occlusions',
        'data.split=random_70_15_15',
        'data/augmentations=default',
    ]

    # ##  CelebV-HQ
    # dataset_options = [
    #     'data/datasets=celebvhq_no_occlusion', # training on a single video (and therefore identity)
    #     # 'data.split=specific_video_temporal_eknCAJ0ik8c_0_0_80_10_10',
    #     'data.split=specific_video_temporal_6jRVZQMKlxw_1_0_80_10_10',
    # ]

    fixed_overrides_coarse = []
    fixed_overrides_coarse += dataset_options

    if not submit_: 
        fixed_overrides_coarse += [
            'inout.output_dir=/is/cluster/work/rdanecek/face_reconstruction/debug_trainings/',
        ]

    # config_pairs = []
    for fmode in finetune_modes:
        coarse_overrides = fixed_overrides_coarse.copy()
        coarse_overrides += fmode[0]

        # coarse_overrides += [emonet_weight_override]
        # detail_overrides += [emonet_weight_override]

        conf = script.configure(
            coarse_conf, coarse_overrides,
            # detail_conf, detail_overrides
        )
        # cfgs = list(cfgs)

        GlobalHydra.instance().clear()
        # config_pairs += [cfgs]

        init_from = None
        if "emica_deca_stage" in coarse_conf:
            if conf.data.data_class == "MEADDataModule":
                if not swin:
                    # ## old flame, older training
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_09_22-58-35_-2919259285031416864_FaceReconstructionBase_MEADDEmicaEncoder_Aug/cfg.yaml"    

                    ### align images is True
                    # old flame, with augmentation
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_21_18-41-56_8051151870301285469_FaceReconstructionBase_MEADD_ResNet50_Pe_Aug/cfg.yaml"

                    ### old flame, no augmentation 
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_21_18-41-24_-2981073980088472269_FaceReconstructionBase_MEADD_ResNet50_Pe/cfg.yaml"
                    
                    # ## flame 2023, augmentation
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_21_18-41-02_1027982613487497446_FaceReconstructionBase_MEADD_ResNet50_Pe_Aug/cfg.yaml"

                    # # ## flame 2023, no augmentation
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_21_18-41-24_-49456108034774743_FaceReconstructionBase_MEADD_ResNet50_Pe/cfg.yaml"

                    ### align images is False
                    # old flame, with augmentation
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_21_18-22-55_-5440029370915161031_FaceReconstructionBase_MEADD_ResNet50_Pe_Aug/cfg.yaml"

                    ### old flame, no augmentation 
                    # init_from = "/cfg.yaml"
                    
                    # ## flame 2023, augmentation
                    init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_21_18-22-55_-2509059966034608004_FaceReconstructionBase_MEADD_ResNet50_Pe_Aug/cfg.yaml"

                    # # ## flame 2023, no augmentation
                    # init_from = "/cfg.yaml"

                else:
                    ## old flame
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_13_12-27-27_-4587188938616529107_FaceReconstructionBase_MEADD_Swin_Pe_Aug/cfg.yaml"
                    
                    ## flame 2023 with jaw
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_15_16-49-27_4353790877397815112_FaceReconstructionBase_MEADD_Swin_Pe_Aug/cfg.yaml"
                    ## flame 2023 without jaw
                    init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_15_16-49-38_1892552007605304224_FaceReconstructionBase_MEADD_Swin_Pe_Aug/cfg.yaml"
            elif conf.data.data_class == "lrs3":
                init_from = "todo"
            elif conf.data.data_class == "celebvhq":
                init_from = "todo"
            elif conf.data.data_class == "CelebVTextDataModule":
                if not swin:
                    ## old flame, older trainings
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_11_10-38-21_-5536342182286255904_FaceReconstructionBase_CelebEmicaEncoder_Aug/cfg.yaml"

                    ## align images is True
                    ## old flame, newer trainings
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_21_18-40-32_-1937368037397236846_FaceReconstructionBase_Celeb_ResNet50_Pe_Aug/cfg.yaml"

                    # flame 2023
                    init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_21_18-40-32_-3169766546583792807_FaceReconstructionBase_Celeb_ResNet50_Pe_Aug/cfg.yaml"
                else:
                    pass
            elif conf.data.data_class == "cmumosei":
                init_from = "todo"
            else:
                raise ValueError(f"Unknown data class {conf.data.data_class}")
        elif "emica_emoca_stage" in coarse_conf:
            if conf.data.data_class == "MEADDataModule":
                if not swin:
                    ## align is True
                    ## old FLAME, no augmentation
                    init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_10_17_22-38-04_8211836596115022807_FaceReconstructionBase_MEADD_ResNet50_Pe/cfg.yaml"
                    ## FLAME 2023, no augmentation
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_22_10-15-36_1106960685689912156_FaceReconstructionBase_MEADD_ResNet50_Pe/cfg.yaml"

                    ## old FLAME, augmentation 
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_22_10-14-14_-1685380080178585813_FaceReconstructionBase_MEADD_ResNet50_Pe_Aug/cfg.yaml"

                    ## FLAME 2023, augmentation
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_22_10-16-58_3117774512957757240_FaceReconstructionBase_MEADD_ResNet50_Pe_Aug/cfg.yaml"

                    ## align is False
                    ## old FLAME, augmentation
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_22_10-37-32_-3993590541843054075_FaceReconstructionBase_MEADD_ResNet50_Pe_Aug/cfg.yaml"

                    ## FLAME 2023, augmentation
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_22_10-37-10_1144309682740580447_FaceReconstructionBase_MEADD_ResNet50_Pe_Aug/cfg.yaml"

                else:
                    ## old flame
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_13_12-31-43_-2250655239062560418_FaceReconstructionBase_Celeb_Swin_Pe_Aug/cfg.yaml"
                    ## flame 2023 with jaw
                    init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_15_22-09-48_5056076481827759078_FaceReconstructionBase_MEADD_Swin_Pe_Aug/cfg.yaml"

                    ## flame 2023 without jaw
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_15_22-12-45_2968646682376037339_FaceReconstructionBase_MEADD_Swin_Pe_Aug/cfg.yaml"
            elif conf.data.data_class == "lrs3":
                init_from = "todo"
            elif conf.data.data_class == "celebvhq":
                init_from = "todo"
            elif conf.data.data_class == "CelebVTextDataModule":
                if not swin:
                    ## align is True
                    ## old FLAME, augmentation
                    # init_from = ""

                    ## FLAME 2023, augmentation
                    init_from = ""

                    ## align is False
                    ## old FLAME, augmentation
                    # init_from = ""

                    ## FLAME 2023, augmentation
                    # init_from = ""


                else: 
                    # #### mixed up backbones
                    # # # no relative
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_13_23-45-03_-9205535432843645416_FaceReconstructionBase_Celeb_Swin_Pe_Aug/cfg.yaml"
                    # # # relative 
                    # # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_13_23-44-48_-4367144990188847988_FaceReconstructionBase_Celeb_Swin_Pe_Aug/cfg.yaml"
                    
                    ## flame 2023 with jaw
                    init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_15_22-23-43_-7479990913435329425_FaceReconstructionBase_Celeb_Swin_Pe_Aug/cfg.yaml"

                    ## flame 2023 without jaw
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_15_22-25-57_179462316877087924_FaceReconstructionBase_Celeb_Swin_Pe_Aug/cfg.yaml"
                    
            elif conf.data.data_class == "cmumosei":
                init_from = "todo"
            else:
                raise ValueError(f"Unknown data class {conf.data.data_class}")

        elif "emica_jaw_deca_stage" in coarse_conf:
            if conf.data.data_class == "MEADDataModule":
                if not swin:
                    ## old flame
                    init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_13_11-33-41_-6795290146162890555_FaceReconstructionBase_MEADD_ResNet50_Pej_Aug/cfg.yaml"
                else: 
                    ## flame 2023
                    init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_15_19-33-39_-4790262499280179166_FaceReconstructionBase_MEADD_Swin_Pej_Aug/cfg.yaml"
            elif conf.data.data_class == "CelebVTextDataModule":
                if not swin:
                    raise ValueError("No pretrained model for swin")
                else: 
                    ## old flame
                    # init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_13_23-46-01_-1656100747510451883_FaceReconstructionBase_Celeb_Swin_Pej_Aug/cfg.yaml"
                    ## flame 2023 
                    init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_15_19-33-15_1591369140996179512_FaceReconstructionBase_Celeb_Swin_Pej_Aug/cfg.yaml"
            else: 
                raise ValueError(f"Unknown data class {conf.data.data_class}")
        elif "emica_jaw_emoca_stage" in coarse_conf:
            if conf.data.data_class == "MEADDataModule":
                if not swin:
                    init_from = ""
                    raise ValueError("No pretrained model for swin")
                else: 
                    # flame 2023
                    init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_16_17-52-54_-725881493246472517_FaceReconstructionBase_MEADD_Swin_Pej_Aug/cfg.yaml"
            elif conf.data.data_class == "CelebVTextDataModule":
                if not swin:
                    raise ValueError("No pretrained model for swin")
                else: 
                    ## flame 2023 
                    init_from = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_16_17-56-02_-5820293105518523409_FaceReconstructionBase_Celeb_Swin_Pej_Aug/cfg.yaml"
            else: 
                raise ValueError(f"Unknown data class {conf.data.data_class}")
                    


        bid = 150
        OmegaConf.set_struct(conf, False)
        with open_dict(conf) as d:
            tags = []
            tags += [coarse_conf]
            if not submit_:
                tags += ["DEBUG_FROM_WORKSTATION"]
            if '2023' in Path(conf.model.face_encoder.encoders.mica_deca_encoder.encoders.mica_encoder.mica_model_path).name: 
                tags += ["MICA_2023"]
            if "no_jaw" in Path(conf.model.shape_model.flame.flame_model_path).name: 
                # tags += ["FLAME_NO_JAW"]
                tags += ["FLAME_NO_JAW_v2"]
            # tags += ["LR_COS_SWEEP"]
            # tags += ["LR_MSE_SWEEP"]
            if d.learning.tags is None:
                d.learning.tags = tags
            if init_from is not None:
                d.model.init_from = init_from
        cfg = OmegaConf.to_container(conf)

        conf = OmegaConf.create(cfg)

        if submit_:
            submit(conf, bid=bid)
        else:
            script.train_model(conf, resume_from_previous=False)






if __name__ == "__main__":
    # default_main()
    submit_trainings()

