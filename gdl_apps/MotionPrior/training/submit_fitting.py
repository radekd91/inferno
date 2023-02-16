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
import omegaconf
from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import gdl_apps.MotionPrior.training.fitting_optimization as script
import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
import time as t
import copy
import sys
from munch import Munch, munchify
import yaml
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
):
    # cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    cluster_repo_path = user_config.cluster_repo_path

    # submission_dir_local_mount = "/is/cluster/work/rdanecek/talkinghead/submission"
    submission_dir_local_mount = user_config.submission_dir_local_mount
    # submission_dir_cluster_side = "/is/cluster/work/rdanecek/talkinghead/submission"
    submission_dir_cluster_side = user_config.submission_dir_cluster_side

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

    submission_folder_name = time + "_" + str(hash(time)+ random.randint(-10000,10000)) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(script.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[2].name / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    config_file = submission_folder_local / "config.yaml"

    # with open(config_file, 'w') as outfile:
    #     OmegaConf.save(config=DictConfig(cfg), f=outfile)

    # save the config as yaml 
    with open(config_file, 'w') as outfile:
        yaml.safe_dump(cfg, outfile, default_flow_style=False)

    # python_bin = 'python'
    # python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    python_bin = user_config.python_bin
    # username = 'rdanecek'
    username = user_config.username
    # gpu_mem_requirement_mb = cfg.learning.batching.gpu_memory_min_gb * 1024
    gpu_mem_requirement_mb = 35 * 1024
    # gpu_mem_requirement_mb_max = cfg.learning.batching.get('gpu_memory_max_gb' , None)
    # gpu_mem_requirement_mb_max = cfg.learning.batching.get('gpu_memory_max_gb' , None)
    gpu_mem_requirement_mb_max = 42 * 1024
    # gpu_mem_requirement_mb = None
    cpus =  2 
    # gpus = cfg.learning.batching.num_gpus
    gpus = 1
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_talking_head"
    # cuda_capability_requirement = 7
    cuda_capability_requirement = 7.5
    mem_gb = 20

    # args = f"{coarse_file.name} {detail_file.name}"
    args = f"{config_file.name}"

    if stage is not None and resume_from_previous is not None and force_new_location is not None:
        args += f" {stage} {int(resume_from_previous)} {int(force_new_location)}"
    elif stage is not None or resume_from_previous is not None or force_new_location is not None:
        raise ValueError("stage, resume_from_previous and force_new_location must be all None or all not None")

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
                    #    env="work38",
                       env="work38_clone",
                       )
    # t.sleep(2)


def submit_trainings():
    bid = 2000

    network_names = []
    tags = []

    # network_name += ["2023_02_08_14-51-54_6121455154279531419_L2lVqVae_Facef_AE"]
    # network_name = ["2023_02_14_12-49-15_-845824944828324001_L2lVqVae_Facef_VAE"]
    # network_name = ["2023_02_14_12-49-31_-5593838506145801374_L2lVqVae_Facef_VAE"]
    
    ## mead trained 
    # ae trained and validated on disjoint identities # still converging
    # network_name = ["2023_02_14_21-16-46_-2546611371074170025_L2lVqVae_MEADP_AE"]

    # big compression ablation on vocaset 
    network_names += ["2023_02_16_03-02-10_-8379697372179529502_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-52-44_7558156538991788724_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-51-15_-2205845311543205984_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-50-09_-3277176671750596762_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-50-35_640088189318919960_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-50-35_4763369704726118240_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-50-35_1688626326852598134_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-50-35_105235826568100794_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-50-35_-765558303620712667_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-50-35_-680498863291769361_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-50-35_-3709574333621241107_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-50-35_-296130785160551390_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-46-48_6459487296302032610_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-43-17_-5800843791440211079_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-42-31_5590378737107640522_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-42-31_-3922861794691504783_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-42-25_-3226766129449995539_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-38-41_-1862091702277836081_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-35-56_8500318720425862541_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-35-16_8206381458634718266_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-34-56_6482521921367462583_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-34-56_2897229355593051814_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-28-55_-300794710179378856_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-21-23_6710793448495429935_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-21-23_8934882131247866579_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-21-23_4626144352265278150_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-21-23_2935116619994305437_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-21-23_1018122986884262347_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-21-23_-9129307174013332328_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-21-23_-707874504920088102_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-21-23_-5452483093065417657_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_02-12-18_-8663630374053631856_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_01-59-08_-8733284731081021251_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_01-58-54_7546029715643808155_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_01-54-57_1005062065405368989_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_01-20-16_190501136390560445_L2lVqVae_Facef_VAE"]
    network_names += ["2023_02_16_00-43-05_303772265033390763_L2lVqVae_Facef_VAE"]


    for network_name in network_names:
        cfg = Munch()
        
        path_to_config = Path(f"/is/cluster/work/rdanecek/motion_prior/trainings/{network_name}/cfg.yaml")
        helper_config = OmegaConf.load(path_to_config)
        cfg.data = munchify(OmegaConf.to_container( helper_config.data))
        cfg.data.batching = Munch() 
        cfg.data.batching.batch_size_train = 1
        cfg.data.batching.batch_size_val = 1
        cfg.data.batching.batch_size_test = 1 
        # cfg.data.batching.sequence_length_train = 1
        # cfg.data.batching.sequence_length_val = 1
        # cfg.data.batching.sequence_length_test = 1

        # cfg.data.batching.sequence_length_train = 25
        # cfg.data.batching.sequence_length_val = 25
        # cfg.data.batching.sequence_length_test = 25
        cfg.data.batching.sequence_length_train = "all"
        cfg.data.batching.sequence_length_val = "all"
        cfg.data.batching.sequence_length_test = "all"

        # cfg.data.batching.sequence_length_train = 150
        # cfg.data.batching.sequence_length_val = 150
        # cfg.data.batching.sequence_length_test = 150

        cfg.model = munchify(OmegaConf.to_container( helper_config.model))
        cfg.model.path_to_config = str(path_to_config)

        cfg.losses = Munch() 

        # ## Emotion feature loss
        # emotion_feature_loss_cfg = {
        #     "weight": 1.0,
        #     "network_path": "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_05-15-38_-8198495972451127810_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early",
        #     # emo_feat_loss: mse_loss
        #     "emo_feat_loss": "masked_mse_loss",
        #     "trainable": False, 
        #     "normalize_features": False, 
        #     "target_method_image": "emoca",
        #     "mask_invalid": "mediapipe_landmarks", # frames with invalid mediapipe landmarks will be masked for loss computation
        # }
        # cfg.losses.emotion_loss = Munch(emotion_feature_loss_cfg)
        # cfg.losses.emotion_loss.from_source = False
        # cfg.losses.emotion_loss.active = False
        # # cfg.losses.emotion_loss.active = True
        # # cfg.losses.emotion_loss.weight = 1.0
        # # cfg.losses.emotion_loss.weight = 15.0
        # # cfg.losses.emotion_loss.weight = 50.0
        # cfg.losses.emotion_loss.weight = 150.0

        ## geometry loss 
        cfg.losses.geometry_loss = Munch()
        cfg.losses.geometry_loss.from_source = False
        cfg.losses.geometry_loss.active = True
        cfg.losses.geometry_loss.weight = 1000000.

        # cfg.losses.video_emotion_loss = Munch()
        # # TODO: experiment with different nets
        # cfg.losses.video_emotion_loss.video_network_folder = "/is/cluster/work/rdanecek/video_emotion_recognition/trainings/"
        # ## best transformer, 4 layers, 512 hidden size
        # cfg.losses.video_emotion_loss.video_network_name = "2023_01_09_12-42-15_7763968562013076567_VideoEmotionClassifier_MEADP_TSC_PE_Lnce"
        # ## gru, 4 layers, 512 hidden size
        # # cfg.losses.video_emotion_loss.video_network_name = "2023_01_09_12-44-24_-8682625798410410834_VideoEmotionClassifier_MEADP_GRUbi_nl-4_Lnce"
        # cfg.losses.video_emotion_loss.network_path = str(Path(cfg.losses.video_emotion_loss.video_network_folder) / cfg.losses.video_emotion_loss.video_network_name)
        # # cfg.losses.video_emotion_loss.from_source = False # set this on emotion_loss instead
        # cfg.losses.video_emotion_loss.active = True
        # # cfg.losses.video_emotion_loss.active = False
        # cfg.losses.video_emotion_loss.feature_extractor = "no"
        # cfg.losses.video_emotion_loss.metric = "mse"
        # # cfg.losses.video_emotion_loss.weight = 1000.0
        # cfg.losses.video_emotion_loss.weight = 100.0
        # # cfg.losses.video_emotion_loss.weight = 10.0
        # # cfg.losses.video_emotion_loss.weight = 5.0
        # # cfg.losses.video_emotion_loss.weight = 2.5
        # # cfg.losses.video_emotion_loss.weight = 1.0
        
        # video_emotion_loss_cfg.feat_extractor_cfg = "no"

        # cfg.losses.lip_reading_loss = munchify(OmegaConf.to_container(helper_config.learning.losses.lip_reading_loss))
        # cfg.losses.lip_reading_loss.from_source = True
        # # cfg.losses.lip_reading_loss.from_source = False
        # cfg.losses.lip_reading_loss.active = True
        # # cfg.losses.lip_reading_loss.active = False
        # cfg.losses.lip_reading_loss.weight = 100.00
        # # cfg.losses.lip_reading_loss.weight = 0
        # cfg.losses.expression_reg = Munch()
        # # cfg.losses.expression_reg.weight = 1.0
        # cfg.losses.expression_reg.weight = 1e-3
        # cfg.losses.expression_reg.active = True

        cfg.settings = Munch()
        # cfg.settings.optimize_exp = True
        # cfg.settings.optimize_jaw_pose = True

        # if cfg.settings.optimize_jaw_pose:
        #     cfg.losses.jaw_pose_reg = Munch()
        #     cfg.losses.jaw_pose_reg.weight = 1.0
        #     # cfg.losses.jaw_pose_reg.active = True
        #     cfg.losses.jaw_pose_reg.active = False
        #     cfg.losses.jaw_pose_reg.from_source = True
        #     # cfg.losses.jaw_pose_reg.from_source = False
        #     cfg.losses.jaw_pose_reg.input_space = 'aa'
        #     cfg.losses.jaw_pose_reg.output_space = '6d'

        cfg.settings.flame_cfg = munchify({
            "type": "flame",
            "flame_model_path": "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl",
            "n_shape": 100 ,
            # n_exp: 100,
            "n_exp": 50,
            "flame_lmk_embedding_path": "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy" ,
            "tex_type": "BFM",
            "tex_path": "/ps/scratch/rdanecek/data/FLAME/texture/FLAME_albedo_from_BFM.npz",
            "n_tex": 50,
        })

        # cfg.settings.renderer = munchify(OmegaConf.to_container(helper_config.model.renderer))

        cfg.optimizer = Munch()
        # cfg.optimizer.type = "adam"
        # cfg.optimizer.type = "sgd"
        cfg.optimizer.type = "lbfgs"
        # cfg.optimizer.lr = 1e-4
        # cfg.optimizer.lr = 1e-3
        # cfg.optimizer.lr = 1e-6
        # cfg.optimizer.lr = 1e-2
        # cfg.optimizer.lr = 1e-1
        cfg.optimizer.lr = 1.
        if cfg.optimizer.type == "lbfgs":
            cfg.optimizer.n_iter = 1000
        else:
            cfg.optimizer.n_iter = 10000
        # cfg.optimizer.n_iter = 100
        cfg.optimizer.patience = 50
        
        cfg.init = Munch()
        # cfg.init.source_sample_idx = 61 #
        # cfg.init.target_sample_idx = 58 #
        cfg.init.source_sample_idx = 0 # 
        cfg.init.target_sample_idx = 1 #
        cfg.init.geometry_type = 'EMOCA_v2_lr_mse_15_with_bfmtex'
        # cfg.init.geometry_type = 'spectre'
        # cfg.init.init = 'random'
        # cfg.init.init = 'source'
        cfg.init.latent_seq_init = 'zeros'
        # cfg.init.shape_from_source = True
        cfg.init.shape_from_source = False

        cfg.inout = Munch()
        cfg.inout.result_root = "/is/cluster/work/rdanecek/talkinghead/motion_prior_fitting"

        if submit_:
            submit(cfg, bid=bid)
        else:
            script.optimize(cfg)




if __name__ == "__main__":
    # default_main()
    submit_trainings()

