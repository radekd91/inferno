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
import random


def submit(cfg , bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"

    submission_dir_local_mount = "/is/cluster/work/rdanecek/motion_prior/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/motion_prior/submission"

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
    # mem_gb = 40
    mem_gb = 45

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

    ##conf = "l2l-ae"
    # conf = "l2l-ae_geometry"
    ## conf = "l2l-ae_geometry_fs"
    ## conf = "l2lvq-vae"
    # conf = "l2lvq-vae_geometry"
    ## conf = "l2lvq-vae_no_flame"
    conf = "l2l-vae_geometry"
    # conf = "l2l-dvae_geometry"
    # conf = "codetalker_vq-vae_geometry"
    ## conf = "codetalker_vq-vae"
    ## conf = "codetalker_vq-vae_no_flame"
    ## conf = "l2l-ae_deepphase_geometry"
    # conf = "deepphase-ae_geometry"

    tags = []
    # tags += ['QUANT_FACTOR']
    # tags += ['NUM_LAYERS']
    # tags += ['ZERO_INIT']
    # tags += ['CODEBOOK_SIZE']
    # tags += ['NO_FLAME']
    # tags += ['NO_CONV']
    # tags += ['CODEBOOK_LOSSES']
    # tags += ['KL']
    # tags += ['LATENT_SIZE']
    # tags += ['COMPRESSION']
    tags += ['REC_TYPES']

    if "l2l" in conf:
        training_modes = [
            # [], # no modifications to defaut config

            # [
            #    '+model/sequence_decoder@model.sequence_decoder=l2l_decoder_zero_init',  
            # ],

            [
            '+model/sequence_decoder@model.sequence_decoder=l2l_decoder_post_proj',  
            ],

            # [
            #    '+model/sequence_decoder@model.sequence_decoder=l2l_decoder_post_proj_no_conv',  
            # ],
        ]
    else: 
        training_modes = [
            [], # no modifications to defaut config
        ]

    dataset = "vocaset"
    # dataset = "vocaset_one_person"
    # dataset = "mead_pseudo_gt"
    
    # batching = "fixed_length"
    # batching = "fixed_length_bs16_35gb"
    # batching = "fixed_length150_bs16_35gb"
    batching = "fixed_length_bs32_35gb"
    # batching = "fixed_length_bs64_35gb"

    if conf in ["l2lvq-vae_no_flame", "vae_no_flame"] and dataset in ["vocaset", "vocaset_one_person"]: 
        preprocessor = None
    else:
        # preprocessor = "emoca"
        preprocessor = "flame"
        # preprocessor = None

    if dataset == "vocaset_one_person": 
        tags += ['ONE_PERSON']

    split = None

    if dataset == "mead_pseudo_gt":
        ### MEAD splits
        ## split = "random_70_15_15"
        ## split = "random_by_sequence_random_70_15_15" 
        # split = "random_by_sequence_sorted_70_15_15" 
        ## split = "random_by_identityV2_random_70_15_15" 
        split = "random_by_identityV2_sorted_70_15_15" 
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
    
    if not submit_:
        fixed_overrides += [
            '+learning.logging=none',
        ]


    # config_pairs = []
    for fmode in training_modes:
        overrides = fixed_overrides.copy()
        overrides += fmode

        # num_layer_list = [None] # defeault 
        # num_layer_list = [1, 2,  4,  6,  8, 12]
        # num_layer_list = [1, 2,  4,  6,  8]
        # num_layer_list = [1, 2, 4]
        # num_layer_list = [4]
        num_layer_list = [1]
        for num_layers in num_layer_list:
            if num_layers is not None:
                overrides += ['model.sequence_encoder.num_layers=' + str(num_layers)]
                overrides += ['model.sequence_decoder.num_layers=' + str(num_layers)]
        

            quant_factor_list = [None] # 
            # quant_factor_list = [1, 2, 3, 4, 5]
            # quant_factor_list = [2, 3, 4]
            # quant_factor_list = [0]
            # quant_factor_list = [5]

            # feature_dims = [None] # default
            # feature_dims = [16, 32, 64, 128, 256]
            feature_dims = [16, 32, 64, 128]
            # feature_dims = [16, 64, 128]
            for feature_dim in feature_dims:
                if feature_dim is not None:
                    overrides += ['model.sequence_encoder.feature_dim=' + str(feature_dim)]
                    overrides += ['model.sequence_decoder.feature_dim=' + str(feature_dim)]


                for quant_factor in quant_factor_list:
                    if quant_factor is not None:
                        overrides += ['model.sizes.quant_factor=' + str(quant_factor)]


                    codebook_size_list = [None] # defeault
                    # codebook_size_list = [ 512, 1024] 

                    for codebook_size in codebook_size_list:
                        if codebook_size is not None:
                            overrides += ['model.quantizer.codebook_size=' + str(codebook_size)]

                        codebook_losses = (0.25, 1.0)
                        codebook_loss_factors = [None]
                        # codebook_loss_factors = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
                        # codebook_loss_factors = [5.0, 10.0, 50.0, 100.0,]

                        for ci, codebook_loss in enumerate(codebook_loss_factors):
                            if codebook_loss is not None:
                                overrides += ['learning.losses.codebook_alignment.weight=' + str(codebook_loss * codebook_losses[0])]
                                overrides += ['learning.losses.codebook_commitment.weight=' + str(codebook_loss * codebook_losses[1])]

                            # kl_weights = [None]
                            # kl_weights = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
                            kl_weights = [0.001, 0.005, 0.01, 0.05]
                            # kl_weights = [ 0,5]
                            for kl_weight in kl_weights:
                                if kl_weight is not None:
                                    overrides += ['learning.losses.kl_divergence.weight=' + str(kl_weight)]

                                cfg = script.configure(
                                    conf, overrides,
                                )

                                GlobalHydra.instance().clear()
                                # config_pairs += [cfgs]

                                # OmegaConf.set_struct(cfgs[0], False)

                                with open_dict(cfg) as d:
                                    if not submit_:
                                        d.data.debug_mode = True
                                        tags += ["DEBUG_FROM_WORKSTATION"]
                                    if d.learning.tags is None:
                                        d.learning.tags = tags
                            
                                if submit_:
                                    submit(cfg, bid=bid)
                                else:
                                    script.train_model(cfg, resume_from_previous=False)






if __name__ == "__main__":
    # default_main()
    submit_trainings()

