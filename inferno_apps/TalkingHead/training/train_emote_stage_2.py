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
# For comments or questions, please email us at emote@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""
import inferno_apps.TalkingHead.training.train_talking_head as script
from inferno_apps.TalkingHead.training.train_emote_stage_2 import submit, user_config
from inferno.utils.condor import execute_on_cluster
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, open_dict
from inferno.utils.other import get_path_to_assets


# submit_ = False
submit_ = True


def submit_reconfigured_trainings():
    from hydra.core.global_hydra import GlobalHydra
    ## 0) Set your data paths and result path
    # model_output_dir = None ## default from configs
    model_output_dir = str(get_path_to_assets().absolute() / "TalkingHead/trainings")
    
    ## 1) EMOTE base config for stage 2
    conf = "bertprior_wild_rendering_ex_vid" 

    tags = [] 

    ## 2) Set video emotion loss weights (both reconstruction and disentangled)
    w_emotion =  0.0000025
    w_emotion_dis = 0.0000025
    wl_multiplier = 10.0
    
    ## 3) Set lip-reading loss weights (set them relative to video emotion loss weights)
    w_lip_reading =  w_emotion * wl_multiplier
    w_lip_reading_dis = w_emotion_dis * wl_multiplier

    lr_emo_weights = [
        # (w_lip_reading,      w_lip_reading_dis,        w_emotion,         w_emotion_dis), ## good weight for EMOTE v1
        # (w_lip_reading,      w_lip_reading_dis,        w_emotion*1.2,         w_emotion_dis*1.2),
        # (w_lip_reading,      w_lip_reading_dis,        w_emotion*0.8,         w_emotion_dis*0.8), ## to strong for EMOTE v2
        # (w_lip_reading,      w_lip_reading_dis,        w_emotion*0.7,         w_emotion_dis*0.7),
        (w_lip_reading,      w_lip_reading_dis,        w_emotion*0.6,         w_emotion_dis*0.6),
        # (w_lip_reading,      w_lip_reading_dis,        w_emotion*0.5,         w_emotion_dis*0.5), ## ok for EMOTE v2
        # (w_lip_reading,      w_lip_reading_dis,        w_emotion*0.25,         w_emotion_dis*0.25),
        # (w_lip_reading,      w_lip_reading_dis,        w_emotion*0.1,         w_emotion_dis*0.1),
    ]

    ## 4) Set the paths to the models 
    path_to_talkinghead_models = "<YOUR_TALKINGHEAD_EXPERIMENT_FOLDER>"
    
    ##5) EMOTE-stage-1 model names to be finetuned  
    # resume_folders += ["<YOUR_EMOTE_STAGE_1_MODEL>"]


    ## 6) Set additional hyperparameters - these need to be the same as the for the model you're finetuning
    # use_shape = True
    use_shape = False ## EMOTE does not condition on SHAPE
    # use_identity = False 
    use_identity = True ## use identity one-hot as part of style
# 
    # style_operation = 'add'
    style_operation = 'cat'

    # use_real_video_for_reference = False
    use_real_video_for_emotion_reference = True, ## EMOTE computes the emotion loss against the real video

    use_real_video_for_lip_reference = False ## EMOTE computes the lip reading loss against the rendered pseudo-GT (it seems to work better)
    # use_real_video_for_lip_reference = True

    if use_real_video_for_lip_reference:
        tags += ["LIP_REAL_REF"]


    ## 7) set the video emotion network(s) to be used
    # path_to_video_emotion = get_path_to_assets() / "VideoEmotionRecognition" / "models"
    path_to_video_emotion = Path("/is/cluster/work/rdanecek/video_emotion_recognition/trainings/")
    video_emotion_networks = []
    video_emotion_networks += ["2023_04_12_15-14-44_6653765603916292046_VideoEmotionClassifier_MEADP__TSC_NPE_L_early"]
    # video_emotion_networks += ["<YOUR_OWN_VIDEO_EMOTION_NETWORK>"]
    
    
    ## 8) Add the hyperparameters to the config
    training_modes = [
        *[ ## lip reading and VIDEO emotion with exchange sweep and different disentanglement weights
            [
            # '+model.sequence_decoder.temporal_bias_type=faceformer_future', ## already in the original config
            f'learning.losses.lip_reading_loss.weight={wl}',
            f'learning.losses.lip_reading_loss_disentangled.weight={wld}',
            f'learning.losses.emotion_video_loss.weight={we}',
            f'learning.losses.emotion_video_loss_disentangled.weight={wed}',
            f'model.sequence_decoder.style_embedding.use_shape={use_shape}',
            f'model.sequence_decoder.style_embedding.gt_expression_identity={use_identity}',
            f'+model.sequence_decoder.style_op={style_operation}',
            ] 
            for wl, wld, we, wed in lr_emo_weights
        ],
    ]

    dataset = "mead_pseudo_gt"
    # reconstruction_type = "EMICA_mead_mp_lr_mse_15" ## old version of data used in EMOTE paper
    reconstruction_type = "EMICA-MEAD_flame2020" ## new version of data with much better reconstructions
    batching = "fixed_length_bs4_45gb"
    
    # if not submit_:
        # batching = "fixed_length_bs2" ## small for debugging on workstations

    if "rendering" in conf: 
        preprocessor = None
    else:
        preprocessor = "flame_tex"
        # preprocessor = None

    ### MEAD splits
    ## split = "random_70_15_15"
    ## split = "random_by_sequence_random_70_15_15" 
    # split = "random_by_sequence_sorted_70_15_15" 
    ## split = "random_by_identityV2_random_70_15_15" 
    split = "random_by_identityV2_sorted_70_15_15"  ## EMOTE split
    ## split = "specific_identity_random_80_20_M003"
    # split = "specific_identity_sorted_80_20_M003"
    ## split = "specific_identity_random_80_20_M005"
    # split = "specific_identity_sorted_80_20_M005"

    ## b) set paths to the data (where you store MEAD), or use the default paths which are set in the config    
    # mead_input_dir = "/is/cluster/fast/rdanecek/data/mead_25fps/resampled_videos"
    # mead_processed_dir = "/is/cluster/fast/rdanecek/data/mead_25fps/"
    # mead_processed_subfolder = "processed"
    mead_input_dir = None
    mead_processed_dir = None
    mead_processed_subfolder = None
    


    fixed_overrides = [
    ]

    fixed_overrides += [f'data/datasets={dataset}']
    fixed_overrides += [f'data.reconstruction_type={reconstruction_type}']
    if batching is not None:
        fixed_overrides += [f'+learning/batching@learning.batching={batching}']
    if preprocessor is not None:
        fixed_overrides += [f'+model/preprocessor@model.preprocessor={preprocessor}']
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
    
    bid = 30
    max_price = 450

    if not submit_:
        fixed_overrides += [
            'learning/logging=none',
        ]

    if use_real_video_for_emotion_reference or use_real_video_for_lip_reference: 
        fixed_overrides += [
            'data.read_video=True',
        ]

    for experiment_folder in resume_folders:
    
        fixed_overrides_backup = fixed_overrides.copy()

        for video_emo_net in video_emotion_networks:

            fixed_overrides = fixed_overrides_backup.copy()

            if video_emo_net is not None:
                fixed_overrides += [f'learning.losses.emotion_video_loss.network_path={str(path_to_video_emotion / video_emo_net)}']
                fixed_overrides += [f'learning.losses.emotion_video_loss.use_real_video_for_reference={str(use_real_video_for_emotion_reference)}']

                if "bertprior_wild_rendering_ex_vid" in conf:
                    fixed_overrides += [f'learning.losses.emotion_video_loss_disentangled.network_path={str(path_to_video_emotion / video_emo_net)}']
                    fixed_overrides += [f'learning.losses.emotion_video_loss_disentangled.use_real_video_for_reference={str(use_real_video_for_emotion_reference)}']
                    fixed_overrides += [f'learning.losses.emotion_video_loss_disentangled.target_method_image={str(reconstruction_type)}']
                    # fixed_overrides += ['model.max_epochs=2']
            
            if "bertprior_wild_rendering_ex" == conf:
                    fixed_overrides += [f'learning.losses.emotion_loss.use_real_video_for_reference={str(use_real_video_for_emotion_reference)}']
                    fixed_overrides += [f'learning.losses.emotion_loss_disentangled.use_real_video_for_reference={str(use_real_video_for_emotion_reference)}']
                    fixed_overrides += [f'learning.losses.emotion_loss_disentangled.target_method_image={str(reconstruction_type)}']
                    # fixed_overrides += ['model.max_epochs=2']

            if "bertprior_wild_rendering_ex" in conf:
                fixed_overrides += [f'+learning.losses.lip_reading_loss.use_real_video_for_reference={str(use_real_video_for_lip_reference)}']                
                fixed_overrides += [f'+learning.losses.lip_reading_loss_disentangled.use_real_video_for_reference={str(use_real_video_for_lip_reference)}']                

            for fmode in training_modes: ## loop over the different hyperparameter settings and launch the jobs
                
                overrides = fixed_overrides.copy()
                overrides += fmode

                cfg = script.configure(
                    conf, overrides,
                )

                GlobalHydra.instance().clear()
                original_config_file = Path(path_to_talkinghead_models) / experiment_folder / "cfg.yaml"

                with open(original_config_file, 'r') as f:
                    cfg_orig = OmegaConf.load(f)

                # the original split and the new split must be the same
                assert cfg_orig.data.split == cfg.data.split, \
                    "The original split and the new split must be the same but instead are: " + \
                    f"{cfg_orig.data.split} and {cfg.data.split}. This is probably not what you want."
                
                # the dataset class must be the same 
                assert cfg_orig.data.data_class == cfg.data.data_class, \
                    "The original dataset class and the new dataset class must be the same but instead are: " + \
                    f"{cfg_orig.data.data_class} and {cfg.data.data_class}. This is probably not what you want."

                # the dataset class must be the same 
                assert cfg_orig.data.reconstruction_type == cfg.data.reconstruction_type, \
                    "You most likely want the reconstruction type of the original and the to-be-fientuned model the same. Instead you have: " + \
                    f"{cfg_orig.data.reconstruction_type} and {cfg.data.reconstruction_type}. This is probably not what you want."


                if not submit_:
                    with open_dict(cfg) as d:
                        # d.data.num_workers = 0
                        d.data.num_workers = 1
                        tags += ["DEBUG_FROM_WORKSTATION"]
                        if d.learning.tags is None:
                            d.learning.tags = tags
                        d.data.preload_videos = False


                with open_dict(cfg) as d:
                    if d.learning.tags is None:
                        d.learning.tags = tags
                    
                    if video_emo_net is not None:
                        if 'emotion_video_loss' in d.learning.losses and 'network_path' in d.learning.losses.emotion_video_loss:
                            # load motion prior config
                            video_emo_cfg = OmegaConf.load(path_to_video_emotion / video_emo_net / "cfg.yaml")
                            d.learning.losses.emotion_video_loss.cfg = video_emo_cfg
                
                
                    ## this will set the checkpoint path to the original model (so that it will be loaded)
                    ## don't worry, the model will only be loaded but not overwritten with the finetuned model (the experiment will be forked 
                    ## to a new folder because force_new_location=True below)
                    d.inout = cfg_orig.inout

                    assert 'motion_prior' in d.model.sequence_decoder.keys(), "The original config must have a motion prior"
                    d.model.sequence_decoder.motion_prior = cfg_orig.model.sequence_decoder.motion_prior

                    if cfg_orig.model.sequence_encoder.feature_dim != cfg.model.sequence_encoder.feature_dim:
                        d.model.sequence_encoder.feature_dim = cfg_orig.model.sequence_encoder.feature_dim
                        print("Setting feature dim of sequence encoder to original value: ", cfg_orig.model.sequence_encoder.feature_dim)
                    if cfg_orig.model.sequence_decoder.feature_dim != cfg.model.sequence_decoder.feature_dim:
                        d.model.sequence_decoder.feature_dim = cfg_orig.model.sequence_decoder.feature_dim
                        print("Setting feature dim of sequence decoder to original value: ", cfg_orig.model.sequence_decoder.feature_dim)

                stage=0
                resume_from_previous=False
                force_new_location=True ## make sure you fork the experiment to a new folder
                if submit_:
                    submit(cfg, stage=stage, resume_from_previous=resume_from_previous, 
                           force_new_location=force_new_location, bid=bid, max_price=max_price,)
                else:
                    script.train_model(cfg, start_i=stage, resume_from_previous=resume_from_previous, force_new_location=force_new_location)



if __name__ == "__main__":
    submit_reconfigured_trainings()

