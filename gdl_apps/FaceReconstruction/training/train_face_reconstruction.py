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


from gdl_apps.FaceReconstruction.training.training_pass import ( 
    single_stage_training_pass, 
    get_checkpoint_with_kwargs, 
    create_logger, 
    )
from gdl.models.FaceReconstruction.FaceRecBase import FaceReconstructionBase
from gdl.datasets.LRS3DataModule import LRS3DataModule
from gdl.datasets.CelebVHQDataModule import CelebVHQDataModule
from gdl.datasets.MEADDataModule import MEADDataModule
from gdl.datasets.CelebVTextDataModule import CelebVTextDataModule

from gdl.datasets.CombinedDataModule import CombinedDataModule
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import datetime
from gdl.utils.other import class_from_str
import torch 
from munch import Munch, munchify
# torch.autograd.set_detect_anomaly(True)

project_name = 'FaceReconstruction'


def create_single_dm(cfg, data_class):
    if 'augmentation' in cfg.data.keys() and len(cfg.data.augmentation) > 0:
        augmentation = munchify(OmegaConf.to_container(cfg.data.augmentation))
    else:
        augmentation = None

    occlusion_settings_train = OmegaConf.to_container(cfg.data.occlusion_settings_train) if 'occlusion_settings_train' in cfg.data.keys() else None 
    occlusion_settings_val = OmegaConf.to_container(cfg.data.occlusion_settings_val) if 'occlusion_settings_val' in cfg.data.keys() else None
    occlusion_settings_test = OmegaConf.to_container(cfg.data.occlusion_settings_test) if 'occlusion_settings_test' in cfg.data.keys() else None

    if data_class == "CelebVHQDataModule":
        # condition_source, condition_settings = get_condition_string_from_config(cfg)
        dm = CelebVHQDataModule(
                cfg.data.input_dir, 
                cfg.data.output_dir, 
                processed_subfolder=cfg.data.processed_subfolder, 
                face_detector=cfg.data.face_detector,
                landmarks_from=cfg.data.get('landmarks_from', None),
                face_detector_threshold=cfg.data.face_detector_threshold, 
                image_size=cfg.data.image_size, 
                scale=cfg.data.scale, 
                batch_size_train=cfg.learning.batching.batch_size_train,
                batch_size_val=cfg.learning.batching.batch_size_val, 
                batch_size_test=cfg.learning.batching.batch_size_test, 
                sequence_length_train=cfg.learning.batching.sequence_length_train or cfg.learning.batching.ring_size_train, 
                sequence_length_val=cfg.learning.batching.sequence_length_val or cfg.learning.batching.ring_size_val, 
                sequence_length_test=cfg.learning.batching.sequence_length_test or cfg.learning.batching.ring_size_test, 
                occlusion_settings_train = occlusion_settings_train,
                occlusion_settings_val = occlusion_settings_val,
                occlusion_settings_test = occlusion_settings_test,
                split = cfg.data.split,
                num_workers=cfg.data.num_workers,
                include_processed_audio = cfg.data.include_processed_audio,
                include_raw_audio = cfg.data.include_raw_audio,
                drop_last=cfg.data.drop_last,
                ## end args of FaceVideoDataModule
                ## begin CelebVHQDataModule specific params
                training_sampler=cfg.data.training_sampler,
                landmark_types = cfg.data.landmark_types,
                landmark_sources=cfg.data.landmark_sources,
                segmentation_source=cfg.data.segmentation_source,
                segmentation_type = cfg.data.segmentation_type,
                inflate_by_video_size = cfg.data.inflate_by_video_size,
                preload_videos = cfg.data.preload_videos,
                # test_condition_source=condition_source,
                # test_condition_settings=condition_settings,
                
                read_video=cfg.data.get('read_video', True),
                read_audio=cfg.data.get('read_audio', False),
                align_images = cfg.data.get('align_images', True),
                # reconstruction_type=cfg.data.get('reconstruction_type', None),
                # return_appearance=cfg.data.get('return_appearance', None),
                # average_shape_decode=cfg.data.get('average_shape_decode', None),
                # emotion_type=cfg.data.get('emotion_type', None),
                # return_emotion_feature=cfg.data.get('return_emotion_feature', None),
                return_mica_images=cfg.data.get('return_mica_images', False),
                augmentation=augmentation,
        )
        dataset_name = "CelebVHQ"
    elif data_class == "LRS3DataModule":
        # condition_source, condition_settings = get_condition_string_from_config(cfg)
        dm = LRS3DataModule(
                cfg.data.input_dir, 
                cfg.data.output_dir, 
                processed_subfolder=cfg.data.processed_subfolder, 
                face_detector=cfg.data.face_detector,
                landmarks_from=cfg.data.get('landmarks_from', None),
                face_detector_threshold=cfg.data.face_detector_threshold, 
                image_size=cfg.data.image_size, 
                scale=cfg.data.scale, 
                batch_size_train=cfg.learning.batching.batch_size_train,
                batch_size_val=cfg.learning.batching.batch_size_val, 
                batch_size_test=cfg.learning.batching.batch_size_test, 
                sequence_length_train=cfg.learning.batching.sequence_length_train or cfg.learning.batching.ring_size_train, 
                sequence_length_val=cfg.learning.batching.sequence_length_val or cfg.learning.batching.ring_size_val, 
                sequence_length_test=cfg.learning.batching.sequence_length_test or cfg.learning.batching.ring_size_test, 
                occlusion_settings_train = occlusion_settings_train,
                occlusion_settings_val = occlusion_settings_val,
                occlusion_settings_test = occlusion_settings_test,
                
                split = cfg.data.split,
                num_workers=cfg.data.num_workers,
                # include_processed_audio = cfg.data.include_processed_audio,
                # include_raw_audio = cfg.data.include_raw_audio,
                drop_last=cfg.data.drop_last,
                ## end args of FaceVideoDataModule
                ## begin CelebVHQDataModule specific params
                # training_sampler=cfg.data.training_sampler,
                landmark_types = cfg.data.landmark_types,
                landmark_sources=cfg.data.landmark_sources,
                segmentation_source=cfg.data.segmentation_source,
                segmentation_type = cfg.data.segmentation_type,
                include_processed_audio = cfg.data.include_processed_audio,
                include_raw_audio = cfg.data.include_raw_audio,
                inflate_by_video_size = cfg.data.inflate_by_video_size,
                preload_videos = cfg.data.preload_videos,
                align_images = cfg.data.get('align_images', True),
                # test_condition_source=condition_source,
                # test_condition_settings=condition_settings,
                # read_video=cfg.data.get('read_video', True),
                # reconstruction_type=cfg.data.get('reconstruction_type', None),
                # return_appearance=cfg.data.get('return_appearance', None),
                # average_shape_decode=cfg.data.get('average_shape_decode', None),
                # emotion_type=cfg.data.get('emotion_type', None),
                # return_emotion_feature=cfg.data.get('return_emotion_feature', None),
                return_mica_images=cfg.data.get('return_mica_images', False),
                augmentation=augmentation,
        )

        dataset_name = "LRS3"
    elif data_class == "MEADDataModule":
        # condition_source, condition_settings = get_condition_string_from_config(cfg)
        dm = MEADDataModule(
                cfg.data.input_dir, 
                cfg.data.output_dir, 
                processed_subfolder=cfg.data.processed_subfolder, 
                face_detector=cfg.data.face_detector,
                landmarks_from=cfg.data.get('landmarks_from', None),
                face_detector_threshold=cfg.data.face_detector_threshold, 
                image_size=cfg.data.image_size, 
                scale=cfg.data.scale, 
                batch_size_train=cfg.learning.batching.batch_size_train,
                batch_size_val=cfg.learning.batching.batch_size_val, 
                batch_size_test=cfg.learning.batching.batch_size_test, 
                sequence_length_train=cfg.learning.batching.sequence_length_train or cfg.learning.batching.ring_size_train, 
                sequence_length_val=cfg.learning.batching.sequence_length_val or cfg.learning.batching.ring_size_val, 
                sequence_length_test=cfg.learning.batching.sequence_length_test or cfg.learning.batching.ring_size_test, 
                occlusion_settings_train = occlusion_settings_train,
                occlusion_settings_val = occlusion_settings_val,
                occlusion_settings_test = occlusion_settings_test,
                split = cfg.data.split,
                num_workers=cfg.data.num_workers,
                include_processed_audio = cfg.data.include_processed_audio,
                include_raw_audio = cfg.data.include_raw_audio,
                drop_last=cfg.data.drop_last,
                ## end args of FaceVideoDataModule
                ## begin MEADPseudo3DDM specific params
                # training_sampler=cfg.data.training_sampler,
                landmark_types = cfg.data.landmark_types,
                landmark_sources=cfg.data.landmark_sources,
                segmentation_source=cfg.data.segmentation_source,
                segmentation_type = cfg.data.segmentation_type,
                inflate_by_video_size = cfg.data.inflate_by_video_size,
                preload_videos = cfg.data.preload_videos,
                # test_condition_source=condition_source,
                # test_condition_settings=condition_settings,
                read_video=cfg.data.get('read_video', True),
                read_audio=cfg.data.get('read_audio', False),
                align_images = cfg.data.get('align_images', True),
                # read_audio=cfg.data.get('read_audio', True),
                # reconstruction_type=cfg.data.get('reconstruction_type', None),
                # return_appearance=cfg.data.get('return_appearance', None),
                # average_shape_decode=cfg.data.get('average_shape_decode', None),
                # emotion_type=cfg.data.get('emotion_type', None),
                # return_emotion_feature=cfg.data.get('return_emotion_feature', None),
                # shuffle_validation=cfg.model.get('disentangle_type', False) == 'condition_exchange',
                return_mica_images=cfg.data.get('return_mica_images', False),
                augmentation=augmentation,
        )
        dataset_name = "MEAD"
    elif data_class == "CelebVTextDataModule":
        dm = CelebVTextDataModule(
            cfg.data.input_dir, 
            cfg.data.output_dir, 
            processed_subfolder=cfg.data.processed_subfolder, 
            face_detector=cfg.data.face_detector,
            landmarks_from=cfg.data.get('landmarks_from', None),
            face_detector_threshold=cfg.data.face_detector_threshold, 
            image_size=cfg.data.image_size, 
            scale=cfg.data.scale, 
            batch_size_train=cfg.learning.batching.batch_size_train,
            batch_size_val=cfg.learning.batching.batch_size_val, 
            batch_size_test=cfg.learning.batching.batch_size_test, 
            sequence_length_train=cfg.learning.batching.sequence_length_train or cfg.learning.batching.ring_size_train, 
            sequence_length_val=cfg.learning.batching.sequence_length_val or cfg.learning.batching.ring_size_val, 
            sequence_length_test=cfg.learning.batching.sequence_length_test or cfg.learning.batching.ring_size_test, 
            occlusion_settings_train = occlusion_settings_train,
            occlusion_settings_val = occlusion_settings_val,
            occlusion_settings_test = occlusion_settings_test,
            split = cfg.data.split,
            num_workers=cfg.data.num_workers,
            # include_processed_audio = cfg.data.include_processed_audio,
            # include_raw_audio = cfg.data.include_raw_audio,
            drop_last=cfg.data.drop_last,
            ## end args of FaceVideoDataModule
            ## begin CelebVTextDataModule specific params
            # training_sampler=cfg.data.training_sampler,
            landmark_types = cfg.data.landmark_types,
            landmark_sources=cfg.data.landmark_sources,
            segmentation_source=cfg.data.segmentation_source,
            segmentation_type = cfg.data.segmentation_type,
            include_processed_audio = cfg.data.include_processed_audio,
            include_raw_audio = cfg.data.include_raw_audio,
            inflate_by_video_size = cfg.data.inflate_by_video_size,
            preload_videos = cfg.data.preload_videos,
            align_images = cfg.data.get('align_images', False),
            read_video=cfg.data.get('read_video', True),
            read_audio=cfg.data.get('read_audio', False),
            # reconstruction_type=cfg.data.get('reconstruction_type', None),
            # return_appearance=cfg.data.get('return_appearance', None),
            # average_shape_decode=cfg.data.get('average_shape_decode', None),
            # emotion_type=cfg.data.get('emotion_type', None),
            # return_emotion_feature=cfg.data.get('return_emotion_feature', None),
            return_mica_images=cfg.data.get('return_mica_images', False),
            augmentation=augmentation,
        )
        dataset_name = "CelebVT"
    else:
        raise ValueError(f"Unknown data class: {data_class}")

    return dm, dataset_name

def prepare_data(cfg):
    data_class =  cfg.data.data_class
    # if data_class == 'CombinedDataModule':
    #     data_classes =  cfg.data.data_classes
    #     dms = {}
    #     for data_class in data_classes:
    #         dm, sequence_name = create_single_dm(cfg, data_class)
    #         dms[data_class] = dm
    #     data_class_weights = OmegaConf.to_container(cfg.data.data_class_weights) if 'data_class_weights' in cfg.data.keys() else None
    #     if data_class_weights is not None:
    #         data_class_weights = dict(zip(data_classes, data_class_weights))
    #     dm = CombinedDataModule(dms, data_class_weights)
    #     sequence_name = "ComboNet"
    # else:
    dm, sequence_name = create_single_dm(cfg, data_class)
    return dm, sequence_name




def create_experiment_name(cfg_coarse, version=0):
    # experiment_name = "ExpDECA"
    experiment_name = cfg_coarse.model.pl_module_class
    if version <= 2:

        if cfg_coarse.data.data_class:
            experiment_name += '_' + cfg_coarse.data.data_class[:5]

        face_encoder_name = cfg_coarse.model.face_encoder.type
        experiment_name += face_encoder_name

        if 'augmentation' in cfg_coarse.data.keys() and len(cfg_coarse.data.augmentation) > 0:
            experiment_name += "_Aug"

        # if cfg_coarse.data.occlusion_settings_train.occlusion_length == 0:
        #     experiment_name += "_noOcc"

        if hasattr(cfg_coarse.learning, 'early_stopping') and cfg_coarse.learning.early_stopping: # \
            # and hasattr(cfg_detail.learning, 'early_stopping') and cfg_detail.learning.early_stopping
            experiment_name += "_early"

    return experiment_name


def train_model(cfg_coarse, 
                # cfg_detail=None, 
                start_i=-1, 
                resume_from_previous = True,
               force_new_location=False):
    # configs = [cfg_coarse, cfg_coarse, cfg_detail, cfg_detail]
    # stages = ["train", "test", "train", "test"]
    # stages_prefixes = ["", "", "", ""]
    configs = [cfg_coarse, cfg_coarse]
    stages = ["train", "test"]
    stages_prefixes = ["", "", ]

    # if cfg_detail is not None:
    #     configs  += [cfg_detail, cfg_detail]
    #     stages += ["train", "test"]
    #     stages_prefixes += ["", "", ]

    init_from = cfg_coarse.model.get('init_from', None)
    if init_from is not None:
        # load the cfg from init_from 
        resume_i = start_i - 1 
        init_from_cfg = OmegaConf.load(init_from)
        checkpoint_mode = init_from_cfg.coarse.learning.checkpoint_after_training  # loads latest or best based on cfg
        checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(init_from_cfg.coarse, "", checkpoint_mode)
            
    elif start_i >= 0 or force_new_location or init_from is not None:
        if resume_from_previous:
            resume_i = start_i - 1
            checkpoint_mode = None # loads latest or best based on cfg
            print(f"Resuming checkpoint from stage {resume_i} (and will start from the next stage {start_i})")
        else:
            resume_i = start_i
            print(f"Resuming checkpoint from stage {resume_i} (and will start from the same stage {start_i})")
            checkpoint_mode = 'latest' # resuminng in the same stage, we want to pick up where we left of
        checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(configs[resume_i], stages_prefixes[resume_i], checkpoint_mode)
    else:
        checkpoint, checkpoint_kwargs = None, None

    if cfg_coarse.inout.full_run_dir == 'todo' or force_new_location:
        if force_new_location:
            print("The run will be resumed in a new foler (forked)")
            cfg_coarse.inout.previous_run_dir = cfg_coarse.inout.full_run_dir
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        random_id = str(hash(time))
        experiment_name = create_experiment_name(cfg_coarse)
        full_run_dir = Path(configs[0].inout.output_dir) / (time + "_" + random_id + "_" + experiment_name)
        exist_ok = False # a path for a new experiment should not yet exist
    else:
        experiment_name = cfg_coarse.inout.name
        len_time_str = len(datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))
        if hasattr(cfg_coarse.inout, 'time') and cfg_coarse.inout.time is not None:
            time = cfg_coarse.inout.time
        else:
            time = experiment_name[:len_time_str]
        if hasattr(cfg_coarse.inout, 'random_id') and cfg_coarse.inout.random_id is not None:
            random_id = cfg_coarse.inout.random_id
        else:
            random_id = ""
        full_run_dir = Path(cfg_coarse.inout.full_run_dir).parent
        exist_ok = True # a path for an old experiment should exist

    full_run_dir.mkdir(parents=True, exist_ok=exist_ok)
    print(f"The run will be saved  to: '{str(full_run_dir)}'")
    # with open("out_folder.txt", "w") as f:
        # f.write(str(full_run_dir))

    coarse_checkpoint_dir = full_run_dir / "coarse" / "checkpoints"
    coarse_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_coarse.inout.full_run_dir = str(coarse_checkpoint_dir.parent)
    cfg_coarse.inout.checkpoint_dir = str(coarse_checkpoint_dir)
    cfg_coarse.inout.name = experiment_name
    cfg_coarse.inout.time = time
    cfg_coarse.inout.random_id = random_id

    # # if cfg_detail.inout.full_run_dir == 'todo':
    # if cfg_detail is not None:
    #     detail_checkpoint_dir = full_run_dir / "detail" / "checkpoints"
    #     detail_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    #     cfg_detail.inout.full_run_dir = str(detail_checkpoint_dir.parent)
    #     cfg_detail.inout.checkpoint_dir = str(detail_checkpoint_dir)
    #     cfg_detail.inout.name = experiment_name
    #     cfg_detail.inout.time = time
    #     cfg_detail.inout.random_id = random_id

    # save config to target folder
    conf = DictConfig({})

    # TODO: name the stages dynamically if possible
    conf.coarse = cfg_coarse 
    
    # if cfg_detail is not None:
    #     conf.detail = cfg_detail
    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=conf, f=outfile)

    version = time
    if random_id is not None and len(random_id) > 0:
        # version += "_" + cfg_detail.inout.random_id
        version += "_" + cfg_coarse.inout.random_id

    wandb_logger = create_logger(
                         cfg_coarse.learning.logger_type,
                         name=experiment_name,
                         project_name=project_name,
                         config=OmegaConf.to_container(conf),
                         version=version,
                         save_dir=full_run_dir)

    model = None
    # if start_i >= 0 or force_new_location:
    if checkpoint is not None:
        print(f"Loading a checkpoint: {checkpoint} and starting from stage {start_i}")
    if start_i == -1:
        start_i = 0
    for i in range(start_i, len(configs)):
        cfg = configs[i]

        model_class = class_from_str(cfg_coarse.model.pl_module_class, sys.modules[__name__])

        model = single_stage_training_pass(model, cfg, stages[i], stages_prefixes[i], dm=None, logger=wandb_logger,
                                      data_preparation_function=prepare_data,
                                      checkpoint=checkpoint, checkpoint_kwargs=checkpoint_kwargs, 
                                      instantiation_function=model_class.instantiate)
        checkpoint = None



# def configure_detail(detail_cfg_default, detail_overrides):
#     from hydra.experimental import compose, initialize
#     initialize(config_path="../tempface_conf", job_name="train_tempface")
#     cfg_detail = compose(config_name=detail_cfg_default, overrides=detail_overrides)
#     return cfg_detail

def configure(coarse_cfg_default, coarse_overrides, detail_cfg_default=None, detail_overrides=None):
    from hydra.experimental import compose, initialize
    initialize(config_path="../facerec_conf", job_name="FaceReconstruction")
    cfg_coarse = compose(config_name=coarse_cfg_default, overrides=coarse_overrides)
    return cfg_coarse
    # if detail_cfg_default is not None and detail_overrides is not None:
    #     cfg_detail = compose(config_name=detail_cfg_default, overrides=detail_overrides)
    # else: 
    #     cfg_detail = None
    # return cfg_coarse, cfg_detail



def configure_and_train(coarse_cfg_default, coarse_overrides,
                        # detail_cfg_default, detail_overrides
                        ):
    # cfg_coarse, cfg_detail = configure(coarse_cfg_default, coarse_overrides,
    #                                    detail_cfg_default, detail_overrides)
    cfg_coarse = configure(coarse_cfg_default, coarse_overrides)
    train_model(cfg_coarse)


def configure_and_resume(run_path,
                         coarse_cfg_default, coarse_overrides,
                        #  detail_cfg_default, detail_overrides,
                         start_at_stage):
    # cfg_coarse, cfg_detail = configure(
    #                                    coarse_cfg_default, coarse_overrides,
    #                                    detail_cfg_default, detail_overrides)
    cfg_coarse = configure(coarse_cfg_default, coarse_overrides)

    # cfg_coarse_, cfg_detail_ = load_configs(run_path)
    cfg_coarse_ = load_configs(run_path)

    if start_at_stage < 2:
        raise RuntimeError("Resuming before stage 2 makes no sense, that would be training from scratch")
    elif start_at_stage == 2:
        cfg_coarse = cfg_coarse_
    elif start_at_stage == 3:
        raise RuntimeError("Resuming for stage 3 makes no sense, that is a testing stage")
    else:
        raise RuntimeError(f"Cannot resume at stage {start_at_stage}")

    train_model(cfg_coarse, 
               start_i=start_at_stage,
               resume_from_previous=True, #important, resume from previous stage's checkpoint
               force_new_location=True)


def load_configs(run_path):
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    cfg_coarse = conf.coarse
    # cfg_detail = conf.detail
    # return cfg_coarse, cfg_detail
    return cfg_coarse


# def resume_training(run_path, start_at_stage, resume_from_previous, force_new_location):
#     cfg_coarse, cfg_detail = load_configs(run_path)
#     train_model(cfg_coarse, cfg_detail,
#                start_i=start_at_stage,
#                resume_from_previous=resume_from_previous,
#                force_new_location=force_new_location)


def main():
    configured = False
    num_workers = 0

    if len(sys.argv) == 2: 
        if Path(sys.argv[1]).is_file(): 
            configured = True
            with open(sys.argv[1], 'r') as f:
                coarse_conf = OmegaConf.load(f)
            # detail_conf = None
            resume_from_previous = True
            force_new_location = False
            start_from = -1
        else:
            coarse_conf = sys.argv[1]
            # detail_conf = None
            coarse_override = []
            # detail_override = []
    elif len(sys.argv) < 2:
        coarse_conf = "emica_deca_stage"
        # detail_conf = None
        coarse_override = []
        # detail_override = []

        # coarse_conf = detail_conf
        # coarse_override = detail_override

    elif len(sys.argv) >= 2:
        if Path(sys.argv[1]).is_file():
            configured = True
            print("Found configured file. Loading it")
            with open(sys.argv[1], 'r') as f:
                coarse_conf = OmegaConf.load(f)
            # with open(sys.argv[2], 'r') as f:
            #     detail_conf = OmegaConf.load(f)
            # detail_conf = None
            coarse_override = []
            # detail_override = []
        else:
            coarse_conf = sys.argv[1]
            # detail_conf = sys.argv[2]
        if len(sys.argv) > 2:
            start_from = int(sys.argv[2])
            if len(sys.argv) > 3:
                resume_from_previous = bool(int(sys.argv[3]))
                if len(sys.argv) > 4:
                    force_new_location = bool(int(sys.argv[4]))
                else:
                    force_new_location = True
            else:
                resume_from_previous = True
        else:
            resume_from_previous = True
            force_new_location = False
            start_from = -1
    else:
        coarse_conf = "emica_deca_stage"
        coarse_override = []
        # detail_override = []

    if configured:
        print("Configured file loaded. Running training script")
        train_model(coarse_conf, 
                    # detail_conf,
                    start_from, 
                    resume_from_previous, force_new_location)
    else:
        configure_and_train(coarse_conf, coarse_override)
        # configure_and_train(coarse_conf, coarse_override, detail_conf, detail_override)



if __name__ == "__main__":
    main()

