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
import warnings
warnings.filterwarnings('ignore', message='.*Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. *', )

from inferno.datasets.LRS3Pseudo3DDM import LRS3Pseudo3DDM
from inferno_apps.MotionPrior.training.training_pass import( single_stage_training_pass, 
            get_checkpoint_with_kwargs, create_logger, configure) #, configure_and_train)
# from inferno.datasets.DecaDataModule import DecaDataModule
from inferno.models.temporal.motion_prior.L2lMotionPrior import L2lVqVae
from inferno.models.temporal.motion_prior.DeepPhase import DeepPhase
from inferno.datasets.FaceformerVocasetDM import FaceformerVocasetDM
from inferno.datasets.CelebVHQPseudo3DDM import CelebVHQPseudo3DDM
from inferno.datasets.MEADPseudo3DDM import MEADPseudo3DDM


from omegaconf import DictConfig, OmegaConf, open_dict
import sys, os
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import datetime
from inferno.utils.other import class_from_str


project_name = 'MotionPrior'


def get_condition_string_from_config(cfg):
    try: 
        if cfg.model.sequence_decoder.style_embedding == "none" or \
            (not cfg.model.sequence_decoder.style_embedding.use_expression) and \
            (not cfg.model.sequence_decoder.style_embedding.use_valence) and \
            (not cfg.model.sequence_decoder.style_embedding.use_arousal):
            return "original", None
        # if cfg.model.sequence_decoder.style_embedding.use_shape: 
        #     return "original", None # return original, we do not have conditioned testing for identity change
        if cfg.model.sequence_decoder.style_embedding.use_expression:
            return "expression", None 
        if cfg.model.sequence_decoder.style_embedding.use_valence and cfg.model.sequence_decoder.style_embedding.use_arousal:
            return "valence_arousal", None
        raise NotImplementedError("Conditioning not implemented for this style embedding configuration: ", cfg.model.sequence_decoder.style_embedding)
    except AttributeError as e:
        return "original", None
    

def create_single_dm(cfg, data_class):
    if data_class == "FaceformerVocasetDM": 
        # if 'augmentation' in cfg.data.keys() and len(cfg.data.augmentation) > 0:
        #     augmentation = OmegaConf.to_container(cfg.data.augmentation)
        # else:
        #     augmentation = None
        dm = FaceformerVocasetDM(
                cfg.data.input_dir, 
                cfg.data.template_file,
                cfg.data.train_subjects,
                cfg.data.val_subjects,
                cfg.data.test_subjects,
                # cfg.data.output_dir, 
                # processed_subfolder=cfg.data.processed_subfolder, 
                batch_size_train=cfg.learning.batching.batch_size_train,
                batch_size_val=cfg.learning.batching.batch_size_val, 
                batch_size_test=cfg.learning.batching.batch_size_test, 
                sequence_length_train=cfg.learning.batching.sequence_length_train, 
                sequence_length_val=cfg.learning.batching.sequence_length_val, 
                sequence_length_test=cfg.learning.batching.sequence_length_test, 
                num_workers=cfg.data.num_workers,
                debug_mode= cfg.data.get('debug_mode', False),
                )
        dataset_name = "Vocaset"
    elif data_class == "CelebVHQPseudo3DDM":
        condition_source, condition_settings = get_condition_string_from_config(cfg)
        dm = CelebVHQPseudo3DDM(
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
                sequence_length_train=cfg.learning.batching.sequence_length_train, 
                sequence_length_val=cfg.learning.batching.sequence_length_val, 
                sequence_length_test=cfg.learning.batching.sequence_length_test, 
                # occlusion_settings_train = OmegaConf.to_container(cfg.data.occlusion_settings_train), 
                # occlusion_settings_val = OmegaConf.to_container(cfg.data.occlusion_settings_val), 
                # occlusion_settings_test = OmegaConf.to_container(cfg.data.occlusion_settings_test), 
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
                inflate_by_video_size = cfg.data.inflate_by_video_size,
                preload_videos = cfg.data.preload_videos,
                test_condition_source=condition_source,
                test_condition_settings=condition_settings,
                
                read_video=cfg.data.get('read_video', True),
                reconstruction_type=cfg.data.get('reconstruction_type', None),
                return_appearance=cfg.data.get('return_appearance', None),
                average_shape_decode=cfg.data.get('average_shape_decode', None),
                emotion_type=cfg.data.get('emotion_type', None),
                return_emotion_feature=cfg.data.get('return_emotion_feature', None),
        )
        dataset_name = "CelebVHQ"
    elif data_class == "LRS3Pseudo3DDM":
        condition_source, condition_settings = get_condition_string_from_config(cfg)
        dm = LRS3Pseudo3DDM(
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
                sequence_length_train=cfg.learning.batching.sequence_length_train, 
                sequence_length_val=cfg.learning.batching.sequence_length_val, 
                sequence_length_test=cfg.learning.batching.sequence_length_test, 
                # occlusion_settings_train = OmegaConf.to_container(cfg.data.occlusion_settings_train), 
                # occlusion_settings_val = OmegaConf.to_container(cfg.data.occlusion_settings_val), 
                # occlusion_settings_test = OmegaConf.to_container(cfg.data.occlusion_settings_test), 
                split = cfg.data.split,
                num_workers=cfg.data.num_workers,
                # include_processed_audio = cfg.data.include_processed_audio,
                # include_raw_audio = cfg.data.include_raw_audio,
                drop_last=cfg.data.drop_last,
                ## end args of FaceVideoDataModule
                ## begin CelebVHQDataModule specific params
                # training_sampler=cfg.data.training_sampler,
                # landmark_types = cfg.data.landmark_types,
                # landmark_sources=cfg.data.landmark_sources,
                # segmentation_source=cfg.data.segmentation_source,
                include_processed_audio = cfg.data.include_processed_audio,
                include_raw_audio = cfg.data.include_raw_audio,
                inflate_by_video_size = cfg.data.inflate_by_video_size,
                preload_videos = cfg.data.preload_videos,
                test_condition_source=condition_source,
                test_condition_settings=condition_settings,
                read_video=cfg.data.get('read_video', True),
                reconstruction_type=cfg.data.get('reconstruction_type', None),
                return_appearance=cfg.data.get('return_appearance', None),
                average_shape_decode=cfg.data.get('average_shape_decode', None),

                emotion_type=cfg.data.get('emotion_type', None),
                return_emotion_feature=cfg.data.get('return_emotion_feature', None),
        )

        dataset_name = "LRS3"
    elif data_class == "MEADPseudo3DDM":
        condition_source, condition_settings = get_condition_string_from_config(cfg)
        dm = MEADPseudo3DDM(
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
                sequence_length_train=cfg.learning.batching.sequence_length_train, 
                sequence_length_val=cfg.learning.batching.sequence_length_val, 
                sequence_length_test=cfg.learning.batching.sequence_length_test, 
                # occlusion_settings_train = OmegaConf.to_container(cfg.data.occlusion_settings_train), 
                # occlusion_settings_val = OmegaConf.to_container(cfg.data.occlusion_settings_val), 
                # occlusion_settings_test = OmegaConf.to_container(cfg.data.occlusion_settings_test), 
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
                inflate_by_video_size = cfg.data.inflate_by_video_size,
                preload_videos = cfg.data.preload_videos,
                test_condition_source=condition_source,
                test_condition_settings=condition_settings,
                read_video=cfg.data.get('read_video', True),
                read_audio=cfg.data.get('read_audio', False),
                reconstruction_type=cfg.data.get('reconstruction_type', None),
                return_appearance=cfg.data.get('return_appearance', None),
                average_shape_decode=cfg.data.get('average_shape_decode', None),

                emotion_type=cfg.data.get('emotion_type', None),
                return_emotion_feature=cfg.data.get('return_emotion_feature', None),
        )
        dataset_name = "MEAD"
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


def create_experiment_name(cfg, version=0):
    experiment_name = cfg.model.pl_module_class

    if cfg.model.sequence_encoder.type == "CodeTalkerEncoder":
        experiment_name = "CD"
    
    if cfg.data.data_class:
        experiment_name += '_' + cfg.data.data_class[:5]


    if cfg.model.get('quantizer', None) is not None:
        if cfg.model.quantizer.type == "VectorQuantizer":
            experiment_name += '_VQVAE'
        elif cfg.model.quantizer.type == "GumbelVectorQuantizer":
            experiment_name += '_dVAE'
    else: 
        if cfg.model.sequence_encoder.type == "L2lEncoderWithGaussianHead":
            experiment_name += '_VAE'
        elif cfg.model.sequence_encoder.type == "L2lEncoderWithDeepPhaseHead":
            experiment_name += '_DPAE'
        else:
            experiment_name += '_AE'

    if hasattr(cfg.learning, 'early_stopping') and cfg.learning.early_stopping: # \
        # and hasattr(cfg_detail.learning, 'early_stopping') and cfg_detail.learning.early_stopping
        experiment_name += "_early"

    return experiment_name


def train_model(cfg, start_i=-1, 
                resume_from_previous = True,
                force_new_location=False):
    # configs = [cfg_coarse, cfg_coarse, cfg_detail, cfg_detail]
    # stages = ["train", "test", "train", "test"]
    # stages_prefixes = ["", "", "", ""]
    configs = [cfg, cfg]
    stages = ["train", "test"]
    stages_prefixes = ["", "", ]

    if start_i >= 0 or force_new_location:
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

    if cfg.inout.full_run_dir == 'todo' or force_new_location:
        if force_new_location:
            print("The run will be resumed in a new foler (forked)")
            cfg.inout.previous_run_dir = cfg.inout.full_run_dir
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        random_id = str(hash(time))
        experiment_name = create_experiment_name(cfg)
        full_run_dir = Path(configs[0].inout.output_dir) / (time + "_" + random_id + "_" + experiment_name)
        exist_ok = False # a path for a new experiment should not yet exist
    else:
        experiment_name = cfg.inout.name
        len_time_str = len(datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))
        if hasattr(cfg.inout, 'time') and cfg.inout.time is not None:
            time = cfg.inout.time
        else:
            time = experiment_name[:len_time_str]
        if hasattr(cfg.inout, 'random_id') and cfg.inout.random_id is not None:
            random_id = cfg.inout.random_id
        else:
            random_id = ""
        full_run_dir = Path(cfg.inout.full_run_dir)
        exist_ok = True # a path for an old experiment should exist

    full_run_dir.mkdir(parents=True, exist_ok=exist_ok)
    print(f"The run will be saved  to: '{str(full_run_dir)}'")
    # with open("out_folder.txt", "w") as f:
    #     f.write(str(full_run_dir))

    checkpoint_dir = full_run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg.inout.full_run_dir = str(checkpoint_dir.parent)
    cfg.inout.checkpoint_dir = str(checkpoint_dir)
    cfg.inout.name = experiment_name
    cfg.inout.time = time
    cfg.inout.random_id = random_id

    # add job id and submission dir to the config 
    with open_dict(cfg) as d:
        job_id_env = os.environ.get('JOB_ID', None)
        if job_id_env is not None:
            if d.inout.job_id_env is None:
                d.inout.job_id_env = [job_id_env]
            else:
                d.inout.job_id_env.append(job_id_env)

            job_id = job_id_env.split("#")[1]
            if d.inout.job_id is None:
                d.inout.job_id = [job_id]
            else:
                d.inout.job_id.append(job_id)

        submission_dir = os.environ.get('SUBMISSION_DIR', None)
        if submission_dir is not None:
            if d.inout.submission_dir is None:
                d.inout.submission_dir = [submission_dir]
            else:
                d.inout.submission_dir.append(submission_dir)

    # save config to target folder
    # conf = DictConfig({})

    # # TODO: name the stages dynamically if possible
    # conf.coarse = cfg_coarse 
    
    # if cfg_detail is not None:
    #     conf.detail = cfg_detail
    cfg_file = full_run_dir / "cfg.yaml"
    if cfg_file.exists():
        # back up the old config
        cfg_file.rename(cfg_file.parent / (cfg_file.name + ".bak"))

    with open(cfg_file, 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)

    version = time
    if random_id is not None and len(random_id) > 0:
        # version += "_" + cfg_detail.inout.random_id
        version += "_" + cfg.inout.random_id

    wandb_logger = create_logger(
                         cfg.learning.logger_type,
                         name=experiment_name,
                         project_name=project_name,
                         config=OmegaConf.to_container(cfg),
                         version=version,
                         save_dir=full_run_dir)

    model = None
    if start_i >= 0 or force_new_location:
        print(f"Loading a checkpoint: {checkpoint} and starting from stage {start_i}")
    if start_i == -1:
        start_i = 0
    for i in range(start_i, len(configs)):
        cfg = configs[i]

        model_class = class_from_str(cfg.model.pl_module_class, sys.modules[__name__])


        if "test" in stages[i]:
            # don't preload the dataset for testing 
            if hasattr(cfg.data, 'preload_videos'):
                cfg.data.preload_videos = False

        model = single_stage_training_pass(model, cfg, stages[i], stages_prefixes[i], dm=None, logger=wandb_logger,
                                      data_preparation_function=prepare_data,
                                      checkpoint=checkpoint, checkpoint_kwargs=checkpoint_kwargs, 
                                      instantiation_function=model_class.instantiate)
        checkpoint = None


def configure_detail(detail_cfg_default, detail_overrides):
    from hydra.experimental import compose, initialize
    initialize(config_path="../videoemorec_conf", job_name="VideoEmoRec")
    cfg_detail = compose(config_name=detail_cfg_default, overrides=detail_overrides)
    return cfg_detail


def configure_and_train(cfg_default, overrides):
    cfg = configure(cfg_default, overrides)
    train_model(cfg)


def configure_and_resume(run_path,
                         cfg_default, cfg_overrides,
                         start_at_stage):
    cfg_coarse = configure(cfg_default, cfg_overrides)

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
    return conf


def resume_training(run_path, start_at_stage, resume_from_previous, force_new_location):
    config = load_configs(run_path)
    train_model(config,
               start_i=start_at_stage,
               resume_from_previous=resume_from_previous,
               force_new_location=force_new_location)


def main():
    configured = False

    if len(sys.argv) == 2: 
        if Path(sys.argv[1]).is_file(): 
            configured = True
            with open(sys.argv[1], 'r') as f:
                config = OmegaConf.load(f)
            resume_from_previous = True
            force_new_location = False
            start_from = -1
        else:
            config = sys.argv[1]
            config_override = []
    elif len(sys.argv) < 2:
        config = "faceformer"
        # config = "flameformer"
        config_override = []
        
    elif len(sys.argv) >= 2:
        if Path(sys.argv[1]).is_file():
            configured = True
            print("Found configured file. Loading it")
            with open(sys.argv[1], 'r') as f:
                config = OmegaConf.load(f)
            config_override = []
        else:
            config = sys.argv[1]
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
        config = "faceformer"
        config_override = []

    if configured:
        print("Configured file loaded. Running training script")
        train_model(config, start_from, resume_from_previous, force_new_location)
    else:
        configure_and_train(config, config_override)


if __name__ == "__main__":
    main()

