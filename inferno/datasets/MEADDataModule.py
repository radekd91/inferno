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

from pathlib import Path
from time import time
from inferno.datasets.FaceDataModuleBase import FaceDataModuleBase
from inferno.datasets.FaceVideoDataModule import FaceVideoDataModule 
import numpy as np
import torch
from inferno.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp
from inferno.transforms.imgaug import create_image_augmenter
from inferno.layers.losses.MediaPipeLandmarkLosses import MEDIAPIPE_LANDMARK_NUMBER
from inferno.utils.collate import robust_collate
from torch.utils.data import DataLoader
import subprocess
import random as rand
from inferno.datasets.AffectNetDataModule import AffectNetExpressions


def get_affectnet_index_from_mead_expression_str(expr_str): 
    # string correction
    if expr_str == "angry": 
        expr_str = "anger"
    elif expr_str == "surprised": 
        expr_str = "surprise"
    elif expr_str == "disgusted": 
        expr_str = "disgust"
    # first letter capitalization
    expr_str = expr_str[0].upper() + expr_str[1:]
    # get index
    try: 
        return AffectNetExpressions[expr_str].value
    except KeyError as e: 
        print(f"Expression {expr_str} not found in AffectNet")
        raise e


def get_index_for_expression_with_intensity(expr_index, intensity):
    # intensity is an integer between 0 and 2 
    intensity -= 1
    assert intensity >= 0 and intensity <= 2, "Intensity must be between 0 and 2"
    if expr_index == 0: 
        assert intensity == 0, "Neutral expression must have intensity 0"
        return expr_index
    expression_index = (expr_index * 3 + intensity) - 2 # -2 because the first expression is neutral and does not have intensities
    return expression_index


def get_expression_and_intensity_from_index(index):
    # index is an integer between 0 and 22 
    assert index >= 0 and index < 22, "Index must be between 0 and 22"
    if index == 0: 
        expr_index = 0
        intensity = 0
    else:
        expr_index = (index + 2) // 3
        intensity = (index + 2) % 3
    assert expr_index >= 0 and expr_index < 8, "Expression index must be between 0 and 7 (inclusive)"
    intensity += 1
    return expr_index, intensity


def get_index_for_expression_with_intensity_identity(expr_index, intensity, identity, num_identities ):
    # intensity is an integer between 0 and 2 
    num_expressions = 8
    num_intensities = 3
    # assert intensity >= 1 and intensity <= 2, "Intensity must be between 0 and 2"
    # assert expr_index >= 0 and expr_index < num_expressions, "Expression index must be between 0 and 7 (inclusive)"
    assert identity >= 0 and identity < num_identities, "Identity index must be between 0 and num_identities-1"
    num_exp_int = num_expressions * num_intensities - 2 # -2 because the first expression is neutral and does not have intensities 2 and 3
    index = get_index_for_expression_with_intensity(expr_index, intensity) + identity * num_exp_int
    return index


def get_expression_and_intensity_and_identity_from_index(index, num_identities):
    # index is an integer between 0 and 22 
    num_expressions = 8
    num_intensities = 3
    num_exp_int = num_expressions * num_intensities - 2 # -2 because the first expression is neutral and does not have intensities 2 and 3

    assert index >= 0 and index < num_exp_int * num_identities

    identity = index % num_identities
    exp_int = index // num_identities
    expr_index, intensity = get_expression_and_intensity_from_index(exp_int)
    return expr_index, intensity, identity


class MEADDataModule(FaceVideoDataModule): 

    def __init__(self, 
            ## begin args of FaceVideoDataModule
            root_dir, 
            output_dir, 
            processed_subfolder=None, 
            face_detector='mediapipe', 
            # landmarks_from='sr_res',
            landmarks_from=None,
            face_detector_threshold=0.5, 
            image_size=224, scale=1.25, 
            processed_video_size=384,
            batch_size_train=16,
            batch_size_val=16,
            batch_size_test=16,
            sequence_length_train=16,
            sequence_length_val=16,
            sequence_length_test=16,
            # occlusion_length_train=0,
            # occlusion_length_val=0,
            # occlusion_length_test=0,            
            bb_center_shift_x=0., # in relative numbers
            bb_center_shift_y=0., # in relative numbers (i.e. -0.1 for 10% shift upwards, ...)
            occlusion_settings_train=None,
            occlusion_settings_val=None,
            occlusion_settings_test=None,
            split = "random_70_15_15", #TODO: implement other splits if required
            num_workers=4,
            device=None,
            augmentation=None,
            drop_last=True,
            include_processed_audio = True,
            include_raw_audio = True,
            preload_videos=False,
            inflate_by_video_size=False,
            ## end args of FaceVideoDataModule
            ## begin MEADDataModule specific params
            training_sampler="uniform",
            landmark_types = None,
            landmark_sources=None,
            segmentation_source=None,
            segmentation_type=None,
            viewing_angles=None,
            read_video=True,
            read_audio=True,
            shuffle_validation=False,
            align_images=True,
            return_mica_images=False,
            ):
        super().__init__(root_dir, output_dir, processed_subfolder, 
            face_detector, face_detector_threshold, image_size, scale, 
            processed_video_size=processed_video_size,
            device=device, 
            unpack_videos=False, save_detection_images=False, 
            # save_landmarks=True,
            save_landmarks=False, # trying out this option
            save_landmarks_one_file=True, 
            save_segmentation_frame_by_frame=False, 
            save_segmentation_one_file=True,
            bb_center_shift_x=bb_center_shift_x, # in relative numbers
            bb_center_shift_y=bb_center_shift_y, # in relative numbers (i.e. -0.1 for 10% shift upwards, ...)
            include_processed_audio = include_processed_audio,
            include_raw_audio = include_raw_audio,
            preload_videos=preload_videos,
            inflate_by_video_size=inflate_by_video_size,
            read_video=read_video,
            read_audio=read_audio,
            align_images=align_images,
            return_mica_images=return_mica_images,
            )
        # self.detect_landmarks_on_restored_images = landmarks_from
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.sequence_length_train = sequence_length_train
        self.sequence_length_val = sequence_length_val
        self.sequence_length_test = sequence_length_test

        self.split = split
        self.num_workers = num_workers
        self.drop_last = drop_last

        # self.occlusion_length_train = occlusion_length_train
        # self.occlusion_length_val = occlusion_length_val
        # self.occlusion_length_test = occlusion_length_test
        self.occlusion_settings_train = occlusion_settings_train or {}
        self.occlusion_settings_val = occlusion_settings_val or {}
        self.occlusion_settings_test = occlusion_settings_test or {}
        self.augmentation = augmentation

        self.training_sampler = training_sampler.lower()
        self.shuffle_validation = shuffle_validation
        self.annotation_json_path = None # Path(root_dir).parent / "celebvhq_info.json" 
        ## assert self.annotation_json_path.is_file()

        self.landmark_types = landmark_types or ["mediapipe", "fan"]
        self.landmark_sources = landmark_sources or ["original", "aligned"]
        self.segmentation_source = segmentation_source or "aligned"
        self.segmentation_type = segmentation_type or "focus"
        self.use_original_video = False

        self.viewing_angles = viewing_angles or ["front"] 
        if isinstance( self.viewing_angles, str): 
            self.viewing_angles = [self.viewing_angles]

        self._must_include_audio = 'warn'
    

    def prepare_data(self):
        # super().prepare_data()
        
        outdir = Path(self.output_dir)
        if Path(self.metadata_path).is_file():
            print("The dataset is already processed. Loading")
            self._loadMeta()
            return
        # else:
        self._gather_data()
        self._saveMeta()
        self._loadMeta()
        # self._unpack_videos()
        # self._saveMeta()


    def get_single_video_dataset(self, i):
        # dataset = MEADDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
        #         [i], 
        #         self.audio_metas, 
        #         # self.sequence_length_test, 
        #         "all", 
        #         image_size=self.image_size, 
        #         # **self.occlusion_settings_test,
        #         hack_length=False, 
        #         use_original_video=self.use_original_video,
        #         include_processed_audio = self.include_processed_audio,
        #         include_raw_audio = self.include_raw_audio,
        #         landmark_types=self.landmark_types,
        #         # landmark_types="mediapipe",
        #         landmark_source=self.landmark_sources,
        #         # landmark_source="original",
        #         segmentation_source=self.segmentation_source,
        #         segmentation_type= self.segmentation_type,
        #         # temporal_split_start=self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
        #         # temporal_split_end= sum(self.temporal_split) if self.temporal_split is not None else None,
        #         preload_videos=self.preload_videos,
        #         inflate_by_video_size=False,
        #         align_images=self.align_images,
        #         original_image_size=self.processed_video_size,
        #         return_mica_images=self.return_mica_images,
        #         )
        dataset = MEADDataset(self.root_dir, self.output_dir, 
                self.video_list, self.video_metas, [i],
                self.audio_metas, 
                "all", 
                image_size=self.image_size,  
                hack_length=False, 
                use_original_video=self.use_original_video,
                include_processed_audio = self.include_processed_audio,
                include_raw_audio = self.include_raw_audio,
                landmark_types=self.landmark_types,
                landmark_source=self.landmark_sources,
                segmentation_source=self.segmentation_source,
                segmentation_type= self.segmentation_type,
                # temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                # temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,
                read_video=self.read_video,
                read_audio=self.read_audio,
                align_images=self.align_images,
                original_image_size=self.processed_video_size,
                return_mica_images=self.return_mica_images,
            )
        return dataset

   
    def _gather_data(self, exist_ok=True):
        print(f"Processing MEAD dataset (video and audio) for angles: {self.viewing_angles} and all emotion intensities")
        Path(self.output_dir).mkdir(parents=True, exist_ok=exist_ok)

        self.video_list = [] 
        self.video_list = [] 
        for viewing_angle in self.viewing_angles:
            # video_list = sorted(list(Path(self.root_dir).rglob(f'**/video/{viewing_angle}/**/**/*.mp4')))
            # find video files using bash find command (faster than python glob)
            video_list = sorted(subprocess.check_output(f"find {self.root_dir} -wholename */{viewing_angle}/*/*/*.mp4", shell=True).decode("utf-8").splitlines())
            video_list = [Path(path).relative_to(self.root_dir) for path in video_list]
            
            self.video_list += video_list
        print("Found %d video files." % len(self.video_list))
        self._gather_video_metadata()

    # ## DOESN'T work as some videos seem to not have corresponding audios :-(
    # def _gather_data(self, exist_ok=True):
    #     print(f"Processing MEAD dataset (video and audio) for angles: {self.viewing_angles} and all emotion intensities")
    #     Path(self.output_dir).mkdir(parents=True, exist_ok=exist_ok)

    #     # find audio files using bash find command
    #     audio_list = sorted(subprocess.check_output(f"find {self.root_dir} -name '*.m4a'", shell=True).decode("utf-8").splitlines())
    #     # audio_list = sorted(Path(self.root_dir).rglob('**/audio/**/**/*.m4a'))
    #     audio_list = [Path(path).relative_to(self.root_dir) for path in audio_list]

    #     # print("videos...")
    #     # prepare video list
    #     self.video_list = [] 
    #     # self.audio_list = []
    #     for viewing_angle in self.viewing_angles:
    #         # video_list = sorted(list(Path(self.root_dir).rglob(f'**/video/{viewing_angle}/**/**/*.mp4')))

    #         # find video files using bash find command
    #         video_list = sorted(subprocess.check_output(f"find {self.root_dir} -wholename */{viewing_angle}/*/*/*.mp4", shell=True).decode("utf-8").splitlines())
    #         video_list = [Path(path).relative_to(self.root_dir) for path in video_list]
            
    #         # assert len(video_list) == len(audio_list), f"Number of videos ({len(video_list)}) and audio files ({len(audio_list)}) do not match"
            
    #         # check the audio and video names are corresponding
    #         audio_names = set([ Path("/".join((audio_list[ai].parts[-5],) + audio_list[ai].parts[-3:])).with_suffix('') for ai in range(len(audio_list)) ])
    #         for vi in range(len(video_list)):
    #             # get the name with the last two relative subfolders
    #             video_name = Path("/".join((video_list[vi].parts[-6],) + video_list[vi].parts[-3:])).with_suffix('')
    #             # audio_name = Path("/".join((audio_list[vi].parts[-5],) + audio_list[vi].parts[-3:])).with_suffix('')
    #             if video_name not in audio_names:
    #                 raise RuntimeError(f"Video {video_name} does not have corresponding audio file")
    #             # if video_name != audio_name:
    #             #     raise ValueError(f"Video name '{video_name} 'is not corresponding to audio name '{audio_name}'")


    #         self.video_list += [video_list]
    #         self.audio_list += [audio_list]

    #     # print("audios...")
    #     # prepare audio list
        
        
    #     # self._gather_video_metadata()
    #     print("Found %d video files." % len(self.video_list))


    def _filename2index(self, filename):
        return self.video_list.index(filename)

    def _get_landmark_method(self):
        return self.face_detector_type

    def _get_segmentation_method(self):
        # return "bisenet"
        return "focus"

    def _detect_faces(self):
        return super()._detect_faces( )

    def _get_num_shards(self, videos_per_shard): 
        num_shards = int(np.ceil( self.num_sequences / videos_per_shard))
        return num_shards

    def _get_path_to_sequence_files(self, sequence_id, file_type, method="", suffix="", assert_=True): 
        if assert_:
            assert file_type in ['videos', 'videos_aligned', 'detections', 
                "landmarks", "landmarks_original", "landmarks_aligned",
                "segmentations", "segmentations_aligned",
                "emotions", "reconstructions", "audio"]
        video_file = self.video_list[sequence_id]
        if len(method) > 0:
            file_type += "_" + method 
        if len(suffix) > 0:
            file_type += suffix

        suffix = Path(file_type) / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder


    def _process_video(self, idx, extract_audio=True, 
            restore_videos=True, 
            detect_landmarks=True, 
            recognize_faces=True,
            # cut_out_faces=True,
            segment_videos=True, 
            detect_aligned_landmarks=False,
            reconstruct_faces=False,
            recognize_emotions=False,
            segmentations_to_hdf5=False,
            ):
        if extract_audio: 
            self._extract_audio_for_video(idx)
        # if restore_videos:
        #     self._deep_restore_sequence_sr_res(idx)
        if detect_landmarks:
            self._detect_faces_in_sequence(idx)
        if recognize_faces: 
            self._recognize_faces_in_sequence(idx)
            self._identify_recognitions_for_sequence(idx)
            self._extract_personal_recognition_sequences(idx)
        # if cut_out_faces: 
        #     self._cut_out_detected_faces_in_sequence(idx)
        if segment_videos:
            if segment_videos:
                # seg_methods = ['bisenet', 'focus']
                seg_methods = ['focus']
                for seg_method in seg_methods:
                    self._segment_faces_in_sequence(idx, use_aligned_videos=True, segmentation_net=seg_method)
            # self._segment_faces_in_sequence(idx, use_aligned_videos=True)
            # raise NotImplementedError()
        if detect_aligned_landmarks: 
            self._detect_landmarkes_in_aligned_sequence(idx)

        if reconstruct_faces: 
            rec_methods = ["EMICA-MEAD_flame2020"]
            # rec_methods = ["EMICA-MEAD_flame2023"]
            self._reconstruct_faces_in_sequence_v2(
                        idx, reconstruction_net=None, device=None,
                        save_obj=False, save_mat=True, save_vis=False, save_images=False,
                        save_video=False, rec_methods=rec_methods, retarget_from=None, retarget_suffix=None)
        if recognize_emotions:
            emo_methods = ['resnet50', ]
            # emo_methods = ['swin-b', ]
            self._extract_emotion_in_sequence(idx, emo_methods=emo_methods)
        
        if segmentations_to_hdf5:
            seg_methods = ['bisenet', 'focus']
            for seg_method in seg_methods:
                self._segmentations_to_hdf5(idx, segmentation_net=seg_method, use_aligned_videos=True)

    def _process_shard(self, videos_per_shard, shard_idx, 
        extract_audio=True,
        restore_videos=True, 
        detect_landmarks=True, 
        segment_videos=True, 
        detect_aligned_landmarks=False,
        reconstruct_faces=False,
        recognize_emotions=False,
        segmentations_to_hdf5=False,
    ):
        num_shards = self._get_num_shards(videos_per_shard)
        start_idx = shard_idx * videos_per_shard
        end_idx = min(start_idx + videos_per_shard, self.num_sequences)

        print(f"Processing shard {shard_idx} of {num_shards}")

        idxs = np.arange(self.num_sequences, dtype=np.int32)
        np.random.seed(0)
        np.random.shuffle(idxs)

        if detect_aligned_landmarks: 
            assert not detect_landmarks, \
                "Cannot detect landmarks for aligned videos and original videos at the same time"  +\
                " since this requries instantiation of a new face detector."
            # self.face_detector_type = 'fan'
            self.face_detector_type = 'fan3d'
            self._instantiate_detector(overwrite=True)

        for i in range(start_idx, end_idx):
            idx = idxs[i]
            self._process_video(idx, 
                extract_audio=extract_audio, 
                restore_videos=restore_videos,
                detect_landmarks=detect_landmarks, 
                segment_videos=segment_videos, 
                detect_aligned_landmarks=detect_aligned_landmarks,
                reconstruct_faces=reconstruct_faces, 
                recognize_emotions=recognize_emotions,
                segmentations_to_hdf5=segmentations_to_hdf5,
                )
            
        print("Done processing shard")

    def _get_path_to_sequence_files(self, sequence_id, file_type, method="", suffix="", assert_=True): 
        if assert_:
            assert file_type in ['videos', 'videos_aligned', 'detections', 
                "landmarks", "landmarks_original", "landmarks_aligned",
                "segmentations", "segmentations_aligned",
                "emotions", "reconstructions", "audio", "rec_videos"], \
                f"Unknown file type: {file_type}"
        video_file = self.video_list[sequence_id]
        if len(method) > 0:
            file_type += "/" + method 
        if len(suffix) > 0:
            file_type += suffix

        if len(video_file.parts) == 6:
            expected_values = ["video", "1", "2"] # for some reason the authors are not consistent with folder names
            assert video_file.parts[1] in expected_values, f"Unexpected path structure. Expected one of {expected_values}, got {video_file.parts[1]}"
      
        # suffix = Path(file_type) / video_file.stem
        person_id = video_file.parts[0]
        
        suffix = Path(file_type) / person_id / "/".join(video_file.parts[2:-1]) / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder

    def _video_identity(self, index): 
        return self.video_list[index].parts[0]

    def _video_expression(self, index): 
        expr = self.video_list[index].parts[3] 
        if expr in ["front", "down", "top", "left_30",  "left_60", "right_30", "right_60", "1", "2"]: # a hack for the MORONIC inconsistency in the paths of the MEAD dataset
            return self.video_list[index].parts[4]
        return expr

    def _expression_intensity(self, index): 
        intensity = self.video_list[index].parts[4]
        if "level" not in intensity: # a hack for the MORONIC inconsistency in the paths of the MEAD dataset
            return self.video_list[index].parts[5]
        return intensity

    def _get_expression_intensity_map(self, indices):
        expression_intensity2idx = {}
        for i in indices:
            expression = self._video_expression(i)
            intensity = self._expression_intensity(i)
            key = (expression, intensity)
            if key not in expression_intensity2idx:
                expression_intensity2idx[key] = []
            expression_intensity2idx[key] += [i]
        return expression_intensity2idx

    def _get_identity_expression_intensity_map(self, indices):
        identity_expression_intensity2idx = {}
        for i in indices:
            identity = self._video_identity(i)
            expression = self._video_expression(i)
            intensity = self._expression_intensity(i)
            key = (identity, expression, intensity)
            if key not in identity_expression_intensity2idx:
                identity_expression_intensity2idx[key] = []
            identity_expression_intensity2idx[key] += [i]
        return identity_expression_intensity2idx

    def _get_identity_map(self, indices):
        identity2idx = {}
        for i in indices:
            identity = self._video_identity(i)
            key = identity
            if key not in identity2idx:
                identity2idx[key] = []
            identity2idx[key] += [i]
        return identity2idx

    def _get_subsets(self, set_type=None):
        set_type = set_type or "unknown"
        self.temporal_split = None
        if "specific_video_temporal" in set_type:
            raise NotImplementedError("Not implemented yet")
        elif "specific_identity" in set_type: 
            res = set_type.split("_")
            random_or_sorted = res[2] 
            assert random_or_sorted in ["random", "sorted"], f"Unknown random_or_sorted value: '{random_or_sorted}'"
            identity = res[-1]
            train = float(res[-3])
            val = float(res[-2])
            train = train / (train + val)
            val = 1 - train
            indices = [i for i in range(len(self.video_list)) if self._video_identity(i) == identity]

            # expression_list = ['angry', 'contempt,' 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
            # intensity_list = ['level_1', 'level_2', 'level_3']

            expression_intensity2idx = self._get_expression_intensity_map(indices)

            # expression_intensity2filename = {}
            # for key, idxs in expression_intensity2idx.items():
            #     expression_intensity2filename[key] = [self.video_list[i] for i in idxs]

            training = []
            validation = []
            for k, idxs in expression_intensity2idx.items():
                if random_or_sorted == "sorted":
                    idxs = sorted(idxs)
                elif random_or_sorted == "random":
                    rand.shuffle(idxs)
                else:
                    raise ValueError(f"Unknown random_or_sorted value: '{random_or_sorted}'")
                num_train = int(len(idxs) * train)
                training += idxs[:num_train]
                validation += idxs[num_train:]

            return training, validation, []
        elif "random_by_identityV2" in set_type:
            res = set_type.split("_")
            random_or_sorted = res[3] 
            assert random_or_sorted in ["random", "sorted"], f"Unknown random_or_sorted value: '{random_or_sorted}'"
            train = float(res[-3])
            val = float(res[-2])
            test = float(res[-1])
            train_ = train / (train + val + test)
            val_ = val / (train + val + test)
            test_ = 1 - train_ - val_
            indices = np.arange(len(self.video_list), dtype=np.int32)

            identity2idx = self._get_identity_map(indices)
            # set of identities
            identities = sorted(list(set(identity2idx.keys())))
            male_identities = [i for i in identities if i.startswith("M")]
            female_identities = [i for i in identities if i.startswith("W")]
            rest = set(identities) - set(male_identities) - set(female_identities)
            assert len(rest) == 0, f"Unexpected identities: {rest}"

            if random_or_sorted == "random":
                seed = 4
                # # get the list of identities
                rand.Random(seed).shuffle(identities)
                # rand.shuffle(identities)

            # training_ids = identities[:int(len(identities) * train_)]
            # validation_ids = identities[int(len(identities) * train_):int(len(identities) * (train_ + val_))]
            # test_ids = identities[int(len(identities) * (train_ + val_)):]

            training_ids = male_identities[:int(len(male_identities) * train_)]
            validation_ids = male_identities[int(len(male_identities) * train_):int(len(male_identities) * (train_ + val_))]
            test_ids = male_identities[int(len(male_identities) * (train_ + val_)):]

            training_ids += female_identities[:int(len(female_identities) * train_)]
            validation_ids += female_identities[int(len(female_identities) * train_):int(len(female_identities) * (train_ + val_))]
            test_ids += female_identities[int(len(female_identities) * (train_ + val_)):]

            training = []
            validation = []
            testing = []
            for id, indices in identity2idx.items():
                if id in training_ids:
                    training += indices
                elif id in validation_ids:
                    validation += indices
                elif id in test_ids:
                    testing += indices
                else:
                    raise RuntimeError(f"Unassigned identity in training/validation/test split: '{id}'. This should not happen")
            training.sort()
            validation.sort()
            testing.sort()
            return training, validation, testing

        elif ("random_by_sequence" in set_type) or ("random_by_identity" in set_type):
            # WARNING: THIS NAME IS NOT ACCURATE, IT IS NOT RANDOM BY IDENTITY BUT RANDOM BY EXPRESSION AND INTENSITY
            # SO ALL IDENTITIES ARE IN BOTH TRAIN AND VAL (BUT THE TRAIN AND VAL VIDEOS DON'T OVERLAP)
            # The new name is "random_by_sequence"
            res = set_type.split("_")
            random_or_sorted = res[3] 
            assert random_or_sorted in ["random", "sorted"], f"Unknown random_or_sorted value: '{random_or_sorted}'"
            train = float(res[-3])
            val = float(res[-2])
            test = float(res[-1])
            train_ = train / (train + val + test)
            val_ = val / (train + val + test)
            test_ = 1 - train_ - val_
            indices = np.arange(len(self.video_list), dtype=np.int32)
            # # get video_clips_by_identity
            # video_clips_by_identity = {}
            # video_clips_by_identity_indices = {}
            # index_counter = 0

            id_expression_intensity2idx = self._get_identity_expression_intensity_map(indices)

            id_expression_intensity2filename = {}
            for key, idxs in id_expression_intensity2idx.items():
                id_expression_intensity2filename[key] = [self.video_list[i] for i in idxs]

            # for i in range(len(self.video_list)):
            #     key = self._video_identity(i)
            #     if key in video_clips_by_identity.keys(): 
            #         video_clips_by_identity[key] += [i]
            #     else: 
            #         video_clips_by_identity[key] = [i]
            #         video_clips_by_identity_indices[key] = index_counter
            #         index_counter += 1
            
            training = []
            validation = []
            testing = []
            for k, idxs in id_expression_intensity2idx.items():
                if random_or_sorted == "sorted":
                    idxs = sorted(idxs)
                elif random_or_sorted == "random":
                    seed = 4
                    rand.Random(seed).shuffle(idxs)
                    # rand.shuffle(idxs)
                else:
                    raise ValueError(f"Unknown random_or_sorted value: '{random_or_sorted}'")
                num_train = int(len(idxs) * train_)
                num_val = int(len(idxs) * val_)
                training += idxs[:num_train]
                validation += idxs[num_train:num_train+num_val]
                testing += idxs[num_train+num_val:]
            return training, validation, testing

            # import random
            # seed = 4
            # # get the list of identities
            # identities = list(video_clips_by_identity.keys())
            # random.Random(seed).shuffle(identities)
            # # identitities randomly shuffled 
            # # this determines which identities are for training and which for validation and testing
            # training = [] 
            # validation = [] 
            # test = []
            # for i, identity in enumerate(identities): 
            #     # identity = identities[i] 
            #     identity_videos = video_clips_by_identity[identity]
            #     if i < int(train_ * len(identities)): 
            #         training += identity_videos
            #     elif i < int((train_ + val_) * len(identities)):
            #         validation += identity_videos
            #     else: 
            #         test += identity_videos
            # training.sort() 
            # validation.sort()
            # test.sort()
            # # at this point, the identities are shuffled but per-identity videos have 
            # # consecutive indices, for training, shuffle afterwards (set shuffle to True or use a 
            # # sampler )
            # return training, validation, test
        elif "random" in set_type:
            res = set_type.split("_")
            assert len(res) >= 3, "Specify the train/val/test split by 'random_train_val_test' to the set_type"
            train = int(res[1])
            val = int(res[2])
            if len(res) == 4:
                test = int(res[3])
            else:
                test = 0
            train_ = train / (train + val + test)
            val_ = val / (train + val + test)
            test_ = 1 - train_ - val_
            indices = [i for i in range(len(self.video_list))]
            import random
            seed = 4
            random.Random(seed).shuffle(indices)
            num_train = int(train_ * len(indices))
            num_val = int(val_ * len(indices))
            num_test = len(indices) - num_train - num_val

            training = indices[:num_train] 
            validation = indices[num_train:(num_train + num_val)]
            test = indices[(num_train + num_val):]
            return training, validation, test
        elif set_type == "overfit_all": 
            training = np.arange(len(self.video_list), dtype=np.int32)
            validation = np.arange(len(self.video_list), dtype=np.int32)
            test = [] 
            return training, validation, test
        elif "temporal" in set_type:
            raise NotImplementedError("Not implemented yet")
        else: 
            raise ValueError(f"Unknown set type: {set_type}")


    def setup(self, stage=None):
        train, val, test = self._get_subsets(self.split)
        training_augmenter = create_image_augmenter(self.image_size, self.augmentation)
        self.training_set = MEADDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, train, 
                self.audio_metas, self.sequence_length_train, image_size=self.image_size, 
                transforms=training_augmenter,
                **self.occlusion_settings_train,
                hack_length=False,
                use_original_video=self.use_original_video,
                include_processed_audio = self.include_processed_audio,
                include_raw_audio = self.include_raw_audio,
                landmark_types=self.landmark_types,
                landmark_source=self.landmark_sources,
                segmentation_source=self.segmentation_source,
                segmentation_type= self.segmentation_type,
                temporal_split_start= 0 if self.temporal_split is not None else None,
                temporal_split_end=self.temporal_split[0] if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,
                read_video=self.read_video,
                read_audio=self.read_audio,
                align_images=self.align_images,
                original_image_size=self.processed_video_size,
                return_mica_images=self.return_mica_images,
              )
                    
        self.validation_set = MEADDataset(self.root_dir, self.output_dir, 
                self.video_list, self.video_metas, val, self.audio_metas, 
                self.sequence_length_val, image_size=self.image_size,  
                **self.occlusion_settings_val,
                hack_length=False, 
                use_original_video=self.use_original_video,
                include_processed_audio = self.include_processed_audio,
                include_raw_audio = self.include_raw_audio,
                landmark_types=self.landmark_types,
                landmark_source=self.landmark_sources,
                segmentation_source=self.segmentation_source,
                segmentation_type= self.segmentation_type,
                temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,
                read_video=self.read_video,
                read_audio=self.read_audio,
                align_images=self.align_images,
                original_image_size=self.processed_video_size,
                return_mica_images=self.return_mica_images,
            )
        self.validation_set._set_identity_label(self.training_set.identity_labels, self.training_set.identity_label2index)

        self.test_set = MEADDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, test, self.audio_metas, 
                self.sequence_length_test, image_size=self.image_size, 
                **self.occlusion_settings_test,
                hack_length=False, 
                use_original_video=self.use_original_video,
                include_processed_audio = self.include_processed_audio,
                include_raw_audio = self.include_raw_audio,
                landmark_types=self.landmark_types,
                landmark_source=self.landmark_sources,
                segmentation_source=self.segmentation_source,
                segmentation_type= self.segmentation_type,
                temporal_split_start=self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                temporal_split_end= sum(self.temporal_split) if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,
                read_video=self.read_video,
                read_audio=self.read_audio,
                align_images=self.align_images,
                original_image_size=self.processed_video_size,
                return_mica_images=self.return_mica_images,
                )
        self.test_set._set_identity_label(self.training_set.identity_labels, self.training_set.identity_label2index)

    def train_sampler(self):
        if self.training_sampler == "uniform":
            sampler = None
        else:
            raise ValueError(f"Invalid sampler value: '{self.training_sampler}'")
        return sampler

    def train_dataloader(self):
        sampler = self.train_sampler()
        dl =  DataLoader(self.training_set, shuffle=sampler is None, num_workers=self.num_workers, pin_memory=True,
                        batch_size=self.batch_size_train, drop_last=self.drop_last, sampler=sampler, 
                        collate_fn=robust_collate, 
                        # persistent_workers=True,
                        persistent_workers=self.num_workers > 0,
                        )
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.validation_set, 
                        # shuffle=False, 
                        shuffle=self.shuffle_validation, 
                        num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.batch_size_val, 
                        #   drop_last=self.drop_last
                          drop_last=False, 
                        collate_fn=robust_collate,
                        # persistent_workers=True,
                        persistent_workers=self.num_workers > 0,
                        )
        if hasattr(self, "validation_set_2"): 
            dl2 =  DataLoader(self.validation_set_2, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                            batch_size=self.batch_size_val, 
                            # drop_last=self.drop_last, 
                            drop_last=False, 
                            collate_fn=robust_collate, 
                        # persistent_workers=True,
                        persistent_workers=self.num_workers > 0,
                            )
                            
            return [dl, dl2]
        return dl 

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.batch_size_test, drop_last=self.drop_last)
    
    def get_num_training_identities(self):
        return len(self.training_set.identity_labels)

    def get_loggable_video_string(self, video_path): 
        return "_".join([video_path.parts[0], video_path.parts[2], video_path.parts[3], video_path.parts[4]])


import imgaug
from inferno.datasets.VideoDatasetBase import VideoDatasetBaseV2

# class MEADDataset(VideoDatasetBase):
class MEADDataset(VideoDatasetBaseV2):

    def __init__(self,
            root_path,
            output_dir,
            video_list, 
            video_metas,
            video_indices,
            # audio_paths, 
            audio_metas,
            sequence_length,
            audio_noise_prob=0.0,
            stack_order_audio=4,
            audio_normalization="layer_norm",
            landmark_types=None, 
            segmentation_type = "bisenet",
            landmark_source = "original",
            segmentation_source = "aligned",
            occlusion_length=0,
            occlusion_probability_mouth = 0.0,
            occlusion_probability_left_eye = 0.0,
            occlusion_probability_right_eye = 0.0,
            occlusion_probability_face = 0.0,
            image_size=None, ## output image size 
            transforms : imgaug.augmenters.Augmenter = None,
            hack_length=False,
            use_original_video=False,
            include_processed_audio = True,
            include_raw_audio = True,
            temporal_split_start=None,
            temporal_split_end=None,
            preload_videos=False,
            inflate_by_video_size=False,
            include_filename=False, # if True includes the filename of the video in the sample
            read_video=True,
            read_audio=True,
            reconstruction_type=None,
            return_global_pose = False,
            return_appearance = False,
            average_shape_decode = True,
            emotion_type=None,
            return_emotion_feature=False,
            align_images = True,
            original_image_size = None,
            return_mica_images = False,
    ) -> None:
        landmark_types = landmark_types or ["mediapipe", "fan"]
        super().__init__(
            root_path,
            output_dir,
            video_list, 
            video_metas,
            video_indices,
            # audio_paths, 
            audio_metas,
            sequence_length,
            audio_noise_prob=audio_noise_prob,
            stack_order_audio=stack_order_audio,
            audio_normalization=audio_normalization,
            landmark_types=landmark_types, 
            segmentation_type = segmentation_type,
            landmark_source = landmark_source,
            segmentation_source = segmentation_source,
            occlusion_length=occlusion_length,
            occlusion_probability_mouth = occlusion_probability_mouth,
            occlusion_probability_left_eye = occlusion_probability_left_eye,
            occlusion_probability_right_eye = occlusion_probability_right_eye,
            occlusion_probability_face = occlusion_probability_face,
            image_size=image_size, 
            transforms = transforms,
            hack_length=hack_length,
            use_original_video=use_original_video,
            include_processed_audio = include_processed_audio,
            include_raw_audio = include_raw_audio,
            temporal_split_start=temporal_split_start,
            temporal_split_end=temporal_split_end,
            preload_videos=preload_videos,
            inflate_by_video_size=inflate_by_video_size,
            include_filename=include_filename,
            read_video=read_video,
            read_audio=read_audio,
            reconstruction_type=reconstruction_type,
            return_global_pose = return_global_pose,
            return_appearance = return_appearance,
            average_shape_decode = average_shape_decode,
            emotion_type=emotion_type,
            return_emotion_feature=return_emotion_feature,
            align_images = align_images,
            original_image_size = original_image_size,
            return_mica_images = return_mica_images,
        )
        self._setup_identity_labels()
        self.read_gt_text = False
    
    def _setup_identity_labels(self):
        # look at all the sample paths and get the their identity label
        self.identity_labels = set()
        for index in range(len(self)):
            self.identity_labels.add(self._get_identity_label(index))
        self.identity_labels = sorted(list(self.identity_labels))
        self.identity_label2index = {label: index for index, label in enumerate(self.identity_labels)}
        self.own_identity_label = True

    def _get_identity_label(self, index):
        identity = self.video_list[self.video_indices[index]].parts[0]
        return identity
    
    def _get_identity_label_index(self, index):
        identity = self._get_identity_label(index)
        if self.own_identity_label: # the training and this (validation/test?) set share the same identity labels
            return self.identity_label2index[identity]
        else:
            # the training and this (validation/test?) set have different identity labels
            # so we need to map the identity label of this set to the training set
            strategy = "random"
            if strategy == "random":
                # import random
                # pick a random identity label from the training set
                # return random.randint(0, len(self.identity_labels) - 1)
                idx = self._identity_rng.integers(low=0, high=len(self.identity_labels))
                return idx
            # elif strategy == "mod":
            #     # map the identity label of this set to the training set
            #     return self.identity_label2index[identity] % len(self.identity_labels) 
            else:
                raise ValueError(f"Unknown strategy '{strategy}'")
    
    def _set_identity_label(self, identity_labels, identity_label2index):
        if identity_labels == self.identity_labels and identity_label2index == self.identity_label2index:
            self.own_identity_label = True
            return 
        self.identity_labels = identity_labels
        self.identity_label2index = identity_label2index
        self.own_identity_label = False
        self._setup_identity_rng()

    def _setup_identity_rng(self):
        self._identity_rng = np.random.default_rng(12345)

    def _video_expression(self, index): 
        expr = self.video_list[self.video_indices[index]].parts[3] 
        if expr in ["front", "down", "top", "left_30",  "left_60", "right_30", "right_60", "1", "2"]: # a hack for MORONIC inconsistency in the paths of the MEAD dataset
            return self.video_list[self.video_indices[index]].parts[4]
        return expr

    def _video_name_idx(self, index): 
        intensity = self.video_list[self.video_indices[index]].stem
        return intensity


    def _expression_intensity(self, index): 
        intensity = self.video_list[self.video_indices[index]].parts[4]
        if "level" not in intensity: # a hack for MORONIC inconsistency in the paths of the MEAD dataset
            return self.video_list[self.video_indices[index]].parts[5]
        return intensity

    def _get_emotions(self, index, start_frame, num_read_frames, video_fps, num_frames, sample):
        sample = super()._get_emotions(index, start_frame, num_read_frames, video_fps, num_frames, sample)
        sample["gt_expression_label"] = get_affectnet_index_from_mead_expression_str(self._video_expression(index))
        sample["gt_expression_intensity"] = int(self._expression_intensity(index)[-1])
        sample["gt_expression_identity"] = self._get_identity_label_index(index)
        sample["gt_expression_label_with_intensity"] = get_index_for_expression_with_intensity(
            sample["gt_expression_label"], sample["gt_expression_intensity"])
        sample["gt_expression_label_with_intensity_identity"] = get_index_for_expression_with_intensity_identity(
             sample["gt_expression_label"], sample["gt_expression_intensity"], sample["gt_expression_identity"], 
            len(self.identity_labels),
            )
        
        # turn into array
        sample["gt_expression_label"] = np.array(sample["gt_expression_label"])
        sample["gt_expression_intensity"] = np.array(sample["gt_expression_intensity"])
        sample["gt_expression_identity"] = np.array(sample["gt_expression_identity"])
        sample["gt_expression_label_with_intensity"] = np.array(sample["gt_expression_label_with_intensity"])
        sample["gt_expression_label_with_intensity_identity"] = np.array(sample["gt_expression_label_with_intensity_identity"])
        return sample

    def _get_video_path(self, index):
        if self.use_original_video:
            video_path = self.root_path / self.video_list[self.video_indices[index]]
        else: 
            path_prefix = Path(self.output_dir) / "videos_aligned"  
            parts = self.video_list[self.video_indices[index]].parts
            video_part = parts[1] 
            expected_values = ["video", "1", "2"]
            assert video_part in expected_values, f"Expected video part to be one of {expected_values}, but got '{video_part}'"
            video_path = path_prefix / parts[0] / "/".join(parts[2:])
        return video_path

    def _get_audio_path(self, index):
        path_prefix = Path(self.output_dir) / "audio"  
        parts = self.video_list[self.video_indices[index]].parts
        video_part = parts[1] 
        expected_values = ["video", "1", "2"]
        assert video_part in expected_values, f"Expected video part to be one of {expected_values}, but got '{video_part}'"
        audio_path = path_prefix / parts[0] / "/".join(parts[2:])
        audio_path = audio_path.with_suffix(".wav")
        return audio_path


    def _path_to_segmentations(self, index): 
        parts = self.video_list[self.video_indices[index]].parts
        video_part = parts[1] 
        expected_values = ["video", "1", "2"]
        assert video_part in expected_values, f"Expected video part to be one of {expected_values}, but got '{video_part}'"
        path_prefix = Path(self.output_dir) / f"segmentations_{self.segmentation_source}" / self.segmentation_type
        seg_path = path_prefix / parts[0] / "/".join(parts[2:])
        return seg_path.with_suffix("")

    def _path_to_landmarks(self, index, landmark_type, landmark_source): 
        parts = self.video_list[self.video_indices[index]].parts
        video_part = parts[1] 
        expected_values = ["video", "1", "2"]
        assert video_part in expected_values, f"Expected video part to be one of {expected_values}, but got '{video_part}'"
        path_prefix = Path(self.output_dir) / f"landmarks_{landmark_source}/{landmark_type}" 
        lmk_path = path_prefix / parts[0] / "/".join(parts[2:])
        return lmk_path.with_suffix("")

    def _path_to_reconstructions(self, index, rec_type): 
        parts = self.video_list[self.video_indices[index]].parts
        video_part = parts[1] 
        expected_values = ["video", "1", "2"]
        assert video_part in expected_values, f"Expected video part to be one of {expected_values}, but got '{video_part}'"
        # path_prefix = Path(self.output_dir) / f"reconstructions" / self.reconstruction_type 
        path_prefix = Path(self.output_dir) / f"reconstructions" / rec_type
        rec_path = path_prefix / parts[0] / "/".join(parts[2:])
        return rec_path.with_suffix("")

    def _path_to_emotions(self, index): 
        parts = self.video_list[self.video_indices[index]].parts
        video_part = parts[1] 
        expected_values = ["video", "1", "2"]
        assert video_part in expected_values, f"Expected video part to be one of {expected_values}, but got '{video_part}'"
        path_prefix = Path(self.output_dir) / f"emotions" / self.emotion_type 
        emo_path = path_prefix / parts[0] / "/".join(parts[2:])
        return emo_path.with_suffix("")

    def _get_text(self, sample, index):
        if self.read_gt_text:
            if not hasattr(self, "_gt_text"):
                path_to_text = Path(self.output_dir) / "list_full_mead_annotated.txt"
                with open(path_to_text, "r") as f:
                    # read the whole file 
                    gt_text = f.read()
                lines = gt_text.split("\n")
                self._gt_text = {line[:line.index(' ')]: line[line.index(' ')+1:] for line in lines}

            expression = self._video_expression(index)
            intensity = self._expression_intensity(index)
            identity = self._get_identity_label(index)
            vid_fname_idx = self._video_name_idx(index)
            key = f"{identity}_{intensity}_{expression}_{vid_fname_idx}"
            sample["gt_text"] = self._gt_text[key]
        return sample

    def _getitem(self, index):
        sample = super()._getitem(index)
        sample = self._get_text(sample, index)
        return sample

