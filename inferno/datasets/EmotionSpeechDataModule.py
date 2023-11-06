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

class EmotionalSpeechDataModule(FaceVideoDataModule): 

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
            ## begin CelebVHQDataModule specific params
            training_sampler="uniform",
            landmark_types = None,
            landmark_sources=None,
            segmentation_source=None,
            viewing_angles=None,
            read_video=True,
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
        self.annotation_json_path = None # Path(root_dir).parent / "celebvhq_info.json" 
        ## assert self.annotation_json_path.is_file()

        self.landmark_types = landmark_types or ["mediapipe", "fan"]
        self.landmark_sources = landmark_sources or ["original", "aligned"]
        self.segmentation_source = segmentation_source or "aligned"
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
        dataset = EmotionalSpeechDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                [i], 
                self.audio_metas, 
                # self.sequence_length_test, 
                "all", 
                image_size=self.image_size, 
                # **self.occlusion_settings_test,
                hack_length=False, 
                use_original_video=self.use_original_video,
                include_processed_audio = self.include_processed_audio,
                include_raw_audio = self.include_raw_audio,
                # landmark_types=self.landmark_types,
                landmark_types="mediapipe",
                # landmark_source=self.landmark_sources,
                landmark_source="original",
                segmentation_source=self.segmentation_source,
                # temporal_split_start=self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                # temporal_split_end= sum(self.temporal_split) if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=False,
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

    def _filename2index(self, filename):
        return self.video_list.index(filename)

    def _get_landmark_method(self):
        return self.face_detector_type

    def _get_segmentation_method(self):
        return "bisenet"

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
            self._segment_faces_in_sequence(idx, use_aligned_videos=True)
            # raise NotImplementedError()
        if detect_aligned_landmarks: 
            self._detect_landmarkes_in_aligned_sequence(idx)

        if reconstruct_faces: 
            # self._reconstruct_faces_in_sequence(idx, 
            #     reconstruction_net=self._get_reconstruction_network('emoca'))
            # self._reconstruct_faces_in_sequence(idx, 
            #     reconstruction_net=self._get_reconstruction_network('deep3dface'))
            # self._reconstruct_faces_in_sequence(idx, 
            #     reconstruction_net=self._get_reconstruction_network('deca'))
            # rec_methods = ['emoca', 'deep3dface', 'deca']
            # rec_methods = ['emoca', 'deep3dface',]
            # rec_methods = ['emoca',]
            rec_methods = ['emoca', 'spectre',]
            # for rec_method in rec_methods:
            #     self._reconstruct_faces_in_sequence(idx, reconstruction_net=None, device=None,
            #                         save_obj=False, save_mat=True, save_vis=False, save_images=False,
            #                         save_video=False, rec_method=rec_method, retarget_from=None, retarget_suffix=None)
            self._reconstruct_faces_in_sequence_v2(
                        idx, reconstruction_net=None, device=None,
                        save_obj=False, save_mat=True, save_vis=False, save_images=False,
                        save_video=False, rec_methods=rec_methods, retarget_from=None, retarget_suffix=None)
        if recognize_emotions:
            emo_methods = ['resnet50', ]
            self._extract_emotion_in_sequence(idx, emo_methods=emo_methods)

    def _process_shard(self, videos_per_shard, shard_idx, 
        extract_audio=True,
        restore_videos=True, 
        detect_landmarks=True, 
        segment_videos=True, 
        detect_aligned_landmarks=False,
        reconstruct_faces=False,
        recognize_emotions=False,
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
                )
            
        print("Done processing shard")

    def _get_path_to_sequence_files(self, sequence_id, file_type, method="", suffix="", assert_=True): 
        if assert_:
            assert file_type in ['videos', 'videos_aligned', 'detections', 
                "landmarks", "landmarks_original", "landmarks_aligned",
                "segmentations", "segmentations_aligned",
                "emotions", "reconstructions", "audio"]
        video_file = self.video_list[sequence_id]
        if len(method) > 0:
            file_type += "/" + method 
        if len(suffix) > 0:
            file_type += suffix

        if len(video_file.parts) == 6:
            expected_values = ["video", "1", "2"] # for some reason the authors are not consistent with folder names
            assert video_file.parts[1] in expected_values, f"Unexpected path structure. Expected one of {expected_values}, got {video_file.parts[1]}"
      
        # suffix = Path(file_type) / video_file.stem
        person_id = self._video_identity(sequence_id)
        
        suffix = Path(file_type) / person_id / "/".join(video_file.parts[2:-1]) / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder

    def _video_identity(self, index): 
        return self.video_list[index].parts[1]

    def _video_expression(self, index): 
        return self.video_list[index].parts[2]

    # def _expression_intensity(self, index): 
    #     return self.video_list[index].parts[4]

    # def _get_expression_intensity_map(self, indices):
    #     expression_intensity2idx = {}
    #     for i in indices:
    #         expression = self._video_expression(i)
    #         intensity = self._expression_intensity(i)
    #         key = (expression, intensity)
    #         if key not in expression_intensity2idx:
    #             expression_intensity2idx[key] = []
    #         expression_intensity2idx[key] += [i]
    #     return expression_intensity2idx

    # def _get_identity_expression_intensity_map(self, indices):
    #     identity_expression_intensity2idx = {}
    #     for i in indices:
    #         identity = self._video_identity(i)
    #         expression = self._video_expression(i)
    #         intensity = self._expression_intensity(i)
    #         key = (identity, expression, intensity)
    #         if key not in identity_expression_intensity2idx:
    #             identity_expression_intensity2idx[key] = []
    #         identity_expression_intensity2idx[key] += [i]
    #     return identity_expression_intensity2idx

    # def _get_identity_map(self, indices):
    #     identity2idx = {}
    #     for i in indices:
    #         identity = self._video_identity(i)
    #         key = identity
    #         if key not in identity2idx:
    #             identity2idx[key] = []
    #         identity2idx[key] += [i]
    #     return identity2idx

    def _get_subsets(self, set_type=None):
        raise NotImplementedError("Not implemented yet")
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
                rand.shuffle(identities)

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

        elif "random_by_identity" in set_type:
            # WARNING: THIS NAME IS NOT ACCURATE, IT IS NOT RANDOM BY IDENTITY BUT RANDOM BY EXPRESSION AND INTENSITY
            # SO ALL IDENTITIES ARE IN BOTH TRAIN AND VAL (BUT THE TRAIN AND VAL VIDEOS DON'T OVERLAP)
            # pretrain_02d_02d, such as pretrain_80_20 
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
                    rand.shuffle(idxs)
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
        elif "temporal" in set_type:
            raise NotImplementedError("Not implemented yet")
        else: 
            raise ValueError(f"Unknown set type: {set_type}")


    def setup(self, stage=None):
        train, val, test = self._get_subsets(self.split)
        training_augmenter = create_image_augmenter(self.image_size, self.augmentation)
        self.training_set = EmotionalSpeechDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, train, 
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
                temporal_split_start= 0 if self.temporal_split is not None else None,
                temporal_split_end=self.temporal_split[0] if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,
                read_video=self.read_video,
              )
                    
        self.validation_set = EmotionalSpeechDataset(self.root_dir, self.output_dir, 
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
                temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,
                read_video=self.read_video,
            )

        self.test_set = EmotionalSpeechDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, test, self.audio_metas, 
                self.sequence_length_test, image_size=self.image_size, 
                **self.occlusion_settings_test,
                hack_length=False, 
                use_original_video=self.use_original_video,
                include_processed_audio = self.include_processed_audio,
                include_raw_audio = self.include_raw_audio,
                landmark_types=self.landmark_types,
                landmark_source=self.landmark_sources,
                segmentation_source=self.segmentation_source,
                temporal_split_start=self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                temporal_split_end= sum(self.temporal_split) if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,
                read_video=self.read_video,
                )

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
        dl = DataLoader(self.validation_set, shuffle=False, num_workers=self.num_workers, pin_memory=True,
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

import imgaug
from inferno.datasets.VideoDatasetBase import VideoDatasetBase

class EmotionalSpeechDataset(VideoDatasetBase):

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
            image_size=None, 
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
            reconstruction_type=None,
            return_global_pose = False,
            return_appearance = False,
            average_shape_decode = True,
            emotion_type=None,
            return_emotion_feature=False,
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
            reconstruction_type=reconstruction_type,
            return_global_pose = return_global_pose,
            return_appearance = return_appearance,
            average_shape_decode = average_shape_decode,
            emotion_type=emotion_type,
            return_emotion_feature=return_emotion_feature,
        )

    def _read_landmarks(self, index, landmark_type, landmark_source):
        landmarks_dir = self._path_to_landmarks(index, landmark_type, landmark_source)
        if landmark_source == "original":
            landmark_list_file = landmarks_dir / f"landmarks_aligned_video_smoothed.pkl"
            landmark_list = FaceDataModuleBase.load_landmark_list(landmark_list_file)  
            landmark_valid_indices = FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmarks_alignment_used_frame_indices.pkl")  
        elif landmark_source == "aligned": 
            landmarks, landmark_confidences, landmark_types = FaceDataModuleBase.load_landmark_list_v2(landmarks_dir / f"landmarks.pkl")  
            landmark_valid_indices = landmark_confidences
        else: 
            raise ValueError(f"Unknown landmark source {landmark_source}")
        return landmark_list, landmark_valid_indices


    def _get_landmarks(self, index, start_frame, num_read_frames, video_fps, num_frames, sample): 
        sequence_length = self._get_sample_length(index)
        landmark_dict = {}
        landmark_validity_dict = {}
        for lti, landmark_type in enumerate(self.landmark_types):
            landmark_source = self.landmark_source[lti]
            # landmarks_dir = (Path(self.output_dir) / f"landmarks_{landmark_source}" / landmark_type /  self.video_list[self.video_indices[index]]).with_suffix("")
            # landmarks_dir = (Path(self.output_dir) / f"landmarks_{landmark_source}/{landmark_type}" /  self.video_list[self.video_indices[index]]).with_suffix("")
            landmarks_dir = self._path_to_landmarks(index, landmark_type, landmark_source)
            landmarks = []
            if landmark_source == "original":
                # landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / f"landmarks_{landmark_source}.pkl")  
                # landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / f"landmarks_aligned_video.pkl")  
                landmark_list_file = landmarks_dir / f"landmarks_aligned_video_smoothed.pkl"

                # if not self.preload_videos: 
                #     # landmark_list = FaceDataModuleBase.load_landmark_list(landm?arks_dir / f"landmarks_{landmark_source}.pkl")  
                #     landmark_list = self._read_landmarks(index, landmark_type, landmark_source)
                #     # landmark_types =  FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmark_types.pkl")  
                # else: 
                #     landmark_list = self.lmk_cache[index][landmark_type]
                #     # landmark_types = self.lmk_cache[index]["landmark_types"]

                if landmark_list_file.exists():
                    if not self.preload_videos:
                        landmark_list, landmark_valid_indices = self._read_landmarks(index, landmark_type, landmark_source)
                    else:
                        landmark_list, landmark_valid_indices = self.lmk_cache[index][landmark_type][landmark_source]
                        
                    # landmark_list = FaceDataModuleBase.load_landmark_list(landmark_list_file)  
                    # landmark_types =  FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmark_types.pkl")  
                    landmarks = landmark_list[start_frame: sequence_length + start_frame] 
                    landmarks = np.stack(landmarks, axis=0)

                    # landmark_valid_indices = FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmarks_alignment_used_frame_indices.pkl")  
                    landmark_validity = np.zeros((len(landmark_list), 1), dtype=np.float32)
                    landmark_validity[landmark_valid_indices] = 1.0
                    landmark_validity = landmark_validity[start_frame: sequence_length + start_frame]
                else:
                    if landmark_type == "mediapipe":
                        num_landmarks = MEDIAPIPE_LANDMARK_NUMBER
                    elif landmark_type in ["fan", "kpt68"]:
                        num_landmarks = 68
                    landmarks = np.zeros((sequence_length, num_landmarks, 2), dtype=np.float32)
                    landmark_validity = np.zeros((sequence_length, 1), dtype=np.float32)
                    # landmark_validity = landmark_validity.squeeze(-1)


            elif landmark_source == "aligned":
                if not self.preload_videos:
                    landmarks, landmark_confidences = self._read_landmarks(index, landmark_type, landmark_source)
                    # landmarks, landmark_confidences, landmark_types = FaceDataModuleBase.load_landmark_list_v2(landmarks_dir / f"landmarks.pkl")  
                else: 
                    landmarks, landmark_confidences = self.lmk_cache[index][landmark_type][landmark_source]

                # scale by image size 
                landmarks = landmarks * sample["video"].shape[1]

                landmarks = landmarks[start_frame: sequence_length + start_frame]
                # landmark_confidences = landmark_confidences[start_frame: sequence_length + start_frame]
                # landmark_validity = landmark_confidences #TODO: something is wrong here, the validity is not correct and has different dimensions
                landmark_validity = None 
            
            else: 
                raise ValueError(f"Invalid landmark source: '{landmark_source}'")

            # landmark_validity = np.ones(len(landmarks), dtype=np.bool)
            # for li in range(len(landmarks)): 
            #     if len(landmarks[li]) == 0: # dropped detection
            #         if landmark_type == "mediapipe":
            #             # [WARNING] mediapipe landmarks coordinates are saved in the scale [0.0-1.0] (for absolute they need to be multiplied by img size)
            #             landmarks[li] = np.zeros((478, 3))
            #         elif landmark_type in ["fan", "kpt68"]:
            #             landmarks[li] = np.zeros((68, 2))
            #         else: 
            #             raise ValueError(f"Unknown landmark type '{landmark_type}'")
            #         landmark_validity[li] = False
            #     elif len(landmarks[li]) > 1: # multiple faces detected
            #         landmarks[li] = landmarks[li][0] # just take the first one for now
            #     else: \
            #         landmarks[li] = landmarks[li][0] 

            # landmarks = np.stack(landmarks, axis=0)

            # pad landmarks with zeros if necessary to match the desired video length
            if landmarks.shape[0] < sequence_length:
                landmarks = np.concatenate([landmarks, np.zeros(
                    (sequence_length - landmarks.shape[0], *landmarks.shape[1:]), 
                    dtype=landmarks.dtype)], axis=0)
                if landmark_validity is not None:
                    landmark_validity = np.concatenate([landmark_validity, np.zeros((sequence_length - landmark_validity.shape[0], landmark_validity.shape[1]), 
                        dtype=landmark_validity.dtype)], axis=0)
                else: 
                    landmark_validity = np.zeros((sequence_length, 1), dtype=np.float32)

            landmark_dict[landmark_type] = landmarks
            if landmark_validity is not None:
                landmark_validity_dict[landmark_type] = landmark_validity

        sample["landmarks"] = landmark_dict
        sample["landmarks_validity"] = landmark_validity_dict
        return sample

    def _get_video_path(self, index):
        if self.use_original_video:
            video_path = self.root_path / self.video_list[self.video_indices[index]]
        else: 
            path_prefix = Path(self.output_dir) / "videos_aligned"  
            parts = self.video_list[self.video_indices[index]].parts
            video_path = path_prefix /  "/".join(parts[1:])
        return video_path

    def _get_audio_path(self, index):
        path_prefix = Path(self.output_dir) / "audio"  
        parts = self.video_list[self.video_indices[index]].parts
        audio_path = path_prefix /  "/".join(parts[1:])
        audio_path = audio_path.with_suffix(".wav")
        return audio_path

    def _path_to_segmentations(self, index): 
        parts = self.video_list[self.video_indices[index]].parts
        path_prefix = Path(self.output_dir) / f"segmentations_{self.segmentation_source}" / self.segmentation_type
        seg_path = path_prefix /  "/".join(parts[1:])
        return seg_path.with_suffix("")

    def _path_to_landmarks(self, index, landmark_type, landmark_source): 
        parts = self.video_list[self.video_indices[index]].parts
        path_prefix = Path(self.output_dir) / f"landmarks_{landmark_source}/{landmark_type}" 
        lmk_path = path_prefix /  "/".join(parts[1:])
        return lmk_path.with_suffix("")

    def _path_to_reconstructions(self, index, rec_type): 
        parts = self.video_list[self.video_indices[index]].parts
        path_prefix = Path(self.output_dir) / f"reconstructions" / rec_type
        rec_path = path_prefix /  "/".join(parts[1:])
        return rec_path.with_suffix("")

    def _path_to_emotions(self, index): 
        parts = self.video_list[self.video_indices[index]].parts
        path_prefix = Path(self.output_dir) / f"emotions" / self.emotion_type 
        emo_path = path_prefix /  "/".join(parts[1:])
        return emo_path.with_suffix("")


def main(): 
    import time

    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/mead/MEAD")
    root_dir = Path("/is/cluster/work/rdanecek/data/mead_25fps/resampled_videos")
    # output_dir = Path("/is/cluster/work/rdanecek/data/mead/")
    output_dir = Path("/is/cluster/work/rdanecek/data/mead_25fps/")

    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs2/mvlrs_v1")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs2")

    processed_subfolder = "processed"
    # processed_subfolder = "processed_orig"

    # seq_len = 50
    seq_len = 16
    # bs = 100
    bs = 1


    import yaml
    from munch import Munch, munchify
    # augmenter = yaml.load(open(Path(__file__).parents[2] / "inferno_apps" / "Speech4D" / "tempface_conf" / "data" / "augmentations" / "default_no_jpeg.yaml"), 
    #     Loader=yaml.FullLoader)["augmentation"]
    # augmenter = munchify(augmenter)
    augmenter = None
    
    occlusion_settings_train = None
    # occlusion_settings_train = {
    #     "occlusion_length": [5, 15],
    #     "occlusion_probability_mouth": 0.5,
    #     "occlusion_probability_left_eye": 0.33,
    #     "occlusion_probability_right_eye": 0.33,
    #     "occlusion_probability_face": 0.2,
    # }
# occlusion_settings_val:
#     occlusion_length: [5, 10]
#     occlusion_probability_mouth: 1.0
#     occlusion_probability_left_eye: 0.33
#     occlusion_probability_right_eye: 0.33
#     occlusion_probability_face: 0.2

# occlusion_settings_test:
#     occlusion_length: [5, 10]
#     occlusion_probability_mouth: 1.0
#     occlusion_probability_left_eye: 0.33
#     occlusion_probability_right_eye: 0.33
#     occlusion_probability_face: 0.2

    # Create the dataset
    dm = EmotionalSpeechDataset(
        root_dir, output_dir, processed_subfolder,
        split="specific_identity_sorted_80_20_M003",
        # split="temporal_80_10_10",
        image_size=224, 
        scale=1.25, 
        processed_video_size=256,
        batch_size_train=bs,
        batch_size_val=bs,
        batch_size_test=bs,
        sequence_length_train=seq_len,
        sequence_length_val=seq_len,
        sequence_length_test=seq_len,
        num_workers=8,            
        include_processed_audio = True,
        include_raw_audio = True,
        augmentation=augmenter,
        occlusion_settings_train=occlusion_settings_train,
        landmark_types = ["mediapipe"],
        landmark_sources=["original"],
    )

    # Create the dataloader
    dm.prepare_data() 
    dm.setup() 

    dl = dm.train_dataloader()
    # dl = dm.val_dataloader()
    dataset = dm.training_set
    print( f"Dataset length: {len(dataset)}")
    # dataset = dm.validation_set
    indices = np.arange(len(dataset), dtype=np.int32)
    np.random.shuffle(indices)

    for i in range(len(indices)): 
        start = time.time()
        sample = dataset[indices[i]]
        end = time.time()
        print(f"Loading sample {i} took {end-start:.3f} s")
        dataset.visualize_sample(sample)

    # from tqdm import auto
    # for bi, batch in enumerate(auto.tqdm(dl)): 
    #     pass


    # iter_ = iter(dl)
    # for i in range(len(dl)): 
    #     start = time.time()
    #     batch = next(iter_)
    #     end = time.time()
    #     print(f"Loading batch {i} took {end-start:.3f} s")
        # dataset.visualize_batch(batch)

    #     break

    # dm._segment_faces_in_sequence(0)
    # idxs = np.arange(dm.num_sequences)
    # np.random.seed(0)
    # np.random.shuffle(idxs)

    # for i in range(dm.num_sequences):
    #     dm._deep_restore_sequence_sr_res(idxs[i])

    # dm.setup()



if __name__ == "__main__": 
    main()
