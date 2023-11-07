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


class CelebVHQDataModule(FaceVideoDataModule): 

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
            split = "original", #TODO: does CelebVHQ offer any split?
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
            segmentation_type=None,
            read_video = True,
            read_audio = True,
            align_images = True,
            return_mica_images = False,
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
            read_video = read_video,
            read_audio = read_audio,
            return_mica_images = return_mica_images,
            align_images=align_images,
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
        self.annotation_json_path = Path(root_dir).parent / "celebvhq_info.json" 
        assert self.annotation_json_path.is_file()

        self.landmark_types = landmark_types or ["mediapipe", "fan"]
        self.landmark_sources = landmark_sources or ["original", "aligned"]
        self.segmentation_source = segmentation_source or "aligned"
        self.segmentation_type = segmentation_type or "focus"
        self.use_original_video = False

    def prepare_data(self):
        # super().prepare_data()
        
        # outdir = Path(self.output_dir)

        # # is dataset already processed?
        # if outdir.is_dir():
        if Path(self.metadata_path).is_file():
            print("The dataset is already processed. Loading")
            self._loadMeta()
            return
        # # else:
        self._gather_data()
        self._saveMeta()
        self._loadMeta()
        # self._unpack_videos()
        # self._saveMeta()

    def _gather_data(self, exist_ok=True):
        super()._gather_data(exist_ok)
        
        # vl = [(path.parent / path.stem).as_posix() for path in self.video_list]
        # al = [(path.parent / path.stem).as_posix() for path in self.annotation_list]

        # vl_set = set(vl)
        # al_set = set(al)

        # vl_diff = vl_set.difference(al_set)
        # al_diff = al_set.difference(vl_set)

        # intersection = vl_set.intersection(al_set) 

        # print(f"Video list: {len(vl_diff)}")
        # print(f"Annotation list: {len(al_diff)}")

        # if len(vl_diff) != 0:
        #     print("Video list is not equal to annotation list") 
        #     print("Video list difference:")
        #     print(vl_diff)
        #     raise RuntimeError("Video list is not equal to annotation list")
        
        # if len(al_diff) != 0: 
        #     print("Annotation list is not equal to video list")    
        #     print("Annotation list difference:")
        #     print(al_diff)
        #     raise RuntimeError("Annotation list is not equal to video list")

        # print(f"Intersection: {len(intersection)}")


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
            # seg_methods = ['bisenet', 'focus']
            seg_methods = ['focus']
            for seg_method in seg_methods:
                self._segment_faces_in_sequence(idx, use_aligned_videos=True, segmentation_net=seg_method)
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
  
        if segmentations_to_hdf5:
            seg_methods = ['bisenet', 'focus']
            for seg_method in seg_methods:
                self._segmentations_to_hdf5(idx, segmentation_net=seg_method, use_aligned_videos=True)


    def _process_shard(self, videos_per_shard, shard_idx, extract_audio=True,
        restore_videos=True, detect_landmarks=True, segment_videos=True, 
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
                "emotions", "reconstructions", "audio"]
        video_file = self.video_list[sequence_id]
        if len(method) > 0:
            file_type += "_" + method 
        if len(suffix) > 0:
            file_type += suffix

        suffix = Path(file_type) / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder


    def _get_subsets(self, set_type=None):
        set_type = set_type or "original"
        self.temporal_split = None
        if "specific_video_temporal" in set_type: 
            res = set_type.split("_")
            # the video name should be enclosed in single quotes 'vid_name'
            # vid = set_type.split("'")
            vid = "_".join(res[3:-3])
            assert len(vid) > 0 # the result should be 3 elements
            # vid = vid[1] # the video name is the second (middle) element
            train = float(res[-3])
            val = float(res[-2])
            test = float(res[-1])
            train_ = train / (train + val + test)
            val_ = val / (train + val + test)
            test_ = test / (train + val + test)
            indices = [i for i in range(len(self.video_list)) if self.video_list[i].stem == vid]
            assert len(indices) == 1
            training = [indices[0]]
            validation = [indices[0]]
            test = [indices[0]]
            self.temporal_split = [train_, val_, test_]
            self.preload_videos = True
            self.inflate_by_video_size = True
            return training, validation, test
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
            # this means first part of the videos are training, second part is validation, third part is test (if any)
            res = set_type.split("_")
            assert len(res) >= 3, "Specify the train/val/test split by 'temporal_train_val_test' to the set_type"
            train = int(res[1])
            val = int(res[2])
            if len(res) == 4:
                test = int(res[3])
            else:
                test = 0
            train_ = train / (train + val + test)
            val_ = val / (train + val + test)
            test_ = 1 - train_ - val_
            self.temporal_split = [train_, val_, test_]
            pretrain = list(range(len(self.video_list)))
            trainval = list(range(len(self.video_list)))
            test = list(range(len(self.video_list)))
            return pretrain, trainval, test
        elif set_type == "all":
            pretrain = list(range(len(self.video_list)))
            trainval = list(range(len(self.video_list)))
            test = list(range(len(self.video_list)))
            return pretrain, trainval, test
        else: 
            raise ValueError(f"Unknown set type: {set_type}")


    def setup(self, stage=None):
        train, val, test = self._get_subsets(self.split)
        training_augmenter = create_image_augmenter(self.image_size, self.augmentation)
        self.training_set = CelebVHQDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, train, 
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
                segmentation_type=self.segmentation_type,
                temporal_split_start= 0 if self.temporal_split is not None else None,
                temporal_split_end=self.temporal_split[0] if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,
                original_image_size=self.processed_video_size,
                return_mica_images=self.return_mica_images,
                align_images=self.align_images,
              )
                    
        self.validation_set = CelebVHQDataset(self.root_dir, self.output_dir, 
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
                segmentation_type=self.segmentation_type,
                temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,
                original_image_size=self.processed_video_size,
                return_mica_images=self.return_mica_images,
                align_images=self.align_images,
            )

        self.test_set = CelebVHQDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, test, self.audio_metas, 
                self.sequence_length_test, image_size=self.image_size, 
                **self.occlusion_settings_test,
                hack_length=False, 
                use_original_video=self.use_original_video,
                include_processed_audio = self.include_processed_audio,
                include_raw_audio = self.include_raw_audio,
                landmark_types=self.landmark_types,
                landmark_source=self.landmark_sources,
                segmentation_source=self.segmentation_source,
                segmentation_type=self.segmentation_type,
                temporal_split_start=self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                temporal_split_end= sum(self.temporal_split) if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,
                original_image_size=self.processed_video_size,
                return_mica_images=self.return_mica_images,
                align_images=self.align_images,
                )

    def get_single_video_dataset(self, i):
        dataset = CelebVHQDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
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
                landmark_types=self.landmark_types,
                landmark_source=self.landmark_sources,
                segmentation_source=self.segmentation_source,
                segmentation_type=self.segmentation_type,
                # temporal_split_start=self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                # temporal_split_end= sum(self.temporal_split) if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=False,
                original_image_size=self.processed_video_size,
                return_mica_images=self.return_mica_images,
                align_images=self.align_images,
                )
        dataset._allow_alignment_fail = False
        return dataset


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
                        persistent_workers=self.num_workers > 0,
                        )
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.validation_set, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.batch_size_val, 
                        #   drop_last=self.drop_last
                          drop_last=False, 
                        collate_fn=robust_collate, 
                        persistent_workers=self.num_workers > 0,
                        )
        if hasattr(self, "validation_set_2"): 
            dl2 =  DataLoader(self.validation_set_2, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                            batch_size=self.batch_size_val, 
                            # drop_last=self.drop_last, 
                            drop_last=False, 
                            collate_fn=robust_collate, 
                            persistent_workers=self.num_workers > 0,
                            )
                            
            return [dl, dl2]
        return dl 

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.batch_size_test, drop_last=self.drop_last)

import imgaug
from inferno.datasets.VideoDatasetBase import VideoDatasetBase

class CelebVHQDataset(VideoDatasetBase):

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
            segmentation_type = "focus",
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
            original_image_size=None,
            return_mica_images=False,
            align_images=True,
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
            original_image_size=original_image_size,
            return_mica_images=return_mica_images,
            align_images=align_images,
        )


    def _path_to_landmarks(self, index, landmark_type, landmark_source): 
        return (Path(self.output_dir) / f"landmarks_{landmark_source}_{landmark_type}" /  self.video_list[self.video_indices[index]]).with_suffix("")


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
            landmarks_dir = (Path(self.output_dir) / f"landmarks_{landmark_source}_{landmark_type}" /  self.video_list[self.video_indices[index]]).with_suffix("")
            landmarks = []
            if landmark_source == "original":
                # landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / f"landmarks_{landmark_source}.pkl")  
                # landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / f"landmarks_aligned_video.pkl")  
                landmark_list_file = landmarks_dir / f"landmarks_aligned_video_smoothed.pkl"
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
                # landmarks = landmarks * sample["video"].shape[1]
                landmarks = landmarks * self.original_image_size

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


    def _path_to_segmentations(self, index): 
        return (Path(self.output_dir) / f"segmentations_{self.segmentation_source}_{self.segmentation_type}" /  self.video_list[self.video_indices[index]]).with_suffix("")


    def _path_to_reconstructions(self, index, rec_type): 
        return (Path(self.output_dir) / f"reconstructions_{rec_type}" /  self.video_list[self.video_indices[index]]).with_suffix("")
        # return (Path(self.output_dir) / f"reconstructions_{self.reconstruction_type}" /  self.video_list[self.video_indices[index]]).with_suffix("")


    def _path_to_emotions(self, index): 
        return (Path(self.output_dir) / f"emotions_{self.emotion_type}" /  self.video_list[self.video_indices[index]]).with_suffix("")


def main(): 
    import time
    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/celebvhq/auto_processed_online_25fps")
    # output_dir = Path("/is/cluster/work/rdanecek/data/celebvhq/")

    root_dir = Path("/ps/project/EmotionalFacialAnimation/data/celebvhq/auto_processed_combined_25fps")
    output_dir = Path("/is/cluster/work/rdanecek/data/celebvhq/")

    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs2/mvlrs_v1")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs2")

    # processed_subfolder = "processed"
    processed_subfolder = "processed_orig"

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
    dm = CelebVHQDataModule(
        root_dir, output_dir, processed_subfolder,
        split="all",
        # split="temporal_80_10_10",
        # split="specific_video_temporal_z0ecgTX08pI_0_1_80_10_10",  # missing audio
        # split="specific_video_temporal_8oKLUz8phdg_1_0_80_10_10",
        # split="specific_video_temporal_eknCAJ0ik8c_0_0_80_10_10",
        # split="specific_video_temporal_YgJ5ZEn67tk_2_80_10_10",
        # split="specific_video_temporal_zZrDihnANpM_4_80_10_10", 
        # split = "specific_video_temporal_6jRVZQMKlxw_1_0_80_10_10",
        # split = "specific_video_temporal_6jRVZQMKlxw_1_0_80_10_10",
        # split="specific_video_temporal_T0BMVyJ1OXk_0_0_80_10_10", # missing audio
        # split="specific_video_temporal_2T3YWtHj_Ag_0_0_80_10_10", # missing audio
        # split="specific_video_temporal_7Eha1lreIyg_0_0_80_10_10", # missing audio
        # split="specific_video_temporal_UHY7k99ugXc_0_2_80_10_10", # missing audio
        # split="specific_video_temporal_e4Ylz6WgBrg_0_0_80_10_10", # missing audio
        # split="specific_video_temporal_Px5769-CPaQ_0_0_80_10_10", # missing audio
        # split="specific_video_temporal_HhlT8RJaQEY_0_0_80_10_10", # missing audio
        # split="specific_video_temporal_eQZ-f9Vll3c_0_0_80_10_10", # missing audio
        # split="specific_video_temporal_IubhiJFulKk_2_0_80_10_10", # missing audio
        # split="specific_video_temporal_uYC1dIPHoRQ_0_0_80_10_10", # missing audio
        # split="specific_video_temporal_20n3XeaEd1c_0_0_80_10_10", # missing audio
        # split="specific_video_temporal_CWdm32em3xQ_0_80_10_10",
        # split="specific_video_temporal_Px5769-CPaQ_0_0_80_10_10",  # missing audio
        # split="specific_video_temporal_HhlT8RJaQEY_0_0_80_10_10",  # missing audio
        # split="specific_video_temporal_uYC1dIPHoRQ_0_0_80_10_10", # missing audio
        # split="specific_video_temporal_OfVYgE_hT88_0_0_80_10_10", # missing audio
        # split="specific_video_temporal_lBwtMLK_qEE_1_0_80_10_10", # missing audio
        # split="specific_video_temporal_Gq17Orwh4c4_9_1_80_10_10", # missing audio
        # split="specific_video_temporal_-rjR4El7qzg_4_80_10_10",
        # split="specific_video_temporal_lBwtMLK_qEE_1_0_80_10_10",
        # split="specific_video_temporal_lBwtMLK_qEE_1_0_80_10_10",
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
