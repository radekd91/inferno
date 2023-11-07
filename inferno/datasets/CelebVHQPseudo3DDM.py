from inferno.datasets.CelebVHQDataModule import CelebVHQDataModule, CelebVHQDataset, robust_collate
from inferno.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp 
from inferno.datasets.ConditionedVideoTestDatasetWrapper import ConditionedVideoTestDatasetWrapper

import numpy as np
import torch


class CelebVHQPseudo3DDM(CelebVHQDataModule): 

    def __init__(self, root_dir, output_dir, 
            processed_subfolder=None, 
            face_detector='mediapipe', 
            landmarks_from=None, 
            face_detector_threshold=0.5, 
            image_size=224, 
            scale=1.25, 
            processed_video_size=384, 
            batch_size_train=16, 
            batch_size_val=16, 
            batch_size_test=16, 
            sequence_length_train=16, 
            sequence_length_val=16, 
            sequence_length_test=16, 
            bb_center_shift_x=0, 
            bb_center_shift_y=0, 
            occlusion_settings_train=None, 
            occlusion_settings_val=None, 
            occlusion_settings_test=None, 
            split="original", 
            num_workers=4, 
            device=None, 
            augmentation=None, 
            drop_last=True, 
            include_processed_audio=True, 
            include_raw_audio=True, 
            preload_videos=False, 
            inflate_by_video_size=False, 
            training_sampler="uniform", 
            landmark_types=None, 
            landmark_sources=None, 
            segmentation_source=None, 
            test_condition_source=None, 
            test_condition_settings=None,
            read_video=True,
            read_audio=True,
            reconstruction_type=None, 
            return_global_pose= False,
            return_appearance= False,
            average_shape_decode= True,
            emotion_type=None,
            return_emotion_feature=False,
            ):
        super().__init__(root_dir, output_dir, processed_subfolder, face_detector, landmarks_from, 
            face_detector_threshold, image_size, scale, 
            processed_video_size, batch_size_train, batch_size_val, batch_size_test, 
            sequence_length_train, sequence_length_val, sequence_length_test, 
            bb_center_shift_x, bb_center_shift_y, 
            occlusion_settings_train, occlusion_settings_val, occlusion_settings_test, 
            split, 
            num_workers, device, augmentation, drop_last, include_processed_audio, include_raw_audio, preload_videos, inflate_by_video_size, training_sampler, landmark_types, landmark_sources, segmentation_source,
            read_video = read_video,
            read_audio = read_audio,
            )

        self.test_condition_source = test_condition_source or "original"
        self.test_condition_settings = test_condition_settings
        # self.read_video = read_video

        self.reconstruction_type = reconstruction_type
        if self.reconstruction_type is not None: 
            if isinstance(self.reconstruction_type, str): 
                self.reconstruction_type = [self.reconstruction_type]
            elif isinstance(self.reconstruction_type, omegaconf.listconfig.ListConfig): 
                self.reconstruction_type = list(self.reconstruction_type)
            assert isinstance(self.reconstruction_type, list), "reconstruction_type must be a list or None"

        self.return_global_pose = return_global_pose
        self.return_appearance = return_appearance
        self.average_shape_decode = average_shape_decode

        self.emotion_type = emotion_type
        self.return_emotion_feature = return_emotion_feature


    def setup(self, stage=None):
        train, val, test = self._get_subsets(self.split)
        # training_augmenter = create_image_augmenter(self.image_size, self.augmentation)
        training_augmenter = None
        self.training_set = CelebVHQPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, train, 
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
                read_audio=self.read_audio,
                reconstruction_type=self.reconstruction_type,
                return_global_pose=self.return_global_pose,
                return_appearance=self.return_appearance,
                average_shape_decode=self.average_shape_decode,
                emotion_type=self.emotion_type,
                return_emotion_feature=self.return_emotion_feature,
              )
                    
        self.validation_set = CelebVHQPseudo3dDataset(self.root_dir, self.output_dir, 
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
                read_audio=self.read_audio,
                reconstruction_type=self.reconstruction_type,
                return_global_pose=self.return_global_pose,
                return_appearance=self.return_appearance,
                average_shape_decode=self.average_shape_decode,
                emotion_type=self.emotion_type,
                return_emotion_feature=self.return_emotion_feature,
            )

        max_test_videos = 5
        self.test_set_names = []
        self.test_set_ = CelebVHQPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                test[:max_test_videos], 
                self.audio_metas, 
                # sequence_length=self.sequence_length_test, 
                sequence_length="all", 
                image_size=self.image_size, 
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
                # inflate_by_video_size=self.inflate_by_video_size,
                inflate_by_video_size=False,
                include_filename=True,

                read_video=self.read_video,
                read_audio=self.read_audio,
                reconstruction_type=self.reconstruction_type,
                return_global_pose=self.return_global_pose,
                return_appearance=self.return_appearance,
                average_shape_decode=self.average_shape_decode,
                emotion_type=self.emotion_type,
                return_emotion_feature=self.return_emotion_feature,
                )

        self.test_set = ConditionedVideoTestDatasetWrapper(
            self.test_set_,
            None, 
            None,
            key_prefix="gt_",
        )

        self.test_set_names += ["test"]

        max_training_test_samples = 2
        self.test_set_train_ = CelebVHQPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                train[:max_training_test_samples], 
                self.audio_metas, 
                # sequence_length=self.sequence_length_test, 
                sequence_length="all",
                image_size=self.image_size, 
                **self.occlusion_settings_test,
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
                # inflate_by_video_size=self.inflate_by_video_size,
                inflate_by_video_size=False,
                include_filename=True,

                read_video=self.read_video,
                read_audio=self.read_audio,
                reconstruction_type=self.reconstruction_type,
                return_global_pose=self.return_global_pose,
                return_appearance=self.return_appearance,
                average_shape_decode=self.average_shape_decode,
                emotion_type=self.emotion_type,
                return_emotion_feature=self.return_emotion_feature,
                )

        self.test_set_train = ConditionedVideoTestDatasetWrapper(
            self.test_set_train_,
            None, 
            None,
            key_prefix="gt_",
        )        
        self.test_set_names += ["train"]

        max_validation_test_samples = 2
        self.test_set_val_ = CelebVHQPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                val[:max_validation_test_samples], 
                self.audio_metas, 
                # sequence_length=self.sequence_length_test, 
                sequence_length="all", 
                image_size=self.image_size, 
                **self.occlusion_settings_test,
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
                # inflate_by_video_size=self.inflate_by_video_size,
                inflate_by_video_size=False,
                include_filename=True,

                read_video=self.read_video,
                read_audio=self.read_audio,
                reconstruction_type=self.reconstruction_type,
                return_global_pose=self.return_global_pose,
                return_appearance=self.return_appearance,
                average_shape_decode=self.average_shape_decode,
                emotion_type=self.emotion_type,
                return_emotion_feature=self.return_emotion_feature,
                )

        self.test_set_val = ConditionedVideoTestDatasetWrapper(
            self.test_set_val_,
            None, 
            None,
            key_prefix="gt_",
        )        
        self.test_set_names += ["val"]

        # conditioned test set

        if self.test_condition_source != "original":
            self.test_set_cond_ = CelebVHQPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                    test[:max_test_videos], 
                    self.audio_metas, 
                    # sequence_length=self.sequence_length_test, 
                    sequence_length="all", 
                    image_size=self.image_size, 
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
                    # inflate_by_video_size=self.inflate_by_video_size,
                    inflate_by_video_size=False,
                    include_filename=True,

                    read_video=self.read_video,
                    read_audio=self.read_audio,
                    reconstruction_type=self.reconstruction_type,
                    return_global_pose=self.return_global_pose,
                    return_appearance=self.return_appearance,
                    average_shape_decode=self.average_shape_decode,
                    emotion_type=self.emotion_type,
                    return_emotion_feature=self.return_emotion_feature,
                    )

            self.test_set_cond = ConditionedVideoTestDatasetWrapper(
                self.test_set_cond_,
                self.test_condition_source, 
                self.test_condition_settings, 
                key_prefix="gt_",
            )

            self.test_set_names += ["test_cond"]

            max_training_test_samples = 2
            self.test_set_train_cond_ = CelebVHQPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                    train[:max_training_test_samples], 
                    self.audio_metas, 
                    # sequence_length=self.sequence_length_test, 
                    sequence_length="all",
                    image_size=self.image_size, 
                    **self.occlusion_settings_test,
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
                    # inflate_by_video_size=self.inflate_by_video_size,
                    inflate_by_video_size=False,
                    include_filename=True,

                    read_video=self.read_video,
                    read_audio=self.read_audio,
                    reconstruction_type=self.reconstruction_type,
                    return_global_pose=self.return_global_pose,
                    return_appearance=self.return_appearance,
                    average_shape_decode=self.average_shape_decode,
                    emotion_type=self.emotion_type,
                    return_emotion_feature=self.return_emotion_feature,
                    )

            self.test_set_train_cond = ConditionedVideoTestDatasetWrapper(
                self.test_set_train_cond_,
                self.test_condition_source, 
                self.test_condition_settings, 
                key_prefix="gt_",
            )        
            self.test_set_names += ["train_cond"]

            max_validation_test_samples = 2
            self.test_set_val_cond_ = CelebVHQPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                    val[:max_validation_test_samples], 
                    self.audio_metas, 
                    # sequence_length=self.sequence_length_test, 
                    sequence_length="all", 
                    image_size=self.image_size, 
                    **self.occlusion_settings_test,
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
                    # inflate_by_video_size=self.inflate_by_video_size,
                    inflate_by_video_size=False,
                    include_filename=True,

                    read_video=self.read_video,
                    read_audio=self.read_audio,
                    reconstruction_type=self.reconstruction_type,
                    return_global_pose=self.return_global_pose,
                    return_appearance=self.return_appearance,
                    average_shape_decode=self.average_shape_decode,
                    emotion_type=self.emotion_type,
                    return_emotion_feature=self.return_emotion_feature,
                    )

            self.test_set_val_cond = ConditionedVideoTestDatasetWrapper(
                self.test_set_val_cond_,
                self.test_condition_source, 
                self.test_condition_settings, 
                key_prefix="gt_",
            )        
            self.test_set_names += ["val_cond"]



    def test_dataloader(self):
        test_dls = []
        test_dl = super().test_dataloader()
        if test_dl is not None:
            if not isinstance(test_dl, list): 
                test_dl = [test_dl]
            test_dls += test_dl

        test_dls += [torch.utils.data.DataLoader(self.test_set_train, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.batch_size_test, 
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )]

        test_dls += [torch.utils.data.DataLoader(self.test_set_val, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.batch_size_test, 
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )]

        
        if hasattr(self, 'test_set_cond') and self.test_set_cond is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_cond, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                    batch_size=self.batch_size_test, 
                    drop_last=False,
                #   drop_last=self.drop_last,
                    collate_fn=robust_collate
                    )]

        if hasattr(self, 'test_set_train_cond') and self.test_set_train_cond is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_train_cond, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.batch_size_test, 
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )]

        if hasattr(self, 'test_set_val_cond') and self.test_set_val_cond is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_val_cond, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                            batch_size=self.batch_size_test, 
                            drop_last=False,
                            #   drop_last=self.drop_last,
                            collate_fn=robust_collate
                            )]

        return test_dls



class CelebVHQPseudo3dDataset(CelebVHQDataset):

    def __init__(self, 
            root_path, 
            output_dir, 
            video_list, 
            video_metas, 
            video_indices, 
            audio_metas, 
            sequence_length, 
            audio_noise_prob=0, 
            stack_order_audio=4, 
            audio_normalization="layer_norm", 
            landmark_types=None, 
            segmentation_type="bisenet", 
            landmark_source="original", 
            segmentation_source="aligned", 
            occlusion_length=0, 
            occlusion_probability_mouth=0, 
            occlusion_probability_left_eye=0, 
            occlusion_probability_right_eye=0, 
            occlusion_probability_face=0, 
            image_size=None, 
            transforms = None, 
            hack_length=False, use_original_video=False, include_processed_audio=True, include_raw_audio=True, 
            temporal_split_start=None, 
            temporal_split_end=None, 
            preload_videos=False, 
            inflate_by_video_size=False, 
            include_filename=False, # if True includes the filename of the video in the sample

            read_video=True,
            read_audio=True,
            reconstruction_type=None,
            return_global_pose=False,
            return_appearance=False,
            average_shape_decode=True,
            emotion_type=None,
            return_emotion_feature=False,
            ) -> None:
        super().__init__(root_path, output_dir, video_list, 
            video_metas, video_indices, audio_metas, sequence_length, audio_noise_prob, stack_order_audio, audio_normalization, 
            landmark_types, 
            segmentation_type, 
            landmark_source, 
            segmentation_source, 
            occlusion_length, 
            occlusion_probability_mouth, 
            occlusion_probability_left_eye, 
            occlusion_probability_right_eye, 
            occlusion_probability_face, 
            image_size, 
            transforms, 
            hack_length, 
            use_original_video, 
            include_processed_audio, 
            include_raw_audio, 
            temporal_split_start, 
            temporal_split_end, 
            preload_videos, 
            inflate_by_video_size, 
            include_filename=include_filename,
            )
            
        self.read_video = read_video
        self.read_audio = read_audio

        self.reconstruction_type = reconstruction_type
        if self.reconstruction_type is not None:
            self.return_global_pose = return_global_pose
            self.return_appearance = return_appearance
            self.average_shape_decode = average_shape_decode
            # self._load_flame()

        self.emotion_type = emotion_type
        self.return_emotion_feature = return_emotion_feature
            

    # def _get_landmarks(self, index, start_frame, num_read_frames, video_fps, num_frames, sample): 
    #     # don't load any landmarks+
    #     return sample

    def _get_landmarks(self, index, start_frame, num_read_frames, video_fps, num_frames, sample): 
    #     # don't load any landmarks+
        sample = super()._get_landmarks(index, start_frame, num_read_frames, video_fps, num_frames, sample)
        lmk_weights = sample["landmarks_validity"]["mediapipe"] / sample["landmarks_validity"]["mediapipe"].sum(axis=0, keepdims=True)
        assert np.any(np.isnan(lmk_weights)) == False, "NaN in weights" # throw an error if there are NaNs, this should cause a new sample to be loaded
        return sample

    def _get_segmentations(self, index, start_frame, num_read_frames, video_fps, num_frames, sample): 
        # don't load any segmentations
        return sample

    # def _augment_sequence_sample(self, sample):
    #     return sample

