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

from inferno.datasets.MEADDataModule import MEADDataModule, MEADDataset, robust_collate
from inferno.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp 
import imgaug
import numpy as np
import torch
from inferno.datasets.ConditionedVideoTestDatasetWrapper import ConditionedVideoTestDatasetWrapper
import omegaconf

class MEADPseudo3DDM(MEADDataModule): 

    def __init__(self, root_dir, output_dir, 
                processed_subfolder=None, 
                face_detector='mediapipe', 
                # landmarks_from='sr_res',
                landmarks_from=None,
                face_detector_threshold=0.9, 
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
                occlusion_settings_train=None,
                occlusion_settings_val=None,
                occlusion_settings_test=None,
                split = "random_70_15_15",
                num_workers=4,
                device=None,
                augmentation=None,
                drop_last=True,
                include_processed_audio = True,
                include_raw_audio = True,
                test_condition_source=None,
                test_condition_settings=None,
                inflate_by_video_size=False,

                landmark_types = None,
                landmark_sources = None,
                segmentation_source = None,
                segmentation_type=None,
                preload_videos=False,
                read_video=True,
                read_audio=True,
                reconstruction_type=None, 
                return_global_pose= False,
                return_appearance= False,
                average_shape_decode= True,
                emotion_type=None,
                return_emotion_feature=False,
                shuffle_validation=False,
                align_images=False,
                return_mica_images=False,
            ):
        super().__init__(root_dir, output_dir, processed_subfolder, face_detector, 
            landmarks_from, 
            face_detector_threshold, 
            image_size, scale, 
            processed_video_size=processed_video_size,
            batch_size_train=batch_size_train, 
            batch_size_val=batch_size_val, 
            batch_size_test=batch_size_test, 
            sequence_length_train=sequence_length_train, 
            sequence_length_val=sequence_length_val, 
            sequence_length_test=sequence_length_test, 
            occlusion_settings_train=occlusion_settings_train, 
            occlusion_settings_val=occlusion_settings_val, 
            occlusion_settings_test=occlusion_settings_test, 
            split=split, 
            num_workers=num_workers, 
            device=device, 
            augmentation=augmentation, 
            drop_last=drop_last, 
            include_processed_audio=include_processed_audio,
            include_raw_audio=include_raw_audio,
            inflate_by_video_size=inflate_by_video_size,
            preload_videos=preload_videos, 
            landmark_types=landmark_types,
            landmark_sources=landmark_sources,
            segmentation_source=segmentation_source,
            segmentation_type=segmentation_type,
            read_video=read_video,
            read_audio=read_audio,
            shuffle_validation=shuffle_validation,
            align_images=align_images,
            return_mica_images=return_mica_images,
            )
        self.test_condition_source = test_condition_source or "original"
        self.test_condition_settings = test_condition_settings

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

    def _get_smaller_renderable_subset_single_identity(self, indices, max_videos_per_category=1, 
                                                       accepted_expression=None, 
                                                       accepted_intensity=None):
        identity_expression_intensity2idx = {}
        identity = None
        for i in indices:
            identity_ = self._video_identity(i)
            if identity is None:
                identity = identity_
            if identity_ != identity:
                continue
            expression = self._video_expression(i)
            if accepted_expression is not None and expression not in accepted_expression:
                continue
            intensity = self._expression_intensity(i)
            if accepted_intensity is not None and intensity not in accepted_intensity:
                continue
            key = (identity, expression, intensity)
            if key not in identity_expression_intensity2idx:
                identity_expression_intensity2idx[key] = []
            if len(identity_expression_intensity2idx[key]) < max_videos_per_category:
                identity_expression_intensity2idx[key] += [i]


        id_expression_intensity2filename = {}
        for key, idxs in identity_expression_intensity2idx.items():
            id_expression_intensity2filename[key] = [self.video_list[i] for i in idxs]

        final_indices = []
        for key, idxs in identity_expression_intensity2idx.items():
            final_indices += [i for i in idxs]

        return final_indices


    def _get_validation_set(self, indices, sequence_length = None ): 
        sequence_length = sequence_length or self.sequence_length_val
        validation_set = MEADPseudo3dDataset(self.root_dir, self.output_dir, 
                self.video_list, self.video_metas, 
                indices, 
                self.audio_metas, 
                sequence_length, 
                image_size=self.image_size,  
                **self.occlusion_settings_val,
                hack_length=False, 
                # use_original_video=self.use_original_video,
                include_processed_audio = self.include_processed_audio,
                include_raw_audio = self.include_raw_audio,
                landmark_types=self.landmark_types,
                landmark_source=self.landmark_sources,
                # segmentation_source=self.segmentation_source,
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
                return_mica_images=self.return_mica_images,
                original_image_size=self.processed_video_size,
            )
        validation_set._set_identity_label(self.training_set.identity_labels, self.training_set.identity_label2index)
        return validation_set
        

    def setup(self, stage=None):
        train, val, test = self._get_subsets(self.split)

        # training_augmenter = create_image_augmenter(self.image_size, self.augmentation)
        training_augmenter = None
        self.training_set = MEADPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                train, 
                self.audio_metas, self.sequence_length_train, image_size=self.image_size, 
                transforms=training_augmenter,
                **self.occlusion_settings_train,
            
                hack_length=False,
                # use_original_video=self.use_original_video,
                include_processed_audio = self.include_processed_audio,
                include_raw_audio = self.include_raw_audio,
                landmark_types=self.landmark_types,
                landmark_source=self.landmark_sources,
                # segmentation_source=self.segmentation_source,
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
                return_mica_images=self.return_mica_images,
                original_image_size=self.processed_video_size,
              )           
        self.validation_set = self._get_validation_set(val)

        val_test_set = self._get_smaller_renderable_subset_single_identity(val, max_videos_per_category=1)
        train_test_set = self._get_smaller_renderable_subset_single_identity(train, max_videos_per_category=1)
        train_test_cond_set = self._get_smaller_renderable_subset_single_identity(train, max_videos_per_category=1, accepted_expression='neutral')
        val_test_cond_set_neutral = self._get_smaller_renderable_subset_single_identity(val, max_videos_per_category=1, accepted_expression='neutral')
        # val_test_cond_set_happy_1 = self._get_smaller_renderable_subset_single_identity(
        #     val, max_videos_per_category=1, accepted_expression='happy', accepted_intensity='level_1')
        # val_test_cond_set_happy_2 = self._get_smaller_renderable_subset_single_identity(
        #     val, max_videos_per_category=1, accepted_expression='happy', accepted_intensity='level_2')
        val_test_cond_set_happy_3 = self._get_smaller_renderable_subset_single_identity(
            val, max_videos_per_category=1, accepted_expression='happy', accepted_intensity='level_3')
        val_test_cond_set_sad_1 = self._get_smaller_renderable_subset_single_identity(
            val, max_videos_per_category=1, accepted_expression='sad', accepted_intensity='level_1')
        val_test_cond_set_sad_2 = self._get_smaller_renderable_subset_single_identity(
            val, max_videos_per_category=1, accepted_expression='sad', accepted_intensity='level_2')
        val_test_cond_set_sad_3 = self._get_smaller_renderable_subset_single_identity(
            val, max_videos_per_category=1, accepted_expression='sad', accepted_intensity='level_3')

        self.test_set_names = []
        # if len(test) > 0:
        #     self.test_set_ = MEADPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, test, self.audio_metas, 
        #             # sequence_length=self.sequence_length_test, 
        #             sequence_length="all", 
        #             image_size=self.image_size, 
        #             **self.occlusion_settings_test,
        #             hack_length=False, 
        #             # use_original_video=self.use_original_video,
        #             include_processed_audio = self.include_processed_audio,
        #             include_raw_audio = self.include_raw_audio,
        #             landmark_types=self.landmark_types,
        #             landmark_source=self.landmark_sources,
        #             # segmentation_source=self.segmentation_source,
        #             temporal_split_start=self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
        #             temporal_split_end= sum(self.temporal_split) if self.temporal_split is not None else None,
        #             # preload_videos=self.preload_videos,
        #             # inflate_by_video_size=self.inflate_by_video_size,
        #             inflate_by_video_size=False,
        #             include_filename=True,
        #             read_video=self.read_video,
        #             reconstruction_type=self.reconstruction_type,
        #             return_global_pose=self.return_global_pose,
        #             return_appearance=self.return_appearance,
        #             average_shape_decode=self.average_shape_decode,
        #             emotion_type=self.emotion_type,
        #             return_emotion_feature=self.return_emotion_feature,
        #             original_image_size=self.processed_video_size,
        #             )

        #     self.test_set = ConditionedVideoTestDatasetWrapper(
        #         self.test_set_,
        #         None, 
        #         None,
        #         key_prefix="gt_",
        #     )

        max_training_test_samples = 2
        self.test_set_train_ = MEADPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                # sorted(train)[:max_training_test_samples], 
                sorted(train_test_set),
                self.audio_metas, 
                # sequence_length=self.sequence_length_test, 
                sequence_length="all", 
                image_size=self.image_size, 
                **self.occlusion_settings_test,
                hack_length=False, 
                # use_original_video=self.use_original_video,
                include_processed_audio = self.include_processed_audio,
                include_raw_audio = self.include_raw_audio,
                landmark_types=self.landmark_types,
                landmark_source=self.landmark_sources,
                # segmentation_source=self.segmentation_source,

                temporal_split_start= 0 if self.temporal_split is not None else None,
                temporal_split_end=self.temporal_split[0] if self.temporal_split is not None else None,
                # preload_videos=self.preload_videos,
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
                return_mica_images=self.return_mica_images,
                original_image_size=self.processed_video_size,
                )

        self.test_set_train_._set_identity_label(self.training_set.identity_labels, self.training_set.identity_label2index)

        self.test_set_train = ConditionedVideoTestDatasetWrapper(
            self.test_set_train_,
            None, 
            None,
            key_prefix="gt_",
        )

        max_validation_test_samples = 2
        if "specific_identity" in self.split: 
            max_validation_test_samples = len(val)

        self.test_set_val_ = MEADPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                # sorted(val)[:max_validation_test_samples], 
                val_test_set, 
                self.audio_metas, 
                # sequence_length=self.sequence_length_test, 
                sequence_length="all", 
                image_size=self.image_size, 
                **self.occlusion_settings_test,
                hack_length=False, 
                # use_original_video=self.use_original_video,
                include_processed_audio = self.include_processed_audio,
                include_raw_audio = self.include_raw_audio,
                landmark_types=self.landmark_types,
                landmark_source=self.landmark_sources,
                # segmentation_source=self.segmentation_source,
                temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                # preload_videos=self.preload_videos,
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
                return_mica_images=self.return_mica_images,
                original_image_size=self.processed_video_size,
                )
        
        self.test_set_val_._set_identity_label(self.training_set.identity_labels, self.training_set.identity_label2index)

        self.test_set_val = ConditionedVideoTestDatasetWrapper(
            self.test_set_val_,
            None, 
            None,
            key_prefix="gt_",
        )

        # conditioned test set
        if self.test_condition_source != "original":
            # if len(test) > 0:
            #     self.test_set_cond_ = MEADPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
            #             test, 
            #             self.audio_metas, 
            #             # sequence_length=self.sequence_length_test, 
            #             sequence_length="all", 
            #             image_size=self.image_size, 
            #             **self.occlusion_settings_test,
            #             hack_length=False, 
            #             # use_original_video=self.use_original_video,
            #             include_processed_audio = self.include_processed_audio,
            #             include_raw_audio = self.include_raw_audio,
            #             landmark_types=self.landmark_types,
            #             landmark_source=self.landmark_sources,
            #             # segmentation_source=self.segmentation_source,
            #             temporal_split_start=self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
            #             temporal_split_end= sum(self.temporal_split) if self.temporal_split is not None else None,
            #             # preload_videos=self.preload_videos,
            #             # inflate_by_video_size=self.inflate_by_video_size,
            #             inflate_by_video_size=False,
            #             include_filename=True,
            #             read_video=self.read_video,
            #             reconstruction_type=self.reconstruction_type,
            #             return_global_pose=self.return_global_pose,
            #             return_appearance=self.return_appearance,
            #             average_shape_decode=self.average_shape_decode,
            #             emotion_type=self.emotion_type,
            #             return_emotion_feature=self.return_emotion_feature,
            #             original_image_size=self.processed_video_size,
            #             )

            #     self.test_set_cond = ConditionedVideoTestDatasetWrapper(
            #         self.test_set_cond_,
            #         self.test_condition_source, 
            #         self.test_condition_settings, 
            #         key_prefix="gt_",
            #     )

            max_training_test_samples = 2
            self.test_set_train_cond_ = MEADPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                    # sorted(train)[:max_training_test_samples], 
                    sorted(train_test_cond_set),
                    self.audio_metas, 
                    # sequence_length=self.sequence_length_test, 
                    sequence_length="all", 
                    image_size=self.image_size, 
                    **self.occlusion_settings_test,
                    hack_length=False, 
                    # use_original_video=self.use_original_video,
                    include_processed_audio = self.include_processed_audio,
                    include_raw_audio = self.include_raw_audio,
                    landmark_types=self.landmark_types,
                    landmark_source=self.landmark_sources,
                    # segmentation_source=self.segmentation_source,

                    temporal_split_start= 0 if self.temporal_split is not None else None,
                    temporal_split_end=self.temporal_split[0] if self.temporal_split is not None else None,
                    # preload_videos=self.preload_videos,
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
                    return_mica_images=self.return_mica_images,
                    original_image_size=self.processed_video_size,
                    )
            
            self.test_set_train_cond_._set_identity_label(self.training_set.identity_labels, self.training_set.identity_label2index)


            self.test_set_train_cond = ConditionedVideoTestDatasetWrapper(
                self.test_set_train_cond_,
                self.test_condition_source, 
                self.test_condition_settings, 
                key_prefix="gt_",
            )

            max_validation_test_samples = 2
            self.test_set_val_cond_neutral_ = MEADPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                    # sorted(val)[:max_validation_test_samples], 
                    sorted(val_test_cond_set_neutral), 
                    self.audio_metas, 
                    # sequence_length=self.sequence_length_test, 
                    sequence_length="all", 
                    image_size=self.image_size, 
                    **self.occlusion_settings_val,
                    hack_length=False, 
                    # use_original_video=self.use_original_video,
                    include_processed_audio = self.include_processed_audio,
                    include_raw_audio = self.include_raw_audio,
                    landmark_types=self.landmark_types,
                    landmark_source=self.landmark_sources,
                    # segmentation_source=self.segmentation_source,
                    temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                    temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                    # preload_videos=self.preload_videos,
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
                    return_mica_images=self.return_mica_images,
                    original_image_size=self.processed_video_size,
                    )

            self.test_set_val_cond_neutral_._set_identity_label(self.training_set.identity_labels, self.training_set.identity_label2index)

            self.test_set_val_cond_neutral_3 = ConditionedVideoTestDatasetWrapper(
                self.test_set_val_cond_neutral_,
                self.test_condition_source, 
                self.test_condition_settings, 
                key_prefix="gt_",
                conditioned_intensity=3
            )

            self.test_set_val_cond_neutral_2 = ConditionedVideoTestDatasetWrapper(
                self.test_set_val_cond_neutral_,
                self.test_condition_source, 
                self.test_condition_settings, 
                key_prefix="gt_",
                conditioned_intensity=2,
            )

            self.test_set_val_cond_neutral_1 = ConditionedVideoTestDatasetWrapper(
                self.test_set_val_cond_neutral_,
                self.test_condition_source, 
                self.test_condition_settings, 
                key_prefix="gt_",
                conditioned_intensity=1,
            )

            self.test_set_val_cond_happy_3_ = MEADPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                    # sorted(val)[:max_validation_test_samples], 
                    sorted(val_test_cond_set_happy_3), 
                    self.audio_metas, 
                    # sequence_length=self.sequence_length_test, 
                    sequence_length="all", 
                    image_size=self.image_size, 
                    **self.occlusion_settings_val,
                    hack_length=False, 
                    # use_original_video=self.use_original_video,
                    include_processed_audio = self.include_processed_audio,
                    include_raw_audio = self.include_raw_audio,
                    landmark_types=self.landmark_types,
                    landmark_source=self.landmark_sources,
                    # segmentation_source=self.segmentation_source,
                    temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                    temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                    # preload_videos=self.preload_videos,
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
                    return_mica_images=self.return_mica_images,
                    original_image_size=self.processed_video_size,
                    )

            self.test_set_val_cond_happy_3_._set_identity_label(self.training_set.identity_labels, self.training_set.identity_label2index)
            self.test_set_val_cond_happy_3 = ConditionedVideoTestDatasetWrapper(
                self.test_set_val_cond_happy_3_,
                self.test_condition_source, 
                self.test_condition_settings, 
                key_prefix="gt_",
            )
            
            self.test_set_val_cond_sad_1_ = MEADPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas,
                    # sorted(val)[:max_validation_test_samples],
                    sorted(val_test_cond_set_sad_1),
                    self.audio_metas,
                    # sequence_length=self.sequence_length_test,
                    sequence_length="all",
                    image_size=self.image_size,
                    **self.occlusion_settings_val,
                    hack_length=False,
                    # use_original_video=self.use_original_video,
                    include_processed_audio=self.include_processed_audio,
                    include_raw_audio=self.include_raw_audio,
                    landmark_types=self.landmark_types,
                    landmark_source=self.landmark_sources,
                    # segmentation_source=self.segmentation_source,
                    temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                    temporal_split_end=self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                    # preload_videos=self.preload_videos,
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
                    return_mica_images=self.return_mica_images,
                    original_image_size=self.processed_video_size,
                    )
            self.test_set_val_cond_sad_1_._set_identity_label(self.training_set.identity_labels, self.training_set.identity_label2index)
            self.test_set_val_cond_sad_1 = ConditionedVideoTestDatasetWrapper(
                self.test_set_val_cond_sad_1_,
                self.test_condition_source,
                self.test_condition_settings,
                key_prefix="gt_",
                # conditioned_intensity=1,
            )


            self.test_set_val_cond_sad_2_ = MEADPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas,
                    # sorted(val)[:max_validation_test_samples],
                    sorted(val_test_cond_set_sad_2),
                    self.audio_metas,
                    # sequence_length=self.sequence_length_test,
                    sequence_length="all",
                    image_size=self.image_size,
                    **self.occlusion_settings_val,
                    hack_length=False,
                    # use_original_video=self.use_original_video,
                    include_processed_audio=self.include_processed_audio,
                    include_raw_audio=self.include_raw_audio,
                    landmark_types=self.landmark_types,
                    landmark_source=self.landmark_sources,
                    # segmentation_source=self.segmentation_source,
                    temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                    temporal_split_end=self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                    # preload_videos=self.preload_videos,
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
                    return_mica_images=self.return_mica_images,
                    original_image_size=self.processed_video_size,
                    )
            self.test_set_val_cond_sad_2_._set_identity_label(self.training_set.identity_labels, self.training_set.identity_label2index)
            self.test_set_val_cond_sad_2 = ConditionedVideoTestDatasetWrapper(
                self.test_set_val_cond_sad_2_,
                self.test_condition_source,
                self.test_condition_settings,
                key_prefix="gt_",
                # conditioned_intensity=2,
            )

            self.test_set_val_cond_sad_3_ = MEADPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, 
                    # sorted(val)[:max_validation_test_samples], 
                    sorted(val_test_cond_set_sad_3), 
                    self.audio_metas, 
                    # sequence_length=self.sequence_length_test, 
                    sequence_length="all", 
                    image_size=self.image_size, 
                    **self.occlusion_settings_val,
                    hack_length=False, 
                    # use_original_video=self.use_original_video,
                    include_processed_audio = self.include_processed_audio,
                    include_raw_audio = self.include_raw_audio,
                    landmark_types=self.landmark_types,
                    landmark_source=self.landmark_sources,
                    # segmentation_source=self.segmentation_source,
                    temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                    temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                    # preload_videos=self.preload_videos,
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
                    return_mica_images=self.return_mica_images,
                    original_image_size=self.processed_video_size,
                    )

            self.test_set_val_cond_sad_3_._set_identity_label(self.training_set.identity_labels, self.training_set.identity_label2index)
            self.test_set_val_cond_sad_3 = ConditionedVideoTestDatasetWrapper(
                self.test_set_val_cond_sad_3_,
                self.test_condition_source, 
                self.test_condition_settings, 
                key_prefix="gt_",
            )




    def test_dataloader(self):
        test_dls = []
        if hasattr(self, "test_set"):
            # test_dl = super().test_dataloader()
            # if test_dl is not None:
            #     if not isinstance(test_dl, list): 
            #         test_dl = [test_dl]
            #     test_dls += test_dl
                test_dls += [torch.utils.data.DataLoader(self.test_set, shuffle=False, 
                            num_workers=self.num_workers, 
                        #   num_workers=0, 
                          pin_memory=True,
                        #   batch_size=self.batch_size_test, 
                          batch_size=1, 
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )]
                self.test_set_names += ["test"]

        test_dls += [torch.utils.data.DataLoader(self.test_set_train, shuffle=False, 
                            num_workers=self.num_workers, 
                        #   num_workers=0, 
                          pin_memory=True,
                        #   batch_size=self.batch_size_test, 
                          batch_size=1, 
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )]
        self.test_set_names += ["train"]

        test_dls += [torch.utils.data.DataLoader(self.test_set_val, shuffle=False, 
                            num_workers=self.num_workers, 
                        #   num_workers=0, 
                          pin_memory=True,
                        #   batch_size=self.batch_size_test, 
                          batch_size=1,
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )]
        self.test_set_names += ["val"]

        if hasattr(self, "test_set_cond") and self.test_set_cond is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_cond, shuffle=False, 
                        #   num_workers=self.num_workers, 
                          num_workers=0, 
                        pin_memory=True,
                        #   batch_size=self.batch_size_test, 
                          batch_size=1,
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )]
            self.test_set_names += ["test"]

        if hasattr(self, "test_set_train_cond") and self.test_set_train_cond is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_train_cond, shuffle=False, 
                              num_workers=self.num_workers, 
                        #   num_workers=0, 
                            pin_memory=True,
                        #   batch_size=self.batch_size_test, 
                          batch_size=1,
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )]
            self.test_set_names += ["train_cond"]

        if hasattr(self, "test_set_val_cond_neutral_3") and self.test_set_val_cond_neutral_3 is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_val_cond_neutral_3, shuffle=False, 
                        #   num_workers=self.num_workers, 
                          num_workers=0, 
                        #   num_workers=1, 
                          pin_memory=True,
                        #   batch_size=self.batch_size_test, 
                          batch_size=1,
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )]
            self.test_set_names += ["val_cond"]

        if hasattr(self, "test_set_val_cond_neutral_2") and self.test_set_val_cond_neutral_2 is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_val_cond_neutral_2, shuffle=False, 
                        #   num_workers=self.num_workers, 
                        #   num_workers=0, 
                          num_workers=1, 
                          pin_memory=True,
                        #   batch_size=self.batch_size_test, 
                          batch_size=1,
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )]
            self.test_set_names += ["val_cond"]

        if hasattr(self, "test_set_val_cond_neutral_1") and self.test_set_val_cond_neutral_1 is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_val_cond_neutral_1, shuffle=False, 
                        #   num_workers=self.num_workers, 
                        #   num_workers=0, 
                          num_workers=1, 
                          pin_memory=True,
                        #   batch_size=self.batch_size_test, 
                          batch_size=1,
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )]
            self.test_set_names += ["val_cond"]

        if hasattr(self, "test_set_val_cond_happy_3") and self.test_set_val_cond_happy_3 is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_val_cond_happy_3, shuffle=False,
                        #   num_workers=self.num_workers,
                            # num_workers=0,
                            num_workers=1,
                            pin_memory=True,
                        #   batch_size=self.batch_size_test,
                            batch_size=1,
                            drop_last=False,
                        #   drop_last=self.drop_last,
                            collate_fn=robust_collate    
                            )]
            self.test_set_names += ["val_cond"]
            
        if hasattr(self, "test_set_val_cond_sad_1") and self.test_set_val_cond_sad_1 is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_val_cond_sad_1, shuffle=False,
                        #   num_workers=self.num_workers,
                            # num_workers=0,
                            num_workers=1,
                            pin_memory=True,
                        #   batch_size=self.batch_size_test,
                            batch_size=1,
                            drop_last=False,
                        #   drop_last=self.drop_last,
                            collate_fn=robust_collate    
                            )]
            self.test_set_names += ["val_cond"]
            
        if hasattr(self, "test_set_val_cond_sad_2") and self.test_set_val_cond_sad_2 is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_val_cond_sad_2, shuffle=False,
                        #   num_workers=self.num_workers,
                            # num_workers=0,
                            num_workers=1,
                            pin_memory=True,
                        #   batch_size=self.batch_size_test,
                            batch_size=1,
                            drop_last=False,
                        #   drop_last=self.drop_last,
                            collate_fn=robust_collate    
                            )]

        if hasattr(self, "test_set_val_cond_sad_3") and self.test_set_val_cond_sad_3 is not None:
            test_dls += [torch.utils.data.DataLoader(self.test_set_val_cond_sad_3, shuffle=False,
                        #   num_workers=self.num_workers,
                            # num_workers=0,
                            num_workers=1,
                            pin_memory=True,
                        #   batch_size=self.batch_size_test,
                            batch_size=1,
                            drop_last=False,
                        #   drop_last=self.drop_last,
                            collate_fn=robust_collate    
                            )]
            self.test_set_names += ["val_cond"]

        return test_dls




class MEADPseudo3dDataset(MEADDataset):

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
            landmark_types="mediapipe", 
            segmentation_type = "bisenet",
            landmark_source = "original",
            segmentation_source = "original",
            temporal_split_start=None,
            temporal_split_end=None,
            occlusion_length=0,
            occlusion_probability_mouth = 0.0,
            occlusion_probability_left_eye = 0.0,
            occlusion_probability_right_eye = 0.0,
            occlusion_probability_face = 0.0,
            image_size=None, 
            transforms : imgaug.augmenters.Augmenter = None,
            hack_length=False,
            preload_videos=False, # cache all videos in memory (recommended for smaller datasets)
            inflate_by_video_size=False, 
            include_filename=False, # if True includes the filename of the video in the sample
            read_video=True,
            read_audio=True,
            reconstruction_type=None,
            return_global_pose = False,
            return_appearance = False,
            average_shape_decode = True,
            include_processed_audio=False,
            include_raw_audio=True,
            emotion_type=None,
            return_emotion_feature=False,
            return_mica_images=False,
            original_image_size = None,
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
            preload_videos=preload_videos, 
            inflate_by_video_size=inflate_by_video_size, 
            include_filename=include_filename,
            temporal_split_start=temporal_split_start,
            temporal_split_end=temporal_split_end, 
            include_processed_audio=include_processed_audio,
            include_raw_audio=include_raw_audio,
            read_video=read_video,
            read_audio=read_audio,
            reconstruction_type=reconstruction_type,
            return_global_pose = return_global_pose,
            return_appearance = return_appearance,
            average_shape_decode = average_shape_decode,
            emotion_type = emotion_type,
            return_emotion_feature = return_emotion_feature,
            return_mica_images = return_mica_images,
            original_image_size = original_image_size,
            )
        # self.read_video = read_video

        # self.reconstruction_type = reconstruction_type
        # if self.reconstruction_type is not None:
        #     self.return_global_pose = return_global_pose
        #     self.return_appearance = return_appearance
        #     self.average_shape_decode = average_shape_decode
            # self._load_flame()

        # self.emotion_type = emotion_type
        # self.return_emotion_feature = return_emotion_feature
            


    # def _get_landmarks(self, index, start_frame, num_read_frames, video_fps, num_frames, sample): 
    #     # don't load any landmarks+
    #     return sample

    def _get_landmarks(self, index, start_frame, num_read_frames, video_fps, num_frames, sample): 
    #     # don't load any landmarks+
        sample = super()._get_landmarks(index, start_frame, num_read_frames, video_fps, num_frames, sample)
        lmk_weights = sample["landmarks_validity"]["mediapipe"] / sample["landmarks_validity"]["mediapipe"].sum(axis=0, keepdims=True)
        # print(lmk_weights.mean())
        # if np.any(np.isnan(lmk_weights)): 
        #     import matplotlib.pyplot as plt
        #     image = np.concatenate( sample["video"].tolist(), axis=1)
        #     plt.figure()
        #     plt.imshow(image)
        #     plt.show()

        assert np.any(np.isnan(lmk_weights)) == False, "NaN in weights" # throw an error if there are NaNs, this should cause a new sample to be loaded
        return sample

    def _get_segmentations(self, index, start_frame, num_read_frames, video_fps, num_frames, sample): 
        # don't load any segmentations
        return sample

    # def _augment_sequence_sample(self, sample):
    #     return sample


