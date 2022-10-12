from gdl.datasets.CelebVHQDataModule import CelebVHQDataModule, CelebVHQDataset
from gdl.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp 

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
            segmentation_source=None):
        super().__init__(root_dir, output_dir, processed_subfolder, face_detector, landmarks_from, face_detector_threshold, image_size, scale, processed_video_size, batch_size_train, batch_size_val, batch_size_test, 
            sequence_length_train, sequence_length_val, sequence_length_test, 
            bb_center_shift_x, bb_center_shift_y, 
            occlusion_settings_train, occlusion_settings_val, occlusion_settings_test, 
            split, 
            num_workers, device, augmentation, drop_last, include_processed_audio, include_raw_audio, preload_videos, inflate_by_video_size, training_sampler, landmark_types, landmark_sources, segmentation_source)


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
            )

        self.test_set = CelebVHQPseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, test, self.audio_metas, 
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
                )


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
            inflate_by_video_size=False
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
            inflate_by_video_size)

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

     


def main(): 
    import time
    from pathlib import Path
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

    augmenter = None

        # Create the dataset
    dm = CelebVHQPseudo3DDM(
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
        occlusion_settings_train=None,
    )

    # Create the dataloader
    dm.prepare_data() 
    dm.setup() 

    # dl = dm.train_dataloader()
    # # dl = dm.val_dataloader()
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



if __name__ == "__main__": 
    main()