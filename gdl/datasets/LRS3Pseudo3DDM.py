from gdl.datasets.LRS3DataModule import LRS3DataModule, LRS3Dataset
from gdl.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp 
import imgaug
import numpy as np
import torch


class LRS3Pseudo3DDM(LRS3DataModule): 

    def __init__(self, root_dir, output_dir, 
                processed_subfolder=None, face_detector='mediapipe', 
                # landmarks_from='sr_res',
                landmarks_from=None,
                face_detector_threshold=0.9, 
                image_size=224, scale=1.25, 
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
                split = "original",
                num_workers=4,
                device=None,
                augmentation=None,
                drop_last=True,
            ):
        super().__init__(root_dir, output_dir, processed_subfolder, face_detector, 
            landmarks_from, 
            face_detector_threshold, 
            image_size, scale, batch_size_train, batch_size_val, batch_size_test, 
            sequence_length_train, sequence_length_val, sequence_length_test, 
            occlusion_settings_train, occlusion_settings_val, occlusion_settings_test, 
            split, 
            num_workers, device, augmentation, drop_last
            )


    def setup(self, stage=None):
        train, val, test = self._get_subsets(self.split)
        # training_augmenter = create_image_augmenter(self.image_size, self.augmentation)
        training_augmenter = None
        self.training_set = LRS3Pseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, train, 
                self.audio_metas, self.sequence_length_train, image_size=self.image_size, 
                transforms=training_augmenter,
                **self.occlusion_settings_train,
            
                hack_length=False,
                # use_original_video=self.use_original_video,
                # include_processed_audio = self.include_processed_audio,
                # include_raw_audio = self.include_raw_audio,
                # landmark_types=self.landmark_types,
                # landmark_source=self.landmark_sources,
                # segmentation_source=self.segmentation_source,
                # temporal_split_start= 0 if self.temporal_split is not None else None,
                # temporal_split_end=self.temporal_split[0] if self.temporal_split is not None else None,
                # preload_videos=self.preload_videos,
                # inflate_by_video_size=self.inflate_by_video_size,
              )
                    
        self.validation_set = LRS3Pseudo3dDataset(self.root_dir, self.output_dir, 
                self.video_list, self.video_metas, val, self.audio_metas, 
                self.sequence_length_val, image_size=self.image_size,  
                **self.occlusion_settings_val,
                hack_length=False, 
                # use_original_video=self.use_original_video,
                # include_processed_audio = self.include_processed_audio,
                # include_raw_audio = self.include_raw_audio,
                # landmark_types=self.landmark_types,
                # landmark_source=self.landmark_sources,
                # segmentation_source=self.segmentation_source,
                # temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                # temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                # preload_videos=self.preload_videos,
                # inflate_by_video_size=self.inflate_by_video_size,
            )

        self.test_set = LRS3Pseudo3dDataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, test, self.audio_metas, 
                self.sequence_length_test, image_size=self.image_size, 
                **self.occlusion_settings_test,
                hack_length=False, 
                # use_original_video=self.use_original_video,
                # include_processed_audio = self.include_processed_audio,
                # include_raw_audio = self.include_raw_audio,
                # landmark_types=self.landmark_types,
                # landmark_source=self.landmark_sources,
                # segmentation_source=self.segmentation_source,
                # temporal_split_start=self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                # temporal_split_end= sum(self.temporal_split) if self.temporal_split is not None else None,
                # preload_videos=self.preload_videos,
                # inflate_by_video_size=self.inflate_by_video_size,
                )


class LRS3Pseudo3dDataset(LRS3Dataset):

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
            occlusion_length=0,
            occlusion_probability_mouth = 0.0,
            occlusion_probability_left_eye = 0.0,
            occlusion_probability_right_eye = 0.0,
            occlusion_probability_face = 0.0,
            image_size=None, 
            transforms : imgaug.augmenters.Augmenter = None,
            hack_length=False,
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
            hack_length)

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

     


def main(): 
    import time
    from pathlib import Path

    root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs3/extracted")
    output_dir = Path("/is/cluster/work/rdanecek/data/lrs3/")

    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs2/mvlrs_v1")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs2")

    # processed_subfolder = "processed"
    processed_subfolder = "processed2"

    seq_len = 50
    # seq_len = 16
    # bs = 100
    bs = 1

    augmenter = None

        # Create the dataset
    dm = LRS3Pseudo3DDM(
        root_dir, output_dir, processed_subfolder,
        split="original",
        image_size=224, 
        scale=1.25, 
        # processed_video_size=256,
        batch_size_train=bs,
        batch_size_val=bs,
        batch_size_test=bs,
        sequence_length_train=seq_len,
        sequence_length_val=seq_len,
        sequence_length_test=seq_len,
        num_workers=8,            
        # include_processed_audio = True,
        # include_raw_audio = True,
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
        # dataset.visualize_sample(sample)



if __name__ == "__main__": 
    main()
