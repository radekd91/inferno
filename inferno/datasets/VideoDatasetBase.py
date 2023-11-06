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

import torch
import numpy as np 
import imgaug
from inferno.transforms.keypoints import KeypointNormalization, KeypointScale
from inferno.utils.MediaPipeFaceOccluder import MediaPipeFaceOccluder, sizes_to_bb_batch
from pathlib import Path
from scipy.io import wavfile 
from python_speech_features import logfbank
from inferno.datasets.FaceDataModuleBase import FaceDataModuleBase
from inferno.datasets.IO import (load_and_process_segmentation, process_segmentation, 
                             load_segmentation, load_segmentation_list, load_segmentation_list_v2,
                             load_reconstruction_list, load_emotion_list, 
                             load_reconstruction_list_v2, load_emotion_list_v2,
                             )
from inferno.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp
from inferno.utils.FaceDetector import load_landmark
import pandas as pd
from skvideo.io import vread, vreader, FFmpegReader
import torch.nn.functional as F
import subprocess 
import traceback
from inferno.layers.losses.MediaPipeLandmarkLosses import MEDIAPIPE_LANDMARK_NUMBER
import cv2
from decord import VideoReader, cpu

class AbstractVideoDataset(torch.utils.data.Dataset):

    def __init__(self) -> None:
        super().__init__()

    def _augment_sequence_sample(self, index, sample):
        raise NotImplementedError()

    def visualize_sample(self, sample):
        raise NotImplementedError()


class VideoDatasetBase(AbstractVideoDataset): 

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
            image_size=None, ## the image size that the dataset will output
            transforms : imgaug.augmenters.Augmenter = None,
            hack_length=False,
            use_original_video=True,
            include_processed_audio = True,
            include_raw_audio = True,
            temporal_split_start=None, # if temporally splitting the video (train, val, test), this is the start of the split
            temporal_split_end=None, # if temporally splitting the video (train, val, test), this is the end of the split
            preload_videos=False, # cache all videos in memory (recommended for smaller datasets)
            inflate_by_video_size=False, 
            include_filename=False, # if True includes the filename of the video in the sample
            align_images = True,
            use_audio=True, #if True, includes audio in the sample
            reconstruction_type = None,
            return_global_pose = False,
            return_appearance = False,
            average_shape_decode = False,
            emotion_type = None,
            return_emotion_feature=False,
            read_video=True,
            read_audio=True,
            original_image_size=None, ## the processed videos may be different in size and if they are, the landmarks will be, too. This is to remember
            return_mica_images=False,
        ) -> None:
        super().__init__()
        self.root_path = root_path
        self.output_dir = output_dir
        self.video_list = video_list
        self.video_indices = video_indices
        self.video_metas = video_metas
        self.sequence_length = sequence_length or 1
        # self.audio_paths = audio_paths
        self.audio_metas = audio_metas
        self.audio_noise_prob = audio_noise_prob
        self.image_size = image_size
        self.original_image_size = original_image_size or image_size
        self.scale = 1.25
        # if the video is 25 fps and the audio is 16 kHz, stack_order_audio corresponds to 4 
        # (i.e. 4 consecutive filterbanks will be concatenated to sync with the visual frame) 
        self.stack_order_audio = stack_order_audio

        self.audio_normalization = audio_normalization

        self.landmark_types = landmark_types 
        if isinstance(self.landmark_types, str): 
            self.landmark_types = [self.landmark_types]
        self.landmark_source = landmark_source 
        if isinstance(self.landmark_source, str): 
            self.landmark_source = [self.landmark_source] * len(self.landmark_types)
        assert len(self.landmark_types) == len(self.landmark_source), "landmark_types and landmark_source must have the same length"
        self.segmentation_type = segmentation_type 
        
        self.segmentation_source = segmentation_source


        self.landmark_normalizer = KeypointNormalization() # postprocesses final landmarks to be in [-1, 1]
        self.occluder = MediaPipeFaceOccluder()
        self.occlusion_probability_mouth = occlusion_probability_mouth
        self.occlusion_probability_left_eye = occlusion_probability_left_eye
        self.occlusion_probability_right_eye = occlusion_probability_right_eye
        self.occlusion_probability_face = occlusion_probability_face
        self.occlusion_length = occlusion_length
        if isinstance(self.occlusion_length, int):
            self.occlusion_length = [self.occlusion_length, self.occlusion_length+1]
        # self.occlusion_length = [20, 30]
        self.occlusion_length = sorted(self.occlusion_length)

        self.include_processed_audio = include_processed_audio
        self.include_raw_audio = include_raw_audio
        # self.align_images = True
        # self.align_images = False
        self.align_images = align_images
        self.use_audio = use_audio
        self.use_original_video = use_original_video
        self.transforms = transforms or imgaug.augmenters.Resize((image_size, image_size))

        self.hack_length = hack_length
        if self.hack_length == "auto": 
            if self._true_len() < 64: # hacks the length for supersmall test datasets
                self.hack_length = (64 // self._true_len())
                if 64 % self._true_len() != 0:
                    self.hack_length += 1
                self.hack_length = float(self.hack_length)
            # useful hack to repeat the elements in the dataset for really small datasets
            else: 
                self.hack_length = False

        assert self.occlusion_length[0] >= 0
        # assert self.occlusion_length[1] <= self.sequence_length + 1

        self.temporal_split_start = temporal_split_start
        self.temporal_split_end = temporal_split_end

        self.preload_videos = preload_videos
        self.inflate_by_video_size = inflate_by_video_size

        # self.read_video = True
        self.read_video = read_video
        self.read_audio = read_audio

        self.reconstruction_type = reconstruction_type 
        if self.reconstruction_type is not None: 
            if isinstance(self.reconstruction_type, str): 
                self.reconstruction_type = [self.reconstruction_type]
            assert isinstance(self.reconstruction_type, list), "reconstruction_type must be a list or None"
        self.return_global_pose = return_global_pose
        self.return_appearance = return_appearance
        self.average_shape_decode = average_shape_decode

        self.emotion_type = emotion_type
        self.return_emotion_feature = return_emotion_feature

        self.video_cache = {}
        self.audio_cache = {}
        self.seg_cache = {}
        self.lmk_cache = {}
        self.rec_cache = {}
        self.emo_cache = {}
        if self.preload_videos:
            self._preload_videos()
            
        self.video_sample_indices = None
        if self.inflate_by_video_size: 
            self._inflate_by_video_size()

        self.include_filename = include_filename

        # if True, face alignment will not crash if invalid. By default this should be False to avoid silent data errors
        self._allow_alignment_fail = False 

        self.return_mica_image = return_mica_images
        if not self.read_video:
            assert not bool(self.return_mica_image), "return_mica_image is only supported when read_video is True"

        if bool(self.return_mica_image):
            from inferno.models.mica.MicaInputProcessing import MicaInputProcessor
            if self.return_mica_image is True: 
                self.return_mica_image = "fan"
            self.mica_preprocessor = MicaInputProcessor(self.return_mica_image)


    @property
    def invalid_cutoff(self):
        max_cutoff = 0 
        for rec_type in self.reconstruction_type:
            if rec_type == "spectre": 
                max_cutoff = max(max_cutoff, 2 )
            elif rec_type in ["emoca", "deca"] or "emoca" in rec_type.lower() or "emica" in rec_type.lower() or "edeca" in rec_type.lower(): 
                max_cutoff = max(max_cutoff, 0 )
            else:
                raise ValueError(f"Invalid reconstruction type: '{rec_type}'")
        return max_cutoff


    def _load_flame(self):
        if self.reconstruction_type is not None: 
            from munch import Munch
            flame_cfg = Munch() 

            flame_cfg.type = "flame"

            flame_cfg.flame = Munch({ 
            "flame_model_path": "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl" ,
            "n_shape": 100 ,
            # n_exp: 100
            "n_exp": 50,
            "flame_lmk_embedding_path": "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy" 
            })
            flame_cfg.use_texture = False
            from inferno.models.temporal.Preprocessors import FlamePreprocessor
            self.flame = FlamePreprocessor(flame_cfg)
            # prep = prep.to("cuda")



    def _preload_videos(self): 
        # indices = np.unique(self.video_indices)
        from tqdm import auto
        for i in auto.tqdm( range(len(self.video_indices)), desc="Preloading videos" ):
            video_path = str(self._get_video_path(i))
            if self.read_video:
                if video_path not in self.video_cache:
                    self.video_cache[video_path] = vread(video_path)
                    self.seg_cache[video_path] = self._read_segmentations(i) 
            if self.read_audio:
                if video_path not in self.audio_cache:
                    self.audio_cache[i] = self._read_audio(i)
            for lmk_type, lmk_source in zip(self.landmark_types, self.landmark_source):
                if i not in self.lmk_cache:
                    self.lmk_cache[i] = {}
                if lmk_type not in self.lmk_cache[i]:
                    self.lmk_cache[i][lmk_type] = {}
                # if lmk_source not in self.lmk_cache[i][lmk_type]:
                self.lmk_cache[i][lmk_type][lmk_source] = self._read_landmarks(i, lmk_type, lmk_source)
            if self.reconstruction_type is not None: 
                for rec_type in self.reconstruction_type: 
                    shape_pose_cam, appearance = self._load_reconstructions(i, rec_type, self.return_appearance)
                    if i not in self.rec_cache:
                        self.rec_cache[i] = {}
                    video_dict = self.rec_cache[i]
                    if rec_type not in video_dict:
                        video_dict[rec_type] = {}
                    self.rec_cache[i][rec_type]["shape_pose_cam"] = shape_pose_cam
                    self.rec_cache[i][rec_type]["appearance"] = appearance
            if self.emotion_type is not None: 
                emotions, features = self._load_emotions(i, features=self.return_emotion_feature)
                if i not in self.emo_cache:
                    self.emo_cache[i] = {}
                self.emo_cache[i]["emotions"] = emotions 
                self.emo_cache[i]["features"] = features
    

        print("Video cache loaded")

    def _inflate_by_video_size(self):
        assert isinstance( self.sequence_length, int), "'sequence_length' must be an integer when inflating by video size"
        inflated_video_indices = []
        video_sample_indices = []
        for i in range(len(self.video_indices)):
        # for i in self.video_indices:
            idx = self.video_indices[i]
            num_frames = self._get_num_frames(i)
            if self.temporal_split_start is not None and self.temporal_split_end is not None:
                num_frames = int((self.temporal_split_end - self.temporal_split_start) * num_frames)
            num_samples_in_video = num_frames // self.sequence_length
            if num_frames % self.sequence_length != 0:
                num_samples_in_video += 1
            num_samples_in_video = max(1, num_samples_in_video)
            inflated_video_indices += [idx] * num_samples_in_video
            video_sample_indices += list(range(num_samples_in_video))
        self.video_indices = np.array(inflated_video_indices, dtype=np.int32)
        self.video_sample_indices = np.array(video_sample_indices, dtype=np.int32)

    def __getitem__(self, index):
        # max_attempts = 10
        max_attempts = 50
        for i in range(max_attempts):
            try: 
                return self._getitem(index)
            except AssertionError as e:
                if not hasattr(self, "num_total_failed_attempts"):
                    self.num_total_failed_attempts = 0
                old_index = index
                index = np.random.randint(0, self.__len__())
                tb = traceback.format_exc()
                if self.num_total_failed_attempts % 50 == 0:
                    print(f"[ERROR] AssertionError in {self.__class__.__name__} dataset while retrieving sample {old_index}, retrying with new index {index}")
                    print(f"In total, there has been {self.num_total_failed_attempts} failed attempts. This number should be very small. If it's not, check the data.")
                    print("See the exception message for more details.")
                    print(tb)
                self.num_total_failed_attempts += 1
        print("[ERROR] Failed to retrieve sample after {} attempts".format(max_attempts))
        raise RuntimeError("Failed to retrieve sample after {} attempts".format(max_attempts))

    def _getitem(self, index):
        time = False
        if time:
            import timeit
            start_time = timeit.default_timer()
        if self.hack_length: 
            index = index % self._true_len()

        # 1) VIDEO
        # load the video         
        sample, start_frame, num_read_frames, video_fps, num_frames, num_available_frames = self._get_video(index)
        if time:
            video_read_time = timeit.default_timer() - start_time

        # 2) AUDIO
        if self.read_audio:
            sample = self._get_audio(index, start_frame, num_read_frames, video_fps, num_frames, sample)
        if time:
            audio_read_time = timeit.default_timer() - start_time - video_read_time

        # 3) LANDMARKS 
        sample = self._get_landmarks(index, start_frame, num_read_frames, video_fps, num_frames, sample)
        if time:
            lmk_read_time = timeit.default_timer() - start_time - video_read_time - audio_read_time

        # 4) SEGMENTATIONS
        if self.read_video:
            sample = self._get_segmentations(index, start_frame, num_read_frames, video_fps, num_frames, sample)
        if time:
            seg_read_time = timeit.default_timer() - start_time - video_read_time - audio_read_time - lmk_read_time

        # 5) FACE ALIGNMENT IF ANY
        if self.read_video:
            sample = self._align_faces(index, sample)
        if time:
            face_align_time = timeit.default_timer() - start_time - video_read_time - audio_read_time - lmk_read_time - seg_read_time

        # 6) GEOMETRY 
        if self.reconstruction_type is not None:
            sample = self._get_reconstructions(index, start_frame, num_read_frames, video_fps, num_frames, sample)
        if time:
            geom_read_time = timeit.default_timer() - start_time - video_read_time - audio_read_time - lmk_read_time - seg_read_time - face_align_time

        # 7) EMOTION 
        if self.emotion_type is not None:
            sample = self._get_emotions(index, start_frame, num_read_frames, video_fps, num_frames, sample)
        if time:
            emo_read_time = timeit.default_timer() - start_time - video_read_time - audio_read_time - lmk_read_time - seg_read_time - face_align_time - geom_read_time

        # 8) AUGMENTATION
        if self.read_video:
            sample = self._augment_sequence_sample(index, sample)
        if time:
            aug_time = timeit.default_timer() - start_time - video_read_time - audio_read_time - lmk_read_time - seg_read_time - face_align_time - geom_read_time - emo_read_time

        # TO TORCH
        sample = to_torch(sample)


        # AUDIO NORMALIZATION (if any), this is a remnant from av-hubert and is not being used anywhere, will be removed in the future
        if self.read_audio:
            if self.include_processed_audio:
                if self.audio_normalization is not None:
                    if self.audio_normalization == "layer_norm":
                        sample["audio"] = F.layer_norm(sample["audio"], sample["audio"].shape[1:])
                    else: 
                        raise ValueError(f"Unsupported audio normalization {self.audio_normalization}")
        # audio_process_time = timeit.default_timer() - start_time - video_read_time - audio_read_time - lmk_read_time - seg_read_time - face_align_time - geom_read_time - emo_read_time - aug_time

        if self.read_video:
            # T,H,W,C to T,C,H,W
            sample["video"] = sample["video"].permute(0, 3, 1, 2)
            if "video_masked" in sample.keys():
                sample["video_masked"] = sample["video_masked"].permute(0, 3, 1, 2)
            # sample["segmenation"] = sample["segmenation"].permute(0, 2, 1)
            # sample["segmentation_masked"] = sample["segmentation_masked"].permute(0, 2, 1)

            if self.return_mica_image: 
                fan_landmarks = None
                landmarks_validity = None
                if "landmarks" in sample.keys():
                    if isinstance(sample["landmarks"], dict):
                        if "fan3d" in sample["landmarks"].keys():
                            fan_landmarks = sample["landmarks"]["fan3d"]
                            landmarks_validity = sample["landmarks_validity"]["fan3d"]
                        elif "fan" in sample["landmarks"].keys():
                            fan_landmarks = sample["landmarks"]["fan"]
                            landmarks_validity = sample["landmarks_validity"]["fan"]
                    elif isinstance(sample["landmarks"], (np.ndarray, torch.Tensor)):
                        if sample["landmarks"].shape[1] == 68:
                            fan_landmarks = sample["landmarks"]
                            landmarks_validity = sample["landmarks_validity"]
                
                sample["mica_video"] = self.mica_preprocessor(sample["video"], fan_landmarks, landmarks_validity=landmarks_validity)
                sample["mica_video_masked"] = self.mica_preprocessor(sample["video_masked"], fan_landmarks, landmarks_validity=landmarks_validity)


        # # normalize landmarks 
        # if self.landmark_normalizer is not None:
        #     if isinstance(self.landmark_normalizer, KeypointScale):
        #         raise NotImplementedError("Landmark normalization is deprecated")
        #         self.landmark_normalizer.set_scale(
        #             img.shape[0] / input_img_shape[0],
        #             img.shape[1] / input_img_shape[1])
        #     elif isinstance(self.landmark_normalizer, KeypointNormalization):
        #         self.landmark_normalizer.set_scale(sample["video"].shape[2], sample["video"].shape[3])
        #     else:
        #         raise ValueError(f"Unsupported landmark normalizer type: {type(self.landmark_normalizer)}")
        #     for key in sample["landmarks"].keys():
        #         sample["landmarks"][key] = self.landmark_normalizer(sample["landmarks"][key])
        if time:    
            print(f"Video read time: {video_read_time:.2f} s")
            print(f"Audio read time: {audio_read_time:.2f} s")
            print(f"Landmark read time: {lmk_read_time:.2f} s")
            print(f"Segmentation read time: {seg_read_time:.2f} s")
            print(f"Face alignment time: {face_align_time:.2f} s")
            print(f"Geometry read time: {geom_read_time:.2f} s")
            print(f"Emotion read time: {emo_read_time:.2f} s")
            print(f"Augmentation time: {aug_time:.2f} s")
            # print(f"Audio process time: {audio_process_time:.2f} s")
            print(f"Total read time: {timeit.default_timer() - start_time:.2f} s")
        return sample

    def _get_video_path(self, index):
        if self.use_original_video:
            video_path = self.root_path / self.video_list[self.video_indices[index]]
        else: 
            video_path = Path(self.output_dir) / "videos_aligned" / self.video_list[self.video_indices[index]]
        return video_path

    def _get_audio_path(self, index):
        audio_path = (Path(self.output_dir) / "audio" / self.video_list[self.video_indices[index]]).with_suffix(".wav")
        return audio_path

    def _get_num_frames(self, index):
        video_meta = self.video_metas[self.video_indices[index]]
        # print("Video path: ", video_path)
        # num video frames 
        num_frames = video_meta["num_frames"]
        video_path = self._get_video_path(index)
        if num_frames == 0: 
            # use ffprobe to get the number of frames
            num_frames = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video_path)]))
        if num_frames == 0: 
            _vr =  FFmpegReader(str(video_path))
            num_frames = _vr.getShape()[0]
            del _vr
        return num_frames

    def _get_sample_length(self, index):
        if isinstance(self.sequence_length, int): # if sequence length set, use it
            return self.sequence_length
        elif isinstance(self.sequence_length, str): # otherwise use the one from the metadata
            if self.sequence_length == "all":
                if self.temporal_split_start is not None and self.temporal_split_end is not None:
                    num_frames = self._get_num_frames(index)
                    temporal_split_start_frame = int(self.temporal_split_start * num_frames)
                    temporal_split_end_frame = int(self.temporal_split_end * num_frames) 
                    return temporal_split_end_frame - temporal_split_start_frame
                else:
                    num_frames = self._get_num_frames(index)
            else: 
                raise ValueError(f"Unsupported sequence length value: '{self.sequence_length}'")
            return num_frames
        raise 

    def _get_video(self, index):
        video_path = self._get_video_path(index)
        video_meta = self.video_metas[self.video_indices[index]]
        # print("Video path: ", video_path)
        # num video frames 
        num_frames = self._get_num_frames(index)
        assert num_frames > 0, "Number of frames is 0 for video {}".format(video_path)
        video_fps = video_meta["fps"]
        n1, n2 = video_fps.split("/")
        n1 = int(n1)
        n2 = int(n2)
        assert n1 % n2 == 0
        video_fps = n1 // n2

        # assert num_frames >= self.sequence_length, f"Video {video_path} has only {num_frames} frames, but sequence length is {self.sequence_length}"
        # TODO: handle the case when sequence length is longer than the video length

        sequence_length = self._get_sample_length(index)

        # pick the starting video frame 
        if self.temporal_split_start is not None and self.temporal_split_end is not None:
            temporal_split_start = int(self.temporal_split_start * num_frames)
            temporal_split_end = int(self.temporal_split_end * num_frames) 
            num_available_frames = temporal_split_end - temporal_split_start
            # start_frame = np.random.randint(temporal_split_start, temporal_split_end - sequence_length)
        else: 
            temporal_split_start = 0
            temporal_split_end = num_frames
            num_available_frames = num_frames

        if num_available_frames <= sequence_length:
            start_frame = temporal_split_start
        else:
            if self.video_sample_indices is None: # one video is one sample
                start_frame = np.random.randint(temporal_split_start, temporal_split_end - sequence_length)
            else: # one video is multiple samples (as many as the sequence length allows without repetition)
                start_frame = temporal_split_start + (self.video_sample_indices[index] * sequence_length)
            # start_frame = np.random.randint(0, num_frames - sequence_length)

        sample = {}
        if self.include_filename: 
            sample["filename"] = str(video_path)
            sample["fps"] = video_fps # include the fps in the sample

        # TODO: picking the starting frame should probably be done a bit more robustly 
        # (e.g. by ensuring the sequence has at least some valid landmarks) ... 
        # maybe the video should be skipped altogether if it can't provide that 

        # load the frames
        # frames = []
        # for i in range(start_frame, start_frame + sequence_length):
        #     frame_path = video_path / f"frame_{i:04d}.jpg"
        #     frame = imread(str(frame_path))
        #     frames.append(frame)
        assert video_path.is_file(), f"Video {video_path} does not exist"
        num_read_frames = self._get_sample_length(index)
        num_read_frames_ = self._get_sample_length(index)
        if self.read_video:
            num_read_frames = 0
            try:
                if not self.preload_videos:
                    # import timeit
                    # start_time = timeit.default_timer()
                    # frames = vread(video_path.as_posix())
                    # end_time = timeit.default_timer()
                    # print(f"Video read time: {end_time - start_time:.2f} s")
                    # from decord import VideoReader
                    # from decord import cpu, gpu
                    # start_time = timeit.default_timer()
                    vr = VideoReader(video_path.as_posix(), ctx=cpu(0), width=self.image_size, height=self.image_size) 
                    if len(vr) < sequence_length:
                        sequence_length_ = len(vr)
                    else: 
                        sequence_length_ = sequence_length
                    frames = vr.get_batch(range(start_frame,(start_frame + sequence_length_)))  
                    frames = frames.asnumpy()

                    if sequence_length_ < sequence_length:
                        # pad with zeros if video shorter than sequence length
                        frames = np.concatenate([frames, np.zeros((sequence_length - frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]), dtype=frames.dtype)])

                    # end_time = timeit.default_timer()
                    # print(f"Video read time: {end_time - start_time:.2f} s")
                else: 
                    frames = self.video_cache[video_path.as_posix()]
                    assert len(frames) == num_frames, f"Video {video_path} has {len(frames)} frames, but meta says it has {num_frames}"
                    frames = frames[start_frame:(start_frame + sequence_length)] 
                num_read_frames = frames.shape[0]
                # # plot frames 
                # import matplotlib.pyplot as plt
                # frame_idx = 0 
                # plt.figure()
                # plt.imshow(frames[frame_idx])
                # plt.show()
                if frames.shape[0] < sequence_length:
                    # pad with zeros if video shorter than sequence length
                    frames = np.concatenate([frames, np.zeros((sequence_length - frames.shape[0], frames.shape[1], frames.shape[2]), dtype=frames.dtype)])
            except ValueError: 
                # reader = vreader(video_path.as_posix())
                # create an opencv video reader 
                import cv2
                reader = cv2.VideoCapture(video_path.as_posix())
                fi = 0 
                frames = []
                while fi < start_frame:
                    fi += 1
                    # _ = next(reader) 
                    _, frame = reader.read()
                for i in range(sequence_length):
                    # frames.append(next(reader))
                    if reader.isOpened():
                        _, frame = reader.read()
                        if frame is None: 
                            # frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                            frame = np.zeros_like(frames[0])
                            frames.append(frame)
                            continue
                        num_read_frames += 1
                        # bgr to rgb 
                        frame = frame[:, :, ::-1]
                    else: 
                        # if we ran out of frames, pad with black
                        frame = np.zeros_like(frames[0])
                    frames.append(frame)
                reader.release()
                frames = np.stack(frames, axis=0)
            frames = frames.astype(np.float32) / 255.0

            # sample = { 
            sample["video"] = frames

        sample["frame_indices"] = np.arange(start_frame, start_frame + sequence_length, dtype=np.int32)
        
        if num_read_frames_ != num_read_frames:
            print(f"[Warning]: read {num_read_frames} frames instead of {num_read_frames_} for video {video_path}")

        return sample, start_frame, num_read_frames, video_fps, num_frames, num_available_frames

    def _read_audio(self, index):
        # audio_path = (Path(self.output_dir) / "audio" / self.video_list[self.video_indices[index]]).with_suffix(".wav")
        audio_path = self._get_audio_path(index)
        # audio_meta = self.audio_metas[self.video_indices[index]]
            
        # load the audio 
        # if self.include_raw_audio:
        import librosa
        sampling_rate = 16000
        wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
        # wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
        if wavdata.ndim > 1:
            wavdata = librosa.to_mono(wavdata)
        wavdata = (wavdata.astype(np.float64) * 32768.0).astype(np.int16)
        return wavdata, sampling_rate

    def _get_audio(self, index, start_frame, num_read_frames, video_fps, num_frames, sample):
        if self.preload_videos:
            wavdata, sampling_rate = self.audio_cache[index]
        else:
            wavdata, sampling_rate = self._read_audio(index)
        sequence_length = self._get_sample_length(index)

        # audio augmentation
        if np.random.rand() < self.audio_noise_prob:
            wavdata = self.add_noise(wavdata)

        if self.include_processed_audio:
            # sampling_rate, wavdata = wavfile.read(audio_path.as_posix())

            # assert samplerate == 16000 and len(wavdata.shape) == 1
            audio_feats = logfbank(wavdata, samplerate=sampling_rate).astype(np.float32) # [T (num audio frames), F (num filters)]
            # the audio feats frequency (and therefore num frames) is too high, so we stack them together to match num visual frames 
            audio_feats = stacker(audio_feats, self.stack_order_audio)

            # audio_feats = audio_feats[start_frame:(start_frame + sequence_length)] 
            audio_feats = audio_feats[start_frame:(start_frame + num_read_frames)] 
            # temporal pad with zeros if necessary to match the desired video length 
            if audio_feats.shape[0] < sequence_length:
                # concatente with zeros
                audio_feats = np.concatenate([audio_feats, 
                    np.zeros((sequence_length - audio_feats.shape[0], audio_feats.shape[1]),
                    dtype=audio_feats.dtype)], axis=0)
            
            # stack the frames and audio feats together
            sample["audio"] = audio_feats

            
        if self.include_raw_audio:
            assert sampling_rate % video_fps == 0 
            wav_per_frame = sampling_rate // video_fps 
            wavdata_ = np.zeros((num_frames, wav_per_frame), dtype=wavdata.dtype) 
            wavdata_ = wavdata_.reshape(-1)
            if wavdata.size > wavdata_.size:
                wavdata_[...] = wavdata[:wavdata_.size]
            else: 
                wavdata_[:wavdata.size] = wavdata
            wavdata_ = wavdata_.reshape((num_frames, wav_per_frame))
            wavdata_ = wavdata_[start_frame:(start_frame + num_read_frames)] 
            if wavdata_.shape[0] < sequence_length:
                # concatente with zeros
                wavdata_ = np.concatenate([wavdata_, 
                    np.zeros((sequence_length - wavdata_.shape[0], wavdata_.shape[1]),
                    dtype=wavdata_.dtype)], axis=0)
            wavdata_ = wavdata_.astype(np.float64) / np.int16(np.iinfo(np.int16).max)

            # wavdata_ = np.zeros((sequence_length, samplerate // video_fps), dtype=wavdata.dtype)
            # wavdata_ = np.zeros((n * frames.shape[0]), dtype=wavdata.dtype)
            # wavdata_[:wavdata.shape[0]] = wavdata 
            # wavdata_ = wavdata_.reshape((frames.shape[0], -1))
            sample["raw_audio"] = wavdata_ 
            sample["samplerate"] = sampling_rate

        return sample

    def _path_to_landmarks(self, index, landmark_type, landmark_source): 
        return  (Path(self.output_dir) / f"landmarks_{landmark_source}" / landmark_type /  self.video_list[self.video_indices[index]]).with_suffix("")

    def _read_landmarks(self, index, landmark_type, landmark_source):
        landmarks_dir = self._path_to_landmarks(index, landmark_type, landmark_source)
        landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / f"landmarks_{landmark_source}.pkl")  
        return landmark_list

    def _get_landmarks(self, index, start_frame, num_read_frames, video_fps, num_frames, sample): 
        sequence_length = self._get_sample_length(index)
        landmark_dict = {}
        landmark_validity_dict = {}
        for lti, landmark_type in enumerate(self.landmark_types):
            landmark_source = self.landmark_source[lti]
            landmarks_dir = self._path_to_landmarks(index, landmark_type, landmark_source)
            landmarks = []
            if (landmarks_dir / "landmarks.pkl").exists(): # landmarks are saved per video in a single file
            #    landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmarks.pkl")  
            #    landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmarks_original.pkl") 
                if not self.preload_videos: 
                    # landmark_list = FaceDataModuleBase.load_landmark_list(landm?arks_dir / f"landmarks_{landmark_source}.pkl")  
                    landmark_list = self._read_landmarks(index, landmark_type, landmark_source)
                    # landmark_types =  FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmark_types.pkl")  
                else: 
                    landmark_list = self.lmk_cache[index][landmark_type][landmark_source]
                    # landmark_types = self.lmk_cache[index]["landmark_types"]
                landmarks = landmark_list[start_frame: sequence_length + start_frame] 
                landmark_validity = np.ones((len(landmarks), 1), dtype=np.float32)
                for li in range(len(landmarks)): 
                    if len(landmarks[li]) == 0: # dropped detection
                        if landmark_type == "mediapipe":
                            # [WARNING] mediapipe landmarks coordinates are saved in the scale [0.0-1.0] (for absolute they need to be multiplied by img size)
                            landmarks[li] = np.zeros((MEDIAPIPE_LANDMARK_NUMBER, 3))
                        elif landmark_type in ["fan", "kpt68"]:
                            landmarks[li] = np.zeros((68, 2))
                        else: 
                            raise ValueError(f"Unknown landmark type '{landmark_type}'")
                        landmark_validity[li] = 0.
                    elif len(landmarks[li]) > 1: # multiple faces detected
                        landmarks[li] = landmarks[li][0] # just take the first one for now
                    else: \
                        landmarks[li] = landmarks[li][0] 

                # # pad landmarks with zeros if necessary to match the desired video length
                # # if landmarks.shape[0] < sequence_length:
                # if len(landmarks) < sequence_length:
                #     # concatente with zeros
                #     landmarks += [np.zeros((landmarks.shape[1]))] * (sequence_length - len(landmarks))
                    

                #     landmarks = np.concatenate([landmarks, np.zeros((sequence_length - landmarks.shape[0], landmarks.shape[1]))], axis=0)
                #     landmark_validity = np.concatenate([landmark_validity, np.zeros((sequence_length - landmark_validity.shape[0]), dtype=np.bool)], axis=0)
            else: # landmarks are saved per frame
                landmark_validity = np.ones((len(landmarks), 1), dtype=np.float32)
                for i in range(start_frame, sequence_length + start_frame):
                    landmark_path = landmarks_dir / f"{i:05d}_000.pkl"
                    landmark_type, landmark = load_landmark(landmark_path)
                    landmarks += [landmark]
                    if len(landmark) == 0: # dropped detection
                        landmark = [0, 0]
                        landmark_validity[li] = 0.
                    elif len(landmark) > 1: # multiple faces detected
                        landmarks[li] = landmarks[li][0] # just take the first one for now
                    else: 
                        landmark[li] = landmarks[li][0] 
            landmarks = np.stack(landmarks, axis=0)
            # if landmark_type == "mediapipe" and self.align_images: 
            # #     # [WARNING] mediapipe landmarks coordinates are saved in the scale [0.0-1.0] (for absolute they need to be multiplied by img size)
            # #     # landmarks -= 0.5 
            # #     landmarks -= 1. 
            #     landmarks *= 2 
            # # #     landmarks *= 2 
            #     landmarks -= 1

            # pad landmarks with zeros if necessary to match the desired video length
            if landmarks.shape[0] < sequence_length:
                landmarks = np.concatenate([landmarks, np.zeros(
                    (sequence_length - landmarks.shape[0], *landmarks.shape[1:]), 
                    dtype=landmarks.dtype)], axis=0)
                landmark_validity = np.concatenate([landmark_validity, np.zeros((sequence_length - landmark_validity.shape[0], 1), 
                    dtype=landmark_validity.dtype)], axis=0)

            landmark_dict[landmark_type] = landmarks.astype(np.float32)
            landmark_validity_dict[landmark_type] = landmark_validity


        sample["landmarks"] = landmark_dict
        sample["landmarks_validity"] = landmark_validity_dict
        return sample

    def _path_to_segmentations(self, index): 
        return (Path(self.output_dir) / f"segmentations_{self.segmentation_source}" / self.segmentation_type /  self.video_list[self.video_indices[index]]).with_suffix("")

    def _read_segmentations(self, index, start_frame=None, end_frame=None):
        segmentations_dir = self._path_to_segmentations(index)
        if (segmentations_dir / "segmentations.hdf5").exists(): # if random access hdf5 exists (newest), let's use it
            segmentations, seg_types, seg_names = load_segmentation_list_v2(segmentations_dir / "segmentations.hdf5", start_frame, end_frame)
        elif (segmentations_dir / "segmentations.pkl").exists(): # segmentations are saved in a single pickle (no random access)
            segmentations, seg_types, seg_names = load_segmentation_list(segmentations_dir / "segmentations.pkl")
            if start_frame is not None and end_frame is not None:
                segmentations = segmentations[start_frame: end_frame]
                seg_types = seg_types[start_frame: end_frame]
                seg_names = seg_names[start_frame: end_frame]
        if isinstance(segmentations, list):
            segmentations = np.stack(segmentations, axis=0)
        if segmentations.ndim == 4: # T, C=1, W, H
            segmentations = segmentations[:,0,...]
        if isinstance(seg_types[0], bytes):
            seg_types = [seg_type.decode("utf-8") for seg_type in seg_types]
        if isinstance(seg_names[0], bytes):
            seg_names = [seg_name.decode("utf-8") for seg_name in seg_names]
        return segmentations, seg_types, seg_names

    def _retrieve_segmentations(self, index, start_frame, end_frame):
        if not self.preload_videos:
            # segmentations_dir = self._path_to_segmentations(index)
            # if (segmentations_dir / "segmentations.hdf5").exists(): # random access hdf5 exists, let's use it
            #     segmentations, seg_types, seg_names = load_segmentation_list_v2(segmentations_dir / "segmentations.hdf5", start_frame, end_frame)
            # elif (segmentations_dir / "segmentations.pkl").exists(): # segmentations are saved in a single pickle
            #     seg_images, seg_types, seg_names = load_segmentation_list(segmentations_dir / "segmentations.pkl")
            #     segmentations = seg_images[start_frame: end_frame]
            # if isinstance(seg_images, list):
            #     segmentations = np.stack(seg_images, axis=0)
            # if seg_images.ndim == 4: # T, C=1, W, H
            #     segmentations = segmentations[:,0,...]
            segmentations, seg_types, seg_names = self._read_segmentations(index, start_frame, end_frame)

            return segmentations, seg_types, seg_names
        else:
            video_path = str(self._get_video_path(index))
            segmentations, seg_types, seg_names = self.seg_cache[video_path]
            segmentations = segmentations[start_frame: end_frame]
            seg_types = seg_types[start_frame: end_frame]
            seg_names = seg_names[start_frame: end_frame]
            return segmentations, seg_types, seg_names

    def _load_reconstructions(self, index, rec_type, appearance=False, start_frame=None, end_frame=None): 
        reconstructions_dir = self._path_to_reconstructions(index, rec_type)
        if (reconstructions_dir / "shape_pose_cam.hdf5").exists(): # random access hdf5 exists, let's use it
            shape_pose_cam = load_reconstruction_list_v2(reconstructions_dir / "shape_pose_cam.hdf5", 
                                                         start_frame=start_frame, end_frame=end_frame)
            if appearance:
                appearance = load_reconstruction_list_v2(reconstructions_dir / "appearance.hdf5", 
                                                         start_frame=start_frame, end_frame=end_frame)
            else: 
                appearance = None
        elif (reconstructions_dir / "shape_pose_cam.pkl").exists(): # reconstructions are saved in a single pickle
            shape_pose_cam = load_reconstruction_list(reconstructions_dir / "shape_pose_cam.pkl", 
                                                         start_frame=start_frame, end_frame=end_frame)
            if appearance:
                appearance = load_reconstruction_list(reconstructions_dir / "appearance.pkl", 
                                                         start_frame=start_frame, end_frame=end_frame)
            else: 
                appearance = None

            ## should no longer be necessary as the start/end frame is now handled in the load_reconstruction_list function
            # if start_frame is not None and end_frame is not None:
            #     shape_pose_cam = {key: shape_pose_cam[key][:, start_frame: end_frame] for key in shape_pose_cam.keys()}
            #     if appearance is not None:
            #         appearance = {key: appearance[key][:, start_frame: end_frame] for key in appearance.keys()}
        else: 
            raise RuntimeError(f"Reconstruction file not found in {reconstructions_dir}")
        # for key in shape_pose_cam.keys():
        #     shape_pose_cam[key] = np.copy(shape_pose_cam[key])
            # for key in appearance.keys():
            #     appearance[key] = np.copy(appearance[key])
        return shape_pose_cam, appearance

    def _load_emotions(self, index, features=False, start_frame=None, end_frame=None): 
        emotions_dir = self._path_to_emotions(index)

        if (emotions_dir / "emotions.hdf5").exists(): # random access hdf5 exists, let's use it
            emotions = load_emotion_list_v2(emotions_dir / "emotions.hdf5", start_frame, end_frame)
            if features:
                features = load_emotion_list_v2(emotions_dir / "features.hdf5", start_frame, end_frame)
            else:
                features = None
        elif (emotions_dir / "emotions.pkl").exists(): # emotions are saved in a single pickle
            emotions = load_emotion_list(emotions_dir / "emotions.pkl")
            if features:
                features = load_emotion_list(emotions_dir / "features.pkl")
                assert "feature" in features.keys(), "Features not found in emotion file. This is likely due to a bug in emotions saving. " \
                    "Please delete the emotion feature file and recompute them."
            else: 
                features = None
            if start_frame is not None and end_frame is not None:
                emotions = emotions[start_frame: end_frame]
                if features is not None:
                    features = features[start_frame: end_frame]
        else: 
            raise RuntimeError(f"Emotion file not found in {emotions_dir}")
        return emotions, features
        
    def _path_to_reconstructions(self, index, rec_type): 
        return (Path(self.output_dir) / f"reconstructions" / rec_type /  self.video_list[self.video_indices[index]]).with_suffix("")
        # return (Path(self.output_dir) / f"reconstructions" / self.reconstruction_type /  self.video_list[self.video_indices[index]]).with_suffix("")

    def _path_to_emotions(self, index): 
        return (Path(self.output_dir) / f"emotions" / self.emotion_type /  self.video_list[self.video_indices[index]]).with_suffix("")

    def _get_segmentations(self, index, start_frame, num_read_frames, video_fps, num_frames, sample): 
        segmentations_dir = self._path_to_segmentations(index)
        segmentations = []

        sequence_length = self._get_sample_length(index)

        if (segmentations_dir / "segmentations.pkl").exists() or (segmentations_dir / "segmentations.hdf5").exists(): # segmentations are saved in a single file-per video 
            # seg_images, seg_types, seg_names = load_segmentation_list(segmentations_dir / "segmentations.pkl")
            # segmentations = seg_images[start_frame: sequence_length + start_frame]
            # segmentations = np.stack(segmentations, axis=0)[:,0,...]
            segmentations, seg_types, seg_names  = self._retrieve_segmentations(index, start_frame, sequence_length + start_frame)
            # segmentations = segmentations[start_frame: sequence_length + start_frame]
            segmentations = process_segmentation(segmentations, seg_types[0]).astype(np.uint8)
            # assert segmentations.shape[0] == sequence_length
        else: # segmentations are saved per-frame (old deprecated options)
            for i in range(start_frame, sequence_length + start_frame):
                segmentation_path = segmentations_dir / f"{i:05d}.pkl"
                seg_image, seg_type = load_segmentation(segmentation_path)
                # seg_image = seg_image[:, :, np.newaxis]
                seg_image = process_segmentation(seg_image[0], seg_type).astype(np.uint8)
                segmentations += [seg_image]
            segmentations = np.stack(segmentations, axis=0)
        if segmentations.shape[0] < sequence_length:
                # pad segmentations with zeros to match the sequence length
                segmentations = np.concatenate([segmentations, 
                    np.zeros((sequence_length - segmentations.shape[0], segmentations.shape[1], segmentations.shape[2]),
                        dtype=segmentations.dtype)], axis=0)

        # ## resize segmentation to the expected image size
        # if segmentations.shape[1] != self.image_size and segmentations.shape[2] != self.image_size:
        #     # sample["segmentation"] = np.zeros((seg_frames.shape[0], self.image_size, self.image_size), dtype=seg_frames.dtype)
        #     # for i in range(seg_frames.shape[0]):
        #     #     sample["segmentation"][i] = cv2.resize(seg_frames[i], (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        #     # do the interpolation with pytorch functional instead: 
        #     seg_frames = torch.from_numpy(segmentations) 
        #     channel_added = False
        #     if seg_frames.ndim == 3: 
        #         # add the channel dim 
        #         seg_frames = seg_frames.unsqueeze(1)
        #         channel_added = True
        #     seg_frames = F.interpolate(seg_frames, size=(self.image_size, self.image_size), mode="nearest")
        #     if channel_added:
        #         seg_frames = seg_frames.squeeze(1)
        #     segmentations = seg_frames.numpy()

        sample["segmentation"] = segmentations
        return sample


    def _align_faces(self, index, sample):
        if self.align_images:
            
            sequence_length = self._get_sample_length(index)

            landmarks_for_alignment = "mediapipe"
            left = sample["landmarks"][landmarks_for_alignment][:,:,0].min(axis=1) / self.original_image_size * self.image_size
            top =  sample["landmarks"][landmarks_for_alignment][:,:,1].min(axis=1)  / self.original_image_size * self.image_size
            right =  sample["landmarks"][landmarks_for_alignment][:,:,0].max(axis=1)  / self.original_image_size * self.image_size
            bottom = sample["landmarks"][landmarks_for_alignment][:,:,1].max(axis=1)  / self.original_image_size * self.image_size

            invalid_frames = np.logical_and(left == 0., np.logical_and(right == 0., np.logical_and(top == 0., bottom == 0.)))
            invalid_indices = np.where(invalid_frames)[0]
            valid_indices = np.where(np.logical_not(invalid_frames))[0]

            if len(invalid_indices) > 0: 
                sample["landmarks_validity"][landmarks_for_alignment][invalid_indices] = False

            if len(invalid_indices) == invalid_frames.size:
                # no valid indices, make up dummy one (zoom in a little bit)
                top = np.array([self.image_size//16]*invalid_frames.size)
                left = np.array([self.image_size//16]*invalid_frames.size)
                bottom = np.array([self.image_size - self.image_size//6]*invalid_frames.size)
                right = np.array([self.image_size - self.image_size//8]*invalid_frames.size)

            elif len(invalid_indices) > 0:
                first_valid_frame = valid_indices.min()
                last_valid_frame = valid_indices.max()

                left_ = left.copy()
                top_ = top.copy()
                right_ = right.copy()
                bottom_ = bottom.copy()

                left_[invalid_indices] = np.nan
                top_[invalid_indices] = np.nan
                right_[invalid_indices] = np.nan
                bottom_[invalid_indices] = np.nan

                # just copy over the first valid frame 
                if first_valid_frame > 0:
                    left_[0] = left[first_valid_frame]
                    top_[0] = top[first_valid_frame]
                    right_[0] = right[first_valid_frame]
                    bottom_[0] = bottom[first_valid_frame]

                # just copy over the last valid frame 
                if last_valid_frame < sequence_length - 1:
                    left_[-1] = left[last_valid_frame]
                    top_[-1] = top[last_valid_frame]
                    right_[-1] = right[last_valid_frame]
                    bottom_[-1] = bottom[last_valid_frame]

                # interpolate using pandas
                left_pd = pd.Series(left_)
                top_pd = pd.Series(top_)
                right_pd = pd.Series(right_)
                bottom_pd = pd.Series(bottom_)

                left_pd.interpolate(inplace=True)
                top_pd.interpolate(inplace=True)
                right_pd.interpolate(inplace=True)
                bottom_pd.interpolate(inplace=True)

                left = left_pd.to_numpy()
                top = top_pd.to_numpy()
                right = right_pd.to_numpy()
                bottom = bottom_pd.to_numpy()

            seg_left = left * self.original_image_size / self.image_size
            seg_top = top * self.original_image_size / self.image_size
            seg_right = right * self.original_image_size / self.image_size
            seg_bottom = bottom * self.original_image_size / self.image_size

            old_size, center = bbox2point(left, right, top, bottom, type=landmarks_for_alignment)
            size = (old_size * self.scale).astype(np.int32)


            old_size_seg, center_seg = bbox2point(seg_left, seg_right, seg_top, seg_bottom, type=landmarks_for_alignment)
            size_seg = (old_size_seg * self.scale).astype(np.int32)

            video_frames = sample["video"]
            sample["video"] = np.zeros((video_frames.shape[0], self.image_size, self.image_size, video_frames.shape[-1]), dtype=video_frames.dtype)
            
            if "segmentation" in sample.keys():
                seg_frames = sample["segmentation"]

                sample["segmentation"] = np.zeros((seg_frames.shape[0], self.image_size, self.image_size), dtype=seg_frames.dtype)

            for key in sample["landmarks"].keys():
                sample["landmarks"][key] *= self.image_size / self.original_image_size

            for i in range(sequence_length):
                lmk_to_warp = {k: v[i] for k,v in sample["landmarks"].items()}
                
                img_warped, lmk_warped = bbpoint_warp(video_frames[i], center[i], size[i], self.image_size, landmarks=lmk_to_warp)
                
                if "segmentation" in sample.keys():
                    seg_warped = bbpoint_warp(seg_frames[i], center_seg[i], size_seg[i], self.image_size, 
                        order=0 # nearest neighbor interpolation for segmentation
                        )
                # img_warped *= 255.
                if not self._allow_alignment_fail:
                    assert np.isnan(img_warped).sum() == 0, f"NaNs in image {i} after face aligning image warp." \
                        f"Center: {center[i]}, size: {size[i]}. Are these values valid?"
                else: 
                    if np.isnan(img_warped).sum() > 0:
                        img_warped[np.isnan(img_warped)] = 0.
                        print('[WARNING] NaNs in image after face aligning image warp. Center: {}, size: {}. Are these values valid?'.format(center[i], size[i]))
                sample["video"][i] = img_warped 
                # sample["segmentation"][i] = seg_warped * 255.
                if "segmentation" in sample.keys():
                    sample["segmentation"][i] = seg_warped
                for k,v in lmk_warped.items():
                    sample["landmarks"][k][i][:,:2] = v
        else: 
            # no alignment, just resize the images
            video_frames = sample["video"]
            if sample["video"].shape[1] != self.image_size and sample["video"].shape[2] != self.image_size:
                sample["video"] = np.zeros((video_frames.shape[0], self.image_size, self.image_size, video_frames.shape[-1]), dtype=video_frames.dtype)
                # resize with pytorch 
                video_frames = torch.from_numpy(video_frames)
                channel_added = False
                if video_frames.ndim == 3:
                    video_frames = video_frames.unsqueeze(1)
                    channel_added = True
                video_frames = F.interpolate(video_frames, size=(self.image_size, self.image_size), mode="bilinear")
                if channel_added:
                    video_frames = video_frames.squeeze(1)
                sample["video"] = video_frames.numpy()
                # for i in range(video_frames.shape[0]):
                #     sample["video"][i] = cv2.resize(video_frames[i], (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            
            if "segmentation" in sample.keys():
                seg_frames = sample["segmentation"]
                if seg_frames.shape[1] != self.image_size and seg_frames.shape[2] != self.image_size:
                    # sample["segmentation"] = np.zeros((seg_frames.shape[0], self.image_size, self.image_size), dtype=seg_frames.dtype)
                    # for i in range(seg_frames.shape[0]):
                    #     sample["segmentation"][i] = cv2.resize(seg_frames[i], (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

                    # do the interpolation with pytorch functional instead: 
                    seg_frames = torch.from_numpy(seg_frames) 
                    channel_added = False
                    if seg_frames.ndim == 3: 
                        # add the channel dim 
                        seg_frames = seg_frames.unsqueeze(1)
                        channel_added = True
                    seg_frames = F.interpolate(seg_frames, size=(self.image_size, self.image_size), mode="nearest")
                    if channel_added:
                        seg_frames = seg_frames.squeeze(1)
                    sample["segmentation"] = seg_frames.numpy()
            if "landmarks" in sample.keys():
                for k,v in sample["landmarks"].items():
                    sample["landmarks"][k][...,:2] *= self.image_size / self.original_image_size

        return sample

    def _get_reconstructions(self, index, start_frame, num_read_frames, video_fps, num_frames, sample):
        for reconstruction_type in self.reconstruction_type: 
            sample = self._get_reconstructions_of_type(index, start_frame, num_read_frames, video_fps, num_frames, sample, reconstruction_type)
        return sample

    def _get_reconstructions_of_type(self, index, start_frame, num_read_frames, video_fps, num_frames, sample, rec_type):
        sequence_length = self._get_sample_length(index)
        if self.reconstruction_type is None:
            return sample
        if not self.preload_videos:
            shape_pose_cam, appearance = self._load_reconstructions(index, rec_type, self.return_appearance, 
                                                                    start_frame=start_frame, end_frame=start_frame+num_read_frames)
        else: 
            shape_pose_cam_ = self.rec_cache[index][rec_type]["shape_pose_cam"]
            appearance_ = self.rec_cache[index][rec_type]["appearance"]
            shape_pose_cam = shape_pose_cam_.copy()
            if appearance_ is not None:
                appearance = appearance_.copy()
            else:
                appearance = None
        for key in shape_pose_cam.keys():
            shape_pose_cam[key] = shape_pose_cam[key][0]
        if appearance is not None:
            for key in appearance.keys():
                appearance[key] = appearance[key][0]
            
        weights = sample["landmarks_validity"]["mediapipe"] / sample["landmarks_validity"]["mediapipe"].sum(axis=0, keepdims=True)
        assert np.isnan(weights).any() == False, "NaN in weights"


        if shape_pose_cam['exp'].shape[0] < sequence_length: # pad with zero to be the same length as the sequence
            for key in shape_pose_cam.keys():
                # if key != 'shape':
                shape_pose_cam[key] = np.concatenate([shape_pose_cam[key], np.zeros((sequence_length - shape_pose_cam[key].shape[0], shape_pose_cam[key].shape[1]))], axis=0)
            if appearance is not None:
                for key in appearance.keys():
                    appearance[key] = np.concatenate([appearance[key], np.zeros((sequence_length - appearance[key].shape[0], appearance[key].shape[1]))], axis=0)
        
        # if the start_frame is in the beginnning, zero out the first few frames
        if start_frame < self.invalid_cutoff: 
            diff = self.invalid_cutoff - start_frame
            # if shape_pose_cam['shape'].ndim == 3:
            #     shape_pose_cam['shape'][:diff] = 0
            # for key in shape_pose_cam.keys():
            #     shape_pose_cam[key][:diff] = 0

            # if appearance is not None:
            #     for key in appearance.keys():
            #         appearance[key][:diff] = 0

            sample["landmarks_validity"]["mediapipe"][:diff] = 0

        # if the start_frame is in the end, zero out the last few frames
        if start_frame + num_read_frames > num_frames - self.invalid_cutoff:
            diff = start_frame + num_read_frames - (num_frames - self.invalid_cutoff)
            # if shape_pose_cam['shape'].ndim == 3:
            #     shape_pose_cam['shape'][-diff:] = 0
            # for key in shape_pose_cam.keys():
            #     shape_pose_cam[key][-diff:] = 0

            # if appearance is not None:
            #     for key in appearance.keys():
            #         appearance[key][-diff:] = 0

            sample["landmarks_validity"]["mediapipe"][-diff:] = 0

        if self.average_shape_decode:
            shape = (weights[:shape_pose_cam['shape'].shape[0]] * shape_pose_cam['shape']).sum(axis=0, keepdims=False)
            # shape = np.tile(shape, (shape_pose_cam['shape'].shape[0], 1))
        else:
            shape = shape_pose_cam['shape']
            if shape_pose_cam['exp'].shape[0] < sequence_length:    # pad with zero to be the same length as the sequence
                shape_pose_cam['shape'] = np.concatenate([shape_pose_cam['shape'], 
                                                          np.zeros((sequence_length - shape_pose_cam['shape'].shape[0], shape_pose_cam['exp'].shape[1]))], 
                                                          axis=0)
                # shape = np.concatenate([shape, np.tile(shape[-1], (sequence_length - shape.shape[0], 1))], axis=0)


        if self.return_appearance: 
            if self.average_shape_decode: 
                appearance['tex'] = (weights[:appearance['tex'].shape[0]] * appearance['tex']).sum(axis=0, keepdims=False)
                # appearance = np.tile(appearance, (appearance['tex'].shape[0], 1))
            else:
                # appearance = appearance['tex']
                if  shape_pose_cam['exp'].shape[0] < sequence_length:
                    appearance['tex'] = np.concatenate([appearance['tex'], 
                                                        np.zeros((sequence_length - appearance['tex'].shape[0], appearance['tex'].shape[1]))], 
                                                        axis=0)
                    # appearance = np.concatenate([appearance, np.tile(appearance[-1], (sequence_length - appearance.shape[0], 1))], axis=0)
        
        # # intialize emtpy dicts if they don't exist
        # if "gt_exp" not in sample.keys():
        #     sample["gt_exp"] = {} 
        # if "gt_shape" not in sample.keys():
        #     sample["gt_shape"] = {} 
        # if "gt_jaw" not in sample.keys():
        #     sample["gt_jaw"] = {} 
        # if self.return_global_pose:
        #     if "gt_global_pose" not in sample.keys():
        #         sample["gt_global_pose"] = {}
        #     if "gt_cam" not in sample.keys():
        #         sample["gt_cam"] = {}
        # if self.return_appearance:
        #     if "gt_tex" not in sample.keys():
        #         sample["gt_tex"] = {}
        #     if "gt_light" not in sample.keys():
        #         sample["gt_light"] = {}
        #     if "detailcode" in appearance: 
        #         if "gt_detail" not in sample.keys():
        #             sample["gt_detail"] = {}
        if "reconstruction" not in sample.keys():
            sample["reconstruction"] = {}
        assert rec_type not in sample["reconstruction"].keys(), "reconstruction type already exists in sample"
        sample["reconstruction"][rec_type] = {}
        
            
        sample["reconstruction"][rec_type]["gt_exp"] = shape_pose_cam['exp'].astype(np.float32)
        sample["reconstruction"][rec_type]["gt_shape"] = shape.astype(np.float32)
        sample["reconstruction"][rec_type]["gt_jaw"] = shape_pose_cam['jaw'].astype(np.float32)
        if self.return_global_pose:
            sample["reconstruction"][rec_type]["gt_global_pose"]= shape_pose_cam['global_pose'].astype(np.float32)
            sample["reconstruction"][rec_type] ["gt_cam"] = shape_pose_cam['cam'].astype(np.float32)
        if self.return_appearance: 
            sample["reconstruction"][rec_type]['gt_tex'] = appearance['tex'].astype(np.float32)
            sample["reconstruction"][rec_type]['gt_light'] = appearance['light'].astype(np.float32)
            if 'detailcode' in appearance:
                sample[rec_type]['gt_detail'] = appearance['detailcode'].astype(np.float32)

        if hasattr(self, 'flame') and self.flame is not None:
            flame_sample = {} 
            flame_sample['gt_shape'] = torch.tensor(shape).unsqueeze(0)
            flame_sample['gt_exp'] = torch.tensor(sample[rec_type]["gt_exp"]).unsqueeze(0)
            flame_sample['gt_jaw'] = torch.tensor(sample[rec_type]["gt_jaw"]).unsqueeze(0)
            flame_sample = self.flame(flame_sample, "")
            # for key in flame_sample.keys():
            #     sample[key] = torch.zeros_like(flame_sample[key]).numpy()[0]
            for key in flame_sample.keys():
                # if key not in sample.keys():
                #     sample[key] = {}
                sample["reconstruction"][rec_type][key] = flame_sample[key].detach().clone().contiguous().numpy()[0]
        
        return sample

    def _get_emotions(self, index, start_frame, num_read_frames, video_fps, num_frames, sample):
        sequence_length = self._get_sample_length(index)
        if self.emotion_type is None:
            return sample
        if not self.preload_videos:
            emotions, features = self._load_emotions(index, self.return_emotion_feature, 
                                                     start_frame=start_frame, end_frame=start_frame+num_read_frames)
        else: 
            emotions = self.emo_cache[index]["emotions"].copy() 
            emotions = emotions[start_frame:start_frame+num_read_frames]
            features = self.emo_cache[index]["features"]
            if features is not None:
                features = features.copy()
                features = features[start_frame:start_frame+num_read_frames]
        if features is not None:
            features = {"emo_" + key: features[key] for key in features.keys()} # just add the emo_ prefix to the keys
        for key in emotions.keys():
            assert key not in sample.keys(), f"Key {key} already exists in sample."
            sample['gt_' + key] = emotions[key][0].astype(np.float32)
            
            # if shorter than the sequence, pad with zeros
            if sample['gt_' + key].shape[0] < sequence_length:
                sample['gt_' + key] = np.concatenate([sample['gt_' + key], np.zeros((sequence_length - sample['gt_' + key].shape[0], sample['gt_' + key].shape[1]))], axis=0).astype(np.float32)

        
        if features is not None:
            for key in features.keys():
                assert key not in sample.keys(), f"Key {key} already exists in sample."
                sample['gt_' + key] = features[key][0].astype(np.float32)

                # if shorter than the sequence, pad with zeros
                if sample['gt_' + key].shape[0] < sequence_length:
                    sample['gt_' + key] = np.concatenate([sample['gt_' + key], np.zeros((sequence_length - sample['gt_' + key].shape[0], sample['gt_' + key].shape[1]))], axis=0).astype(np.float32)

        return sample


    def _augment(self, img, seg_image, landmark, input_img_shape=None):
        # workaround to make sure each sequence is augmented the same
        # unfortunately imgaug does not support this out of the box 

        # therefore we split the [B, T, ...] arays into T x [B, ...] arrays 
        # and augment each t from 1 to T separately same way using to_deterministic

        transform_det = self.transforms.to_deterministic()

        T = img.shape[0]

        for t in range(T):
            img_t = img[t, ...] if img is not None else None
            seg_image_t = seg_image[t:t+1, ..., np.newaxis] if seg_image is not None else None
            landmark_t = landmark[t:t+1, ..., :2] if landmark is not None else None

            if self.transforms is not None:
                res = transform_det(
                    # image=img_t.astype(np.float32),
                    image=img_t,
                    segmentation_maps=seg_image_t,
                    keypoints=landmark_t)
                if seg_image_t is not None and landmark is not None:
                    img_t, seg_image_t, landmark_t = res
                elif seg_image_t is not None:
                    img_t, seg_image_t, _ = res
                elif landmark_t is not None:
                    img_t, _, landmark_t = res
                else:
                    img_t = res

            if seg_image_t is not None:
                seg_image_t = np.squeeze(seg_image_t)[..., np.newaxis].astype(np.float32)

            if landmark_t is not None:
                landmark_t = np.squeeze(landmark_t)
                if isinstance(self.landmark_normalizer, KeypointScale):
                    self.landmark_normalizer.set_scale(
                        img_t.shape[0] / input_img_shape[0],
                        img_t.shape[1] / input_img_shape[1])
                elif isinstance(self.landmark_normalizer, KeypointNormalization):
                    self.landmark_normalizer.set_scale(img_t.shape[0], img_t.shape[1])
                    # self.landmark_normalizer.set_scale(input_img_shape[0], input_img_shape[1])
                else:
                    raise ValueError(f"Unsupported landmark normalizer type: {type(self.landmark_normalizer)}")
                landmark_t = self.landmark_normalizer(landmark_t)

            img[t:t+1, ...] = img_t 
            if seg_image is not None:
                seg_image[t:t+1, ...] = seg_image_t[..., 0]
            if landmark is not None:
                landmark[t:t+1, ..., :2] = landmark_t[np.newaxis]
        if landmark is not None:
            landmark = landmark[:, :, :2]
        return img, seg_image, landmark

    def _augment_sequence_sample(self, index, sample):
        # get the mediapipe landmarks 
        mediapipe_landmarks = sample["landmarks"]["mediapipe"]
        mediapipe_landmarks_valid = sample["landmarks_validity"]["mediapipe"]
        # mediapipe_landmarks = []
        images = sample["video"]
        segmentation=None
        if "segmentation" in sample.keys():
            segmentation = sample["segmentation"]
        

        images_masked = np.copy(images)
        segmentation_masked = None
        if segmentation is not None:
            segmentation_masked = np.copy(segmentation)

        masked_frames = np.zeros(images.shape[:1], dtype=np.float32)

        # compute mouth region bounding box
        if np.random.rand() < self.occlusion_probability_mouth: 
            images_masked, segmentation_masked, start_frame_, end_frame_ = self._occlude_sequence(index, images_masked, segmentation_masked, 
                mediapipe_landmarks, mediapipe_landmarks_valid, "mouth")
            masked_frames[start_frame_:end_frame_] = 1.0

        # compute eye region bounding box
        if np.random.rand() < self.occlusion_probability_left_eye:
            images_masked, segmentation_masked, start_frame_, end_frame_ = self._occlude_sequence(index, images_masked, segmentation_masked, 
                mediapipe_landmarks, mediapipe_landmarks_valid, "left_eye")
            masked_frames[start_frame_:end_frame_] = 1.0

        if np.random.rand() < self.occlusion_probability_right_eye:
            images_masked, segmentation_masked, start_frame_, end_frame_ = self._occlude_sequence(index, images_masked, segmentation_masked, 
                mediapipe_landmarks, mediapipe_landmarks_valid, "right_eye")
            masked_frames[start_frame_:end_frame_] = 1.0

        # compute face region bounding box
        if np.random.rand() < self.occlusion_probability_face:
            images_masked, segmentation_masked, start_frame_, end_frame_ = self._occlude_sequence(index, images_masked, segmentation_masked, 
                mediapipe_landmarks, mediapipe_landmarks_valid, "all")
            masked_frames[start_frame_:end_frame_] = 1.0



        #  augment the sequence
        # images, segmentation, mediapipe_landmarks = self._augment(images, segmentation, 
        #             mediapipe_landmarks, images.shape[2:])
        # images_masked, segmentation_masked, _ = self._augment(images_masked, segmentation_masked, 
        #             None, images.shape[2:])


        images_aug = (np.concatenate([images, images_masked], axis=0) * 255.0).astype(np.uint8)
        segmentation_aug = [] 
        if segmentation is not None:
            segmentation_aug += [segmentation]
        if segmentation_masked is not None:
            segmentation_aug += [segmentation_masked]

        segmentation_aug = np.concatenate(segmentation_aug, axis=0) if len(segmentation_aug) > 0 else None
        # segmentation_aug = np.concatenate([segmentation, segmentation_masked], axis=0)
        # mediapipe_landmarks_aug = np.concatenate([mediapipe_landmarks, mediapipe_landmarks], axis=0)

        # concatenate all the available landmarks s.t. it can be used for augmentation
        landmarks_to_augment = [] 
        lmk_counts = []
        for lmk_type in sample["landmarks"].keys():
            landmarks_to_augment += [sample["landmarks"][lmk_type]]
            lmk_counts += [sample["landmarks"][lmk_type].shape[1]]
        lmk_counts = np.array(lmk_counts, dtype=np.int32)

        landmarks_to_augment = np.concatenate(landmarks_to_augment, axis=1)
        landmarks_to_augment_aug = np.concatenate([landmarks_to_augment, landmarks_to_augment], axis=0)

        # images_aug, segmentation_aug, mediapipe_landmarks_aug = self._augment(images_aug, segmentation_aug, 
        #                     mediapipe_landmarks_aug, images.shape[2:])
        images_aug, segmentation_aug, mediapipe_landmarks_aug = self._augment(images_aug, segmentation_aug, 
                    landmarks_to_augment_aug, images.shape[2:])
        images_aug = images_aug.astype(np.float32) / 255.0 # back to float
        images = images_aug[:images_aug.shape[0]//2]

        if segmentation is not None:
            segmentation = segmentation_aug[:segmentation_aug.shape[0]//2]
        # mediapipe_landmarks = mediapipe_landmarks_aug[:mediapipe_landmarks_aug.shape[0]//2]
        landmarks_to_augment_aug = landmarks_to_augment_aug[:landmarks_to_augment_aug.shape[0]//2]
        images_masked = images_aug[images_aug.shape[0]//2 :]

        if segmentation_masked is not None:
            segmentation_masked = segmentation_aug[segmentation_aug.shape[0]//2 :]

        # sample["video"] = images / 255.0
        # sample["video_masked"] = images_masked / 255.0
        sample["video"] = images 
        sample["video_masked"] = images_masked 
        if segmentation is not None:
            sample["segmentation"] = segmentation
        if segmentation_masked is not None:
            sample["segmentation_masked"] = segmentation_masked
        sample["masked_frames"] = masked_frames
        # sample["landmarks"]["mediapipe"] = mediapipe_landmarks 

        # split the augmented landmarks back to their original types
        for i, lmk_type in enumerate(sample["landmarks"].keys()):
            sample["landmarks"][lmk_type] = landmarks_to_augment_aug[:, np.sum(lmk_counts[:i]):np.sum(lmk_counts[:i+1]), :].astype(np.float32)
        return sample

    def _occlude_sequence(self, index, images, segmentation, mediapipe_landmarks, mediapipe_landmarks_valid, region):
        bounding_boxes, sizes = self.occluder.bounding_box_batch(mediapipe_landmarks, region)
        
        sequence_length = self._get_sample_length(index)

        bb_style = "max" # largest bb of the sequence 
        # bb_style = "min" # smallest bb of the sequence 
        # bb_style = "mean" # mean bb of the sequence 
        # bb_style = "original" # original bb of the sequence (different for each frame) 

        # we can enlarge or shrink the bounding box
        scale_size_width = 1 
        scale_size_height = 1 

        scaled_sizes = np.copy(sizes)
        scaled_sizes[:,2] = (scaled_sizes[:,2] * scale_size_width).astype(np.int32)
        scaled_sizes[:,3] = (scaled_sizes[:,3] * scale_size_height).astype(np.int32)


        if bb_style == "max":
            width = sizes[:,2].max()
            height = sizes[:,3].max()
            sizes[:, 2] = width
            sizes[:, 3] = height
            bounding_boxes = sizes_to_bb_batch(sizes)
        elif bb_style == "mean":
            width = sizes[:,2].mean()
            height = sizes[:,3].mean()
            sizes[:, 2] = width
            sizes[:, 3] = height
            bounding_boxes = sizes_to_bb_batch(sizes)
        elif bb_style == "min":
            width = sizes[:,2].min()
            height = sizes[:,3].min()
            sizes[:, 2] = width
            sizes[:, 3] = height
            bounding_boxes = sizes_to_bb_batch(sizes)
        elif bb_style == "original":
            pass 
        else:
            raise ValueError(f"Unsupported bounding box strategy {bb_style}")

        bounding_boxes = bounding_boxes.clip(min=0, max=images.shape[1] - 1)

        occlusion_length = np.random.randint(self.occlusion_length[0], self.occlusion_length[1])
        occlusion_length = min(sequence_length, occlusion_length)

        start_frame = np.random.randint(0, max(sequence_length - occlusion_length + 1, 1))
        end_frame = start_frame + occlusion_length

        images = self.occluder.occlude_batch(images, region, landmarks=None, 
            bounding_box_batch=bounding_boxes, start_frame=start_frame, end_frame=end_frame)
        if segmentation is not None:
            segmentation = self.occluder.occlude_batch(segmentation, region, landmarks=None, 
                bounding_box_batch=bounding_boxes, start_frame=start_frame, end_frame=end_frame)
        return images, segmentation, start_frame, end_frame

    def add_noise(self, wavdata):
        raise NotImplementedError(  )
        noise = np.random.randn(len(wavdata))
        return wavdata + noise

    def _true_len(self):
        return len(self.video_indices)

    def __len__(self): 
        if self.hack_length: 
            return int(self.hack_length*self._true_len())
        return self._true_len()

    def visualize_sample(self, sample_or_index):
        from inferno.utils.MediaPipeLandmarkDetector import np2mediapipe
        
        if isinstance(sample_or_index, (int, np.int32, np.int64)):
            index = sample_or_index
            sample = self[index]
        else:
            sample = sample_or_index

        # visualize the video
        video_frames = sample["video"]
        segmentation = sample.get("segmentation", None)
        video_frames_masked = sample.get("video_masked", None)
        segmentation_masked = sample.get("segmentation_masked", None)

        if "mediapipe" in sample["landmarks"]:
            landmarks_mp = sample["landmarks"]["mediapipe"]

            landmarks_mp = self.landmark_normalizer.inv(landmarks_mp)

            # T, C, W, H to T, W, H, C 
            video_frames = video_frames.permute(0, 2, 3, 1)
            if video_frames_masked is not None:
                video_frames_masked = video_frames_masked.permute(0, 2, 3, 1)
            if segmentation is not None:
                segmentation = segmentation[..., None]
            
            if segmentation_masked is not None:
                segmentation_masked = segmentation_masked[..., None]

            # plot the video frames with plotly
            # horizontally concatenate the frames
            frames = np.concatenate(video_frames.numpy(), axis=1)
            frames_masked=None
            if video_frames_masked is not None:
                frames_masked = np.concatenate(video_frames_masked.numpy(), axis=1)
            if segmentation is not None:
                segmentation = np.concatenate(segmentation.numpy(), axis=1)
            if segmentation_masked is not None:
                segmentation_masked = np.concatenate(segmentation_masked.numpy(), axis=1)
            landmarks_mp_list = [] 
            for i in range(landmarks_mp.shape[0]):
                landmarks_mp_proto = np2mediapipe(landmarks_mp[i].numpy() / self.image_size)
                landmarks_mp_list.append(landmarks_mp_proto)

            # Load drawing_utils and drawing_styles
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils 
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_face_mesh = mp.solutions.face_mesh

            video_frames_landmarks_mp = np.copy(video_frames)*255
            for i in range(video_frames_landmarks_mp.shape[0]):
                mp_drawing.draw_landmarks(
                    image=video_frames_landmarks_mp[i],
                    landmark_list=landmarks_mp_list[i],
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style()
                    )
            
            video_frames_landmarks_mp = np.concatenate(video_frames_landmarks_mp, axis=1)
        else: 
            video_frames_landmarks_mp = None

        fan_lmk_images = None
        if "fan" in sample["landmarks"]:
            from inferno.utils.DecaUtils import tensor_vis_landmarks
            landmarks_fan = sample["landmarks"]["fan"]
            fan_lmk_images = tensor_vis_landmarks(sample["video"],
                            self.landmark_normalizer.inv(landmarks_fan),
                            isScale=False, rgb2bgr=False, scale_colors=True).permute(0, 2, 3, 1).numpy().tolist()
            fan_lmk_images = np.concatenate(fan_lmk_images, axis=1)
            # for i in range(landmarks_fan.shape[0]):
            #     lmk = landmarks_fan[i, ...]
            #     lmk_expanded = lmk[np.newaxis, ...]
            #     lmk_im = tensor_vis_landmarks(video_frames,
            #                 self.landmark_normalizer.inv(lmk_expanded),
            #                 isScale=False, rgb2bgr=False, scale_colors=True).numpy()[0].transpose([1, 2, 0])
            #     fan_lmk_images += [lmk_im]
            # fan_lmk_images = np.concatenate(fan_lmk_images, axis=1)   

        all_images = [frames*255] 
        if segmentation is not None:
            all_images += [np.tile(segmentation*255, (1,1,3))] 
        if frames_masked is not None: 
            all_images += [frames_masked*255] 
        if segmentation_masked is not None:
            all_images += [np.tile(segmentation_masked*255, (1,1,3))] 
        if video_frames_landmarks_mp is not None:
            all_images += [video_frames_landmarks_mp]
        if fan_lmk_images is not None: 
            all_images += [fan_lmk_images * 255]
        all_images = np.concatenate(all_images, axis=0)
        # plot the frames

        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        fig = go.Figure(data=[go.Image(z=all_images)])
        # show the figure 
        fig.show()

        # fig = go.Figure(data=[go.Image(z=frames)])
        # # show the figure 
        # fig.show()
        # fig = go.Figure(data=[go.Image(z=frames_masked)])
        # # show the figure 
        # fig.show()

        # fig = go.Figure(data=[go.Image(z=segmentation)])
        # # show the figure 
        # fig.show()

        # fig = go.Figure(data=[go.Image(z=segmentation_masked)])
        # # show the figure 
        # fig.show()
        # print("ha")


def to_torch(what):
    if isinstance(what, np.ndarray):
        return torch.from_numpy(what)
    elif isinstance(what, list):
        return [to_torch(x) for x in what]
    elif isinstance(what, dict):
        return {k: to_torch(v) for k, v in what.items()}
    else:
        return what


def stacker(feats, stack_order):
    """
    Concatenating consecutive audio frames
    Args:
    feats - numpy.ndarray of shape [T, F]
    stack_order - int (number of neighboring frames to concatenate
    Returns:
    feats - numpy.ndarray of shape [T', F']
    """
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        # pad the end with zeros
        res = stack_order - len(feats) % stack_order
        res = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res], axis=0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
    return feats


class VideoDatasetBaseV2(VideoDatasetBase): 

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
    
    def _read_landmarks(self, index, landmark_type, landmark_source):
        landmarks_dir = self._path_to_landmarks(index, landmark_type, landmark_source)
        if landmark_source == "original":
            landmark_list_file = landmarks_dir / f"landmarks_aligned_video_smoothed.pkl"
            landmark_list = FaceDataModuleBase.load_landmark_list(landmark_list_file)  
            landmark_valid_indices = FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmarks_alignment_used_frame_indices.pkl")  
        elif landmark_source == "aligned": 
            landmark_list, landmark_confidences, landmark_types = FaceDataModuleBase.load_landmark_list_v2(landmarks_dir / f"landmarks.pkl")  
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
                original_image_size =  self.original_image_size
                # landmarks = landmarks * self.image_size # do not use this one, this is the desired size (the video will be resized to this one)
                landmarks = landmarks * original_image_size # use the original size

                landmarks = landmarks[start_frame: sequence_length + start_frame]
                # landmark_confidences = landmark_confidences[start_frame: sequence_length + start_frame]
                # landmark_validity = landmark_confidences #TODO: something is wrong here, the validity is not correct and has different dimensions
                # landmark_validity = None # this line craashes the code if FAN landmarks used (sometimes they are missing)
                # a potentially dangerous hack (we don't know how valid the landmakrs are but MEAD is an easy dataset so it should be OK)
                landmark_validity = np.ones((len(landmarks), 1), dtype=np.float32) 
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