import torch
import numpy as np 
import imgaug
from gdl.transforms.keypoints import KeypointNormalization, KeypointScale
from gdl.utils.MediaPipeFaceOccluder import MediaPipeFaceOccluder, sizes_to_bb_batch
from pathlib import Path
from scipy.io import wavfile 
from python_speech_features import logfbank
from gdl.datasets.FaceDataModuleBase import FaceDataModuleBase
from gdl.datasets.IO import load_and_process_segmentation, process_segmentation, load_segmentation, load_segmentation_list
from gdl.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp
from gdl.utils.FaceDetector import load_landmark
import pandas as pd
from skvideo.io import vread, vreader, FFmpegReader
import torch.nn.functional as F
import subprocess 
import traceback
from gdl.layers.losses.MediaPipeLandmarkLosses import MEDIAPIPE_LANDMARK_NUMBER


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
            image_size=None, 
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
        ) -> None:
        super().__init__()
        self.root_path = root_path
        self.output_dir = output_dir
        self.video_list = video_list
        self.video_indices = video_indices
        self.video_metas = video_metas
        self.sequence_length = sequence_length
        # self.audio_paths = audio_paths
        self.audio_metas = audio_metas
        self.audio_noise_prob = audio_noise_prob
        self.image_size = image_size
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
        self.align_images = True
        # self.align_images = False
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

        self.video_cache = {}
        self.seg_cache = {}
        if self.preload_videos:
            self._preload_videos()
            
        self.video_sample_indices = None
        if self.inflate_by_video_size: 
            self._inflate_by_video_size()

        self.include_filename = include_filename

    def _preload_videos(self): 
        # indices = np.unique(self.video_indices)
        for i in range(len(self.video_indices)):
            video_path = str(self._get_video_path(i))
            if video_path not in self.video_cache:
                self.video_cache[video_path] = vread(video_path)
                self.seg_cache[video_path] = self._read_segmentations(i)
    

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
                self.num_total_failed_attempts += 1
                print(f"[ERROR] AssertionError in {self.__class__.__name__} dataset while retrieving sample {old_index}, retrying with new index {index}")
                print(f"In total, there has been {self.num_total_failed_attempts} failed attempts. This number should be very small. If it's not, check the data.")
                print("See the exception message for more details.")
                print(tb)
        print("[ERROR] Failed to retrieve sample after {} attempts".format(max_attempts))
        raise RuntimeError("Failed to retrieve sample after {} attempts".format(max_attempts))

    def _getitem(self, index):
        if self.hack_length: 
            index = index % self._true_len()

        # 1) VIDEO
        # load the video 
        sample, start_frame, num_read_frames, video_fps, num_frames, num_available_frames = self._get_video(index)

        # 2) AUDIO
        sample = self._get_audio(index, start_frame, num_read_frames, video_fps, num_frames, sample)

        # 3) LANDMARKS 
        sample = self._get_landmarks(index, start_frame, num_read_frames, video_fps, num_frames, sample)

        # 4) SEGMENTATIONS
        sample = self._get_segmentations(index, start_frame, num_read_frames, video_fps, num_frames, sample)

        # 5) FACE ALIGNMENT IF ANY
        sample = self._align_faces(index, sample)

        # 6) AUGMENTATION
        sample = self._augment_sequence_sample(index, sample)

        # TO TORCH
        sample = to_torch(sample)


        # AUDIO NORMALIZATION (if any)
        if self.include_processed_audio:
            if self.audio_normalization is not None:
                if self.audio_normalization == "layer_norm":
                    sample["audio"] = F.layer_norm(sample["audio"], sample["audio"].shape[1:])
                else: 
                    raise ValueError(f"Unsupported audio normalization {self.audio_normalization}")

        # T,H,W,C to T,C,H,W
        sample["video"] = sample["video"].permute(0, 3, 1, 2)
        if "video_masked" in sample.keys():
            sample["video_masked"] = sample["video_masked"].permute(0, 3, 1, 2)
        # sample["segmenation"] = sample["segmenation"].permute(0, 2, 1)
        # sample["segmentation_masked"] = sample["segmentation_masked"].permute(0, 2, 1)


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

        return sample

    def _get_video_path(self, index):
        if self.use_original_video:
            video_path = self.root_path / self.video_list[self.video_indices[index]]
        else: 
            video_path = Path(self.output_dir) / "videos_aligned" / self.video_list[self.video_indices[index]]
        return video_path

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
        num_read_frames = 0
        try:
            if not self.preload_videos:
                frames = vread(video_path.as_posix())
            else: 
                frames = self.video_cache[video_path.as_posix()]
            assert len(frames) == num_frames, f"Video {video_path} has {len(frames)} frames, but meta says it has {num_frames}"
            frames = frames[start_frame:(start_frame + sequence_length)] 
            num_read_frames = frames.shape[0]
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

        sample = { 
            "video": frames,
            "frame_indices": np.arange(start_frame, start_frame + sequence_length, dtype=np.int32),
        }


        if self.include_filename: 
            sample["filename"] = str(video_path)
            sample["fps"] = video_fps # include the fps in the sample


        return sample, start_frame, num_read_frames, video_fps, num_frames, num_available_frames

    def _get_audio(self, index, start_frame, num_read_frames, video_fps, num_frames, sample):
        audio_path = (Path(self.output_dir) / "audio" / self.video_list[self.video_indices[index]]).with_suffix(".wav")
        audio_meta = self.audio_metas[self.video_indices[index]]
            
        sequence_length = self._get_sample_length(index)
        # load the audio 
        # if self.include_raw_audio:
        import librosa
        sampling_rate = 16000
        wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
        # wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
        if wavdata.ndim > 1:
            wavdata = librosa.to_mono(wavdata)
        wavdata = (wavdata.astype(np.float64) * 32768.0).astype(np.int16)

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


    def _get_landmarks(self, index, start_frame, num_read_frames, video_fps, num_frames, sample): 
        sequence_length = self._get_sample_length(index)
        landmark_dict = {}
        landmark_validity_dict = {}
        for lti, landmark_type in enumerate(self.landmark_types):
            landmark_source = self.landmark_source[lti]
            landmarks_dir = (Path(self.output_dir) / f"landmarks_{landmark_source}" / landmark_type /  self.video_list[self.video_indices[index]]).with_suffix("")
            landmarks = []
            if (landmarks_dir / "landmarks.pkl").exists(): # landmarks are saved per video in a single file
            #    landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmarks.pkl")  
            #    landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmarks_original.pkl")  
                landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / f"landmarks_{landmark_source}.pkl")  
                landmark_types =  FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmark_types.pkl")  
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

            landmark_dict[landmark_type] = landmarks
            landmark_validity_dict[landmark_type] = landmark_validity


        sample["landmarks"] = landmark_dict
        sample["landmarks_validity"] = landmark_validity_dict
        return sample

    def _path_to_segmentations(self, index): 
        return (Path(self.output_dir) / f"segmentations_{self.segmentation_source}" / self.segmentation_type /  self.video_list[self.video_indices[index]]).with_suffix("")

    def _read_segmentations(self, index):
        segmentations_dir = self._path_to_segmentations(index)
        seg_images, seg_types, seg_names = load_segmentation_list(segmentations_dir / "segmentations.pkl")
        segmentations = np.stack(seg_images, axis=0)[:,0,...]
        return segmentations, seg_types, seg_names

    def _retrieve_segmentations(self, index):
        if not self.preload_videos:
            segmentations_dir = self._path_to_segmentations(index)
            seg_images, seg_types, seg_names = load_segmentation_list(segmentations_dir / "segmentations.pkl")
            segmentations = np.stack(seg_images, axis=0)[:,0,...]
            return segmentations, seg_types, seg_names
        else:
            video_path = str(self._get_video_path(index))
            return self.seg_cache[video_path]


    def _get_segmentations(self, index, start_frame, num_read_frames, video_fps, num_frames, sample): 
        segmentations_dir = self._path_to_segmentations(index)
        segmentations = []

        sequence_length = self._get_sample_length(index)

        if (segmentations_dir / "segmentations.pkl").exists(): # segmentations are saved in a single file-per video 
            # seg_images, seg_types, seg_names = load_segmentation_list(segmentations_dir / "segmentations.pkl")
            # segmentations = seg_images[start_frame: sequence_length + start_frame]
            # segmentations = np.stack(segmentations, axis=0)[:,0,...]
            segmentations, seg_types, seg_names  = self._retrieve_segmentations(index)
            segmentations = segmentations[start_frame: sequence_length + start_frame]
            segmentations = process_segmentation(segmentations, seg_types[0]).astype(np.uint8)
            # assert segmentations.shape[0] == sequence_length
        else: # segmentations are saved per-frame
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

        sample["segmentation"] = segmentations
        return sample


    def _align_faces(self, index, sample):
        if self.align_images:
            
            sequence_length = self._get_sample_length(index)

            landmarks_for_alignment = "mediapipe"
            left = sample["landmarks"][landmarks_for_alignment][:,:,0].min(axis=1)
            top =  sample["landmarks"][landmarks_for_alignment][:,:,1].min(axis=1)
            right =  sample["landmarks"][landmarks_for_alignment][:,:,0].max(axis=1)
            bottom = sample["landmarks"][landmarks_for_alignment][:,:,1].max(axis=1)

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

            old_size, center = bbox2point(left, right, top, bottom, type=landmarks_for_alignment)
            size = (old_size * self.scale).astype(np.int32)

            video_frames = sample["video"]
            sample["video"] = np.zeros((video_frames.shape[0], self.image_size, self.image_size, video_frames.shape[-1]), dtype=video_frames.dtype)
            
            if "segmentation" in sample.keys():
                seg_frames = sample["segmentation"]
                sample["segmentation"] = np.zeros((seg_frames.shape[0], self.image_size, self.image_size), dtype=seg_frames.dtype)

            for i in range(sequence_length):
                lmk_to_warp = {k: v[i] for k,v in sample["landmarks"].items()}
                img_warped, lmk_warped = bbpoint_warp(video_frames[i], center[i], size[i], self.image_size, landmarks=lmk_to_warp)
                
                if "segmentation" in sample.keys():
                    seg_warped = bbpoint_warp(seg_frames[i], center[i], size[i], self.image_size, 
                        order=0 # nearest neighbor interpolation for segmentation
                        )
                # img_warped *= 255.
                assert np.isnan(img_warped).sum() == 0, f"NaNs in image {i} after face aligning image warp." \
                    f"Center: {center[i]}, size: {size[i]}. Are these values valid?"
                sample["video"][i] = img_warped 
                # sample["segmentation"][i] = seg_warped * 255.
                if "segmentation" in sample.keys():
                    sample["segmentation"][i] = seg_warped
                for k,v in lmk_warped.items():
                    sample["landmarks"][k][i][:,:2] = v
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
                res = transform_det(image=img_t.astype(np.float32),
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


        images_aug = np.concatenate([images, images_masked], axis=0) * 255.0
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
        images = images_aug[:images_aug.shape[0]//2]

        if segmentation is not None:
            segmentation = segmentation_aug[:segmentation_aug.shape[0]//2]
        # mediapipe_landmarks = mediapipe_landmarks_aug[:mediapipe_landmarks_aug.shape[0]//2]
        landmarks_to_augment_aug = landmarks_to_augment_aug[:landmarks_to_augment_aug.shape[0]//2]
        images_masked = images_aug[images_aug.shape[0]//2 :]

        if segmentation_masked is not None:
            segmentation_masked = segmentation_aug[segmentation_aug.shape[0]//2 :]

        sample["video"] = images / 255.0
        sample["video_masked"] = images_masked / 255.0
        if segmentation is not None:
            sample["segmentation"] = segmentation
        if segmentation_masked is not None:
            sample["segmentation_masked"] = segmentation_masked
        sample["masked_frames"] = masked_frames
        # sample["landmarks"]["mediapipe"] = mediapipe_landmarks 

        # split the augmented landmarks back to their original types
        for i, lmk_type in enumerate(sample["landmarks"].keys()):
            sample["landmarks"][lmk_type] = landmarks_to_augment_aug[:, np.sum(lmk_counts[:i]):np.sum(lmk_counts[:i+1]), :]
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
        from gdl.utils.MediaPipeLandmarkDetector import np2mediapipe
        
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
            from gdl.utils.DecaUtils import tensor_vis_landmarks
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
