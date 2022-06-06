from cv2 import imread
import torchaudio
from gdl.datasets.FaceDataModuleBase import FaceDataModuleBase
from gdl.datasets.FaceVideoDataModule import FaceVideoDataModule 
from pathlib import Path
import torch
import torch.nn.functional as F
import os, sys
from gdl.utils.FaceDetector import load_landmark
from gdl.utils.MediaPipeLandmarkDetector import np2mediapipe
from gdl.utils.other import get_path_to_externals
from gdl.utils.MediaPipeFaceOccluder import MediaPipeFaceOccluder, sizes_to_bb, sizes_to_bb_batch
import numpy as np
from skvideo.io import vread, vreader
from scipy.io import wavfile
import time
from python_speech_features import logfbank
from gdl.datasets.IO import load_segmentation, process_segmentation, load_segmentation_list
from gdl.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp


class LRS3DataModule(FaceVideoDataModule):

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
                device=None):
        super().__init__(root_dir, output_dir, processed_subfolder, 
            face_detector, face_detector_threshold, image_size, scale, device, 
            unpack_videos=False, save_detection_images=False, 
            # save_landmarks=True,
            save_landmarks=False, # trying out this option
            save_landmarks_one_file=True, 
            save_segmentation_frame_by_frame=False, 
            save_segmentation_one_file=True,
            )
        self.detect_landmarks_on_restored_images = landmarks_from
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.sequence_length_train = sequence_length_train
        self.sequence_length_val = sequence_length_val
        self.sequence_length_test = sequence_length_test

        self.split = split
        self.num_workers = num_workers
        self.drop_last = True

        # self.occlusion_length_train = occlusion_length_train
        # self.occlusion_length_val = occlusion_length_val
        # self.occlusion_length_test = occlusion_length_test
        self.occlusion_settings_train = occlusion_settings_train or {}
        self.occlusion_settings_val = occlusion_settings_val or {}
        self.occlusion_settings_test = occlusion_settings_test or {}

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

    def _get_landmark_method(self):
        return self.face_detector_type

    def _get_segmentation_method(self):
        return "bisenet"

    def _detect_faces(self):
        return super()._detect_faces( )


    def _gather_data(self, exist_ok=True):
        super()._gather_data(exist_ok)
        
        vl = [(path.parent / path.stem).as_posix() for path in self.video_list]
        al = [(path.parent / path.stem).as_posix() for path in self.annotation_list]

        vl_set = set(vl)
        al_set = set(al)

        vl_diff = vl_set.difference(al_set)
        al_diff = al_set.difference(vl_set)

        intersection = vl_set.intersection(al_set) 

        print(f"Video list: {len(vl_diff)}")
        print(f"Annotation list: {len(al_diff)}")

        if len(vl_diff) != 0:
            print("Video list is not equal to annotation list") 
            print("Video list difference:")
            print(vl_diff)
            raise RuntimeError("Video list is not equal to annotation list")
        
        if len(al_diff) != 0: 
            print("Annotation list is not equal to video list")    
            print("Annotation list difference:")
            print(al_diff)
            raise RuntimeError("Annotation list is not equal to video list")

        print(f"Intersection: {len(intersection)}")
    
    def _video_supername(self, sequence_id):
        video_file = self.video_list[sequence_id]
        return video_file.parents[0].name

    def _video_set(self, sequence_id):
        video_file = self.video_list[sequence_id]
        out_folder = video_file.parts[-3]
        return out_folder

    def _get_path_to_sequence_files(self, sequence_id, file_type, method="", suffix=""): 
        assert file_type in ['videos', 'detections', "landmarks", "landmarks_original", "segmentations", "segmentations_original",
            "emotions", "reconstructions",  "audio", "videos_restored" ]
        video_file = self.video_list[sequence_id]
        if len(method) > 0:
            # file_type += "_" + method 
            file_type += "/" + method 
        if len(suffix) > 0:
            file_type += suffix

        suffix = Path(file_type) / self._video_set(sequence_id) / self._video_supername(sequence_id) / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder


    def _get_path_to_sequence_restored(self, sequence_id, method="", suffix=""):
        return self._get_path_to_sequence_files(sequence_id, "videos_restored", method, suffix).with_suffix(".mp4")


    def _get_restoration_network(self, method):
        # try:
        from gdl.models.external.GPENFaceRestoration import GPENFaceRestoration
        # except ImportError: 
        #     print("Could not import GPENFaceRestoration. Skipping.") 
        return GPENFaceRestoration(method)

    def _get_jpeg_network(self):
        # try:
        from gdl.models.external.SwinIRTranslation import SwinIRCompressionArtifact
        # except ImportError: 
        #     print("Could not import SwinIRTranslation. Skipping.") 
        return SwinIRCompressionArtifact( 256)

    def _get_superres_network(self, method="swin_ir"):
        # try:
        if method == "swin_ir":
            from gdl.models.external.SwinIRTranslation import SwinIRRealSuperRes
            # except ImportError: 
                # print("Could not import SwinIRTranslation. Skipping.") 
            return SwinIRRealSuperRes( 256)
        elif method == "bsrgan":
            from gdl.models.external.BSRGANSuperRes import BSRSuperRes
            # return BSRSuperRes( 256, 4)
            return BSRSuperRes( 256, 2)
        raise ValueError(f"Unknown super-resolution method: {method}")

        # if method == "GPEN-512": 
        #     im_size = 512
        #     model_name = "GPEN-BFR-512"
        # elif method == 'GPEN-256': 
        #     im_size = 256
        #     model_name = "GPEN-BFR-256"
        # else: 
        #     raise NotImplementedError()

        # path_to_gpen = get_path_to_externals() / ".." / ".." / "GPEN" / "face_model"

        # if str(path_to_gpen) not in sys.path:
        #     sys.path.insert(0, str(path_to_gpen))

        # from face_gan import FaceGAN
        # network = FaceGAN(str(path_to_gpen / "..") , im_size, model=model_name, channel_multiplier=2, narrow=1, key=None, device='cuda') 
        # return network


    def _deep_restore_sequence(self, sequence_id, nets, input_videos = None, output_subfolder = None, 
            batch_size=16, resize_to_original=True):
        from skvideo.io import vread, vwrite

        video_file = self.root_dir / self.video_list[sequence_id]
        restored_video_file = self._get_path_to_sequence_restored(sequence_id, method=output_subfolder)

        if input_videos is not None:
            # video_file = Path(str(restored_video_file).replace('videos_restored', input_videos))
            video_file = self._get_path_to_sequence_restored(sequence_id, method=input_videos)

        if output_subfolder is not None:
            # restored_video_file = Path(str(restored_video_file).replace('videos_restored', output_subfolder))
            restored_video_file = self._get_path_to_sequence_restored(sequence_id, method=output_subfolder)


        # restored_video_file = Path(str(restored_video_file).replace('videos_restored', 'videos_restored_jpg_sr'))
        # restored_video_file = Path(str(restored_video_file).replace('videos_restored', 'videos_restored_sr'))
        # restored_video_file = Path(str(restored_video_file).replace('videos_restored', 'videos_restored_jpg_sr_res'))
        # restored_video_file = Path(str(restored_video_file).replace('videos_restored', 'videos_restored_sr_res'))

        # batch_size = 16 
        # batch_size = 12
        # batch_size = 6 

        restored_video_file.parent.mkdir(parents=True, exist_ok=True)

        images =  vread(str(video_file))
        # restored_images = np.zeros_like(images)
        restored_images = None

        start_i = 0 
        end_i = min(start_i + batch_size, images.shape[0])

        # # jpeg_net = self._get_jpeg_network()
        # superres_net = self._get_superres_network()

        # restoration_net = self._get_restoration_network("GPEN-512")

        while start_i < images.shape[0]:
            print(f"Restoring {start_i} to {end_i}")
            images_torch = torch.from_numpy(images[start_i:end_i]).float() / 255.  
            # transpose into torch format 
            images_torch = images_torch.permute(0, 3, 1, 2).cuda()

            with torch.no_grad():
                # time_start = time.time()
                # # restored_images_torch = jpeg_net(images_torch, resize_to_input_size=False)
                # time_jpeg = time.time() 
                # restored_images_torch = superres_net(images_torch, resize_to_input_size=False)
                # # restored_images_torch = superres_net(restored_images_torch, resize_to_input_size=False)
                # time_superres = time.time() 
                # # restored_images_torch = restoration_net(images_torch, resize_to_input_size=False)
                # restored_images_torch = restoration_net(restored_images_torch, resize_to_input_size=False)
                # time_restoration = time.time() 
                
                # time_start = time.time() 
                restored_images_torch = images_torch.clone()
                for ni, net in enumerate(nets):
                    restored_images_torch = net(restored_images_torch, resize_to_input_size=False)
                # time_end = time.time() 
                # print(f"Time: {time_end - time_start}")

                if resize_to_original:
                    restored_images_torch = F.interpolate(restored_images_torch, 
                        size=(self.video_metas[sequence_id]["height"], self.video_metas[sequence_id]["width"]), 
                        mode='bicubic', align_corners=False)
                
            # # print times 
            # print(f"JPEG: {time_jpeg - time_start}")
            # print(f"Superres: {time_superres - time_jpeg}")
            # print(f"Restoration: {time_restoration - time_superres}")

            # back to uint range
            restored_images_torch = restored_images_torch.clamp(0, 1) * 255.

            # back to numpy convention
            restored_images_torch = restored_images_torch.permute(0, 2, 3, 1)
            # cpu and numpy
            restored_images_torch = restored_images_torch.cpu().numpy()
            # to uint
            restored_images_torch = restored_images_torch.astype(np.uint8)
            
            if restored_images is None:
                restored_images = np.zeros(shape=(images.shape[0], *restored_images_torch.shape[1:]), dtype=np.uint8)

            # to video tensor
            restored_images[start_i:end_i] = restored_images_torch

            start_i = end_i
            end_i = min(start_i + batch_size, images.shape[0])
        
        # write the video to file 
        vwrite(str(restored_video_file), restored_images)
        print("Video restored to: ", restored_video_file)


    def _deep_restore_sequence_sr_res(self, sequence_id):
        # jpeg_net = self._get_jpeg_network()
        superres_net = self._get_superres_network()
        restoration_net = self._get_restoration_network("GPEN-512")

        nets = [superres_net, restoration_net] 

        # self._deep_restore_sequence(sequence_id, [superres_net], \
            # output_subfolder = 'sr', batch_size=16, resize_to_original=False)
        self._deep_restore_sequence(sequence_id, nets, output_subfolder = 'sr_res', batch_size=12, 
            resize_to_original=True)
        # self._deep_restore_sequence(sequence_id, [restoration_net], input_videos = 'sr', \
            # output_subfolder = 'sr_res', batch_size=16, resize_to_original=True)

    def _get_num_shards(self, videos_per_shard): 
        num_shards = int(np.ceil( self.num_sequences / videos_per_shard))
        return num_shards


    def _process_shard(self, videos_per_shard, shard_idx, extract_audio=True,
        restore_videos=True, detect_landmarks=True, segment_videos=True, reconstruct_faces=False,
    ):
        num_shards = self._get_num_shards(videos_per_shard)
        start_idx = shard_idx * videos_per_shard
        end_idx = min(start_idx + videos_per_shard, self.num_sequences)

        print(f"Processing shard {shard_idx} of {num_shards}")

        idxs = np.arange(self.num_sequences, dtype=np.int32)
        np.random.seed(0)
        np.random.shuffle(idxs)
        
        for i in range(start_idx, end_idx):
            idx = idxs[i]
            if extract_audio: 
                self._extract_audio_for_video(idx)
            if restore_videos:
                self._deep_restore_sequence_sr_res(idx)
            if detect_landmarks:
                self._detect_faces_in_sequence(idx)
            if segment_videos:
                self._segment_faces_in_sequence(idx)
                # raise NotImplementedError()
            if reconstruct_faces: 
                # self._reconstruct_faces_in_sequence(idx, 
                #     reconstruction_net=self._get_reconstruction_network('emoca'))
                # self._reconstruct_faces_in_sequence(idx, 
                #     reconstruction_net=self._get_reconstruction_network('deep3dface'))
                # self._reconstruct_faces_in_sequence(idx, 
                #     reconstruction_net=self._get_reconstruction_network('deca'))
                # rec_methods = ['emoca', 'deep3dface', 'deca']
                rec_methods = ['emoca', 'deep3dface',]
                # rec_methods = ['emoca',]
                for rec_method in rec_methods:
                    self._reconstruct_faces_in_sequence(idx, reconstruction_net=None, device=None,
                                       save_obj=False, save_mat=True, save_vis=False, save_images=False,
                                       save_video=False, rec_method=rec_method, retarget_from=None, retarget_suffix=None)
            
        print("Done processing shard")

    def _get_subsets(self, set_type=None):
        set_type = set_type or "original"

        if set_type == "original":
            pretrain = []
            trainval = []
            test = []
            for i in range(len(self.video_list)): 
                vid_set = self._video_set(i) 
                if vid_set == "pretrain": 
                    pretrain.append(i)
                elif vid_set == "trainval":
                    trainval.append(i)
                elif vid_set == "test":
                    test.append(i)
                else:
                    raise ValueError(f"Unknown video set: {vid_set}")
            return pretrain, trainval, test

        else: 
            raise ValueError(f"Unknown set type: {set_type}")



    def setup(self, stage=None):
        train, val, test = self._get_subsets(self.split)
        self.training_set = LRS3Dataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, train, 
                self.audio_metas, self.sequence_length_train, image_size=self.image_size,
                **self.occlusion_settings_train,
                # occlusion_length=self.occlusion_length_train,
                # occlusion_probability_mouth = self.occlusion_probability_mouth_train_train,
                # occlusion_probability_left_eye = self.occlusion_probability_left_eye_train,
                # occlusion_probability_right_eye = self.occlusion_probability_right_eye_train,
                # occlusion_probability_face = self.occlusion_probability_face_train,
                )
        self.validation_set = LRS3Dataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, val, self.audio_metas, 
                self.sequence_length_val, image_size=self.image_size,  
                **self.occlusion_settings_val,
                # occlusion_length=self.occlusion_length_val,
                # occlusion_probability_mouth = self.occlusion_probability_mouth_train_val,
                # occlusion_probability_left_eye = self.occlusion_probability_left_eye_val,
                # occlusion_probability_right_eye = self.occlusion_probability_right_eye_val,
                # occlusion_probability_face = self.occlusion_probability_face_val,
            )
        self.test_set = LRS3Dataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, test, self.audio_metas, 
                self.sequence_length_test, image_size=self.image_size, 
                **self.occlusion_settings_test,
                # occlusion_length=self.occlusion_length_test,  
                # occlusion_probability_mouth = self.occlusion_probability_mouth_train_test,
                # occlusion_probability_left_eye = self.occlusion_probability_left_eye_test,
                # occlusion_probability_right_eye = self.occlusion_probability_right_eye_test,
                # occlusion_probability_face = self.occlusion_probability_face_test,
            )

        # if self.mode in ['all', 'manual']:
        #     # self.image_list += sorted(list((Path(self.path) / "Manually_Annotated").rglob(".jpg")))
        #     self.dataframe = pd.load_csv(self.path / "Manually_Annotated" / "Manually_Annotated.csv")
        # if self.mode in ['all', 'automatic']:
        #     # self.image_list += sorted(list((Path(self.path) / "Automatically_Annotated").rglob("*.jpg")))
        #     self.dataframe = pd.load_csv(
        #         self.path / "Automatically_Annotated" / "Automatically_annotated_file_list.csv")

    def train_sampler(self):
        return None
        # if self.sampler == "uniform":
        #     sampler = None
        # elif self.sampler == "balanced_expr":
        #     sampler = make_class_balanced_sampler(self.training_set.df["expression"].to_numpy())
        # elif self.sampler == "balanced_va":
        #     sampler = make_balanced_sample_by_weights(self.training_set.va_sample_weights)
        # elif self.sampler == "balanced_v":
        #     sampler = make_balanced_sample_by_weights(self.training_set.v_sample_weights)
        # elif self.sampler == "balanced_a":
        #     sampler = make_balanced_sample_by_weights(self.training_set.a_sample_weights)
        # else:
        #     raise ValueError(f"Invalid sampler value: '{self.sampler}'")
        # return sampler

    def train_dataloader(self):
        sampler = self.train_sampler()
        dl =  torch.utils.data.DataLoader(self.training_set, shuffle=sampler is None, num_workers=self.num_workers, pin_memory=True,
                        batch_size=self.batch_size_train, drop_last=self.drop_last, sampler=sampler)
        return dl

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_set, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.batch_size_val, drop_last=self.drop_last)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.batch_size_test, drop_last=self.drop_last)


class TemporalDatasetBase(torch.utils.data.Dataset):

    def __init__(self) -> None:
        super().__init__()

    def _augment_sequence_sample(self, sample):
        raise NotImplementedError()

    def visualize_sample(self, sample):
        raise NotImplementedError()



from gdl.transforms.keypoints import KeypointNormalization, KeypointScale

class LRS3Dataset(TemporalDatasetBase):

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
            image_size=None
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
        self.segmentation_type = segmentation_type 
        self.landmark_source = landmark_source 
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

        self.align_images = True

        assert self.occlusion_length[0] >= 0
        assert self.occlusion_length[1] <= self.sequence_length + 1


    def __getitem__(self, index):
        sample = {}

        # 1) VIDEO
        # load the video 
        video_path = self.root_path / self.video_list[self.video_indices[index]]
        video_meta = self.video_metas[self.video_indices[index]]

        # num video frames 
        num_frames = video_meta["num_frames"]

        assert num_frames >= self.sequence_length, f"Video {video_path} has only {num_frames} frames, but sequence length is {self.sequence_length}"

        # pick the starting video frame 
        start_frame = np.random.randint(0, num_frames - self.sequence_length)

        # load the frames
        # frames = []
        # for i in range(start_frame, start_frame + self.sequence_length):
        #     frame_path = video_path / f"frame_{i:04d}.jpg"
        #     frame = imread(str(frame_path))
        #     frames.append(frame)
        assert video_path.is_file(), f"Video {video_path} does not exist"
        try:
            frames = vread(video_path.as_posix())
            assert len(frames) == num_frames, f"Video {video_path} has {len(frames)} frames, but meta says it has {num_frames}"
            frames = frames[start_frame:(start_frame + self.sequence_length)] 
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
            for i in range(self.sequence_length):
                # frames.append(next(reader))
                _, frame = reader.read()
                # bgr to rgb 
                frame = frame[:, :, ::-1]
                frames.append(frame)
            reader.release()
            frames = np.stack(frames, axis=0)
        frames = frames.astype(np.float32) 

        # 2) AUDIO
        # load the audio 
        audio_path = (Path(self.output_dir) / "audio" / self.video_list[self.video_indices[index]]).with_suffix(".wav")
        audio_meta = self.audio_metas[self.video_indices[index]]
        samplerate, wavdata = wavfile.read(audio_path.as_posix())
        assert samplerate == 16000 and len(wavdata.shape) == 1

        # audio augmentation
        if np.random.rand() < self.audio_noise_prob:
            wavdata = self.add_noise(wavdata)

        # 
        audio_feats = logfbank(wavdata, samplerate=samplerate).astype(np.float32) # [T (num audio frames), F (num filters)]
        # the audio feats frequency (and therefore num frames) is too high, so we stack them together to match num visual frames 
        audio_feats = stacker(audio_feats, self.stack_order_audio)

        audio_feats = audio_feats[start_frame:(start_frame + self.sequence_length)] 

        # stack the frames and audio feats together
        sample = { 
            "video": frames,
            "audio": audio_feats,
        }

        # # 3) LANDMARKS 
        landmark_dict = {}
        landmark_validity_dict = {}
        for landmark_type in self.landmark_types:
            landmarks_dir = (Path(self.output_dir) / f"landmarks_{self.landmark_source}" / landmark_type /  self.video_list[self.video_indices[index]]).with_suffix("")
            landmarks = []
            if (landmarks_dir / "landmarks.pkl").exists(): # landmarks are saved per video in a single file
            #    landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmarks.pkl")  
            #    landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmarks_original.pkl")  
                landmark_list = FaceDataModuleBase.load_landmark_list(landmarks_dir / f"landmarks_{self.landmark_source}.pkl")  
                landmark_types =  FaceDataModuleBase.load_landmark_list(landmarks_dir / "landmark_types.pkl")  
                landmarks = landmark_list[start_frame: self.sequence_length + start_frame] 

                landmark_validity = np.ones(len(landmarks), dtype=np.bool)
                for li in range(len(landmarks)): 
                    if len(landmarks[li]) == 0: # dropped detection
                        if landmark_type == "mediapipe":
                            # [WARNING] mediapipe landmarks coordinates are saved in the scale [0.0-1.0] (for absolute they need to be multiplied by img size)
                            landmarks[li] = np.zeros((478, 3))
                        elif landmark_type in ["fan", "kpt68"]:
                            landmarks[li] = np.zeros((68, 2))
                        else: 
                            raise ValueError(f"Unknown landmark type '{landmark_type}'")
                        landmark_validity[li] = False
                    elif len(landmarks[li]) > 1: # multiple faces detected
                        landmarks[li] = landmarks[li][0] # just take the first one for now
                    else: 
                        landmarks[li] = landmarks[li][0] 
            else: # landmarks are saved per frame
                landmark_validity = np.ones(len(landmarks), dtype=np.bool)
                for i in range(start_frame, self.sequence_length + start_frame):
                    landmark_path = landmarks_dir / f"{i:05d}_000.pkl"
                    landmark_type, landmark = load_landmark(landmark_path)
                    landmarks += [landmark]
                    if len(landmark) == 0: # dropped detection
                        landmark = [0, 0]
                        landmark_validity[li] = False
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

            landmark_dict[landmark_type] = landmarks
            landmark_validity_dict[landmark_type] = landmark_validity
        sample["landmarks"] = landmark_dict
        sample["landmarks_validity"] = landmark_validity_dict

        # 4) SEGMENTATIONS
        segmentations_dir = (Path(self.output_dir) / f"segmentations_{self.segmentation_source}" / self.segmentation_type /  self.video_list[self.video_indices[index]]).with_suffix("")
        segmentations = []

        if (segmentations_dir / "segmentations.pkl").exists(): # segmentations are saved in a single file-per video 
            seg_images, seg_types, seg_names = load_segmentation_list(segmentations_dir / "segmentations.pkl")
            segmentations = seg_images[start_frame: self.sequence_length + start_frame]
            segmentations = np.stack(segmentations, axis=0)[:,0,...]
            segmentations = process_segmentation(segmentations, seg_types[0]).astype(np.uint8)
            assert segmentations.shape[0] == self.sequence_length
        else: # segmentations are saved per-frame
            for i in range(start_frame, self.sequence_length + start_frame):
                segmentation_path = segmentations_dir / f"{i:05d}.pkl"
                seg_image, seg_type = load_segmentation(segmentation_path)
                # seg_image = seg_image[:, :, np.newaxis]
                seg_image = process_segmentation(seg_image[0], seg_type).astype(np.uint8)
                segmentations += [seg_image]
            segmentations = np.stack(segmentations, axis=0)
        sample["segmentation"] = segmentations

        # 5) FACE ALIGNMENT IF ANY
        if self.align_images:
            landmarks_for_alignment = "mediapipe"
            left = sample["landmarks"][landmarks_for_alignment][:,:,0].min(axis=1)
            top =  sample["landmarks"][landmarks_for_alignment][:,:,1].min(axis=1)
            right =  sample["landmarks"][landmarks_for_alignment][:,:,0].max(axis=1)
            bottom = sample["landmarks"][landmarks_for_alignment][:,:,1].max(axis=1)
            old_size, center = bbox2point(left, right, top, bottom, type=landmarks_for_alignment)
            size = (old_size * self.scale).astype(np.int32)
            for i in range(self.sequence_length):
                lmk_to_warp = {k: v[i] for k,v in sample["landmarks"].items()}
                img_warped, lmk_warped = bbpoint_warp(sample["video"][i], center[i], size[i], self.image_size, landmarks=lmk_to_warp)
                seg_warped = bbpoint_warp(sample["segmentation"][i], center[i], size[i], self.image_size, 
                    order=0 # nearest neighbor interpolation for segmentation
                    )
                # img_warped *= 255. 
                sample["video"][i] = img_warped 
                # sample["segmentation"][i] = seg_warped * 255.
                sample["segmentation"][i] = seg_warped
                for k,v in lmk_warped.items():
                    sample["landmarks"][k][i][:,:2] = v

        # AUGMENTATION
        sample = self._augment_sequence_sample(sample)

        # TO TORCH
        sample = to_torch(sample)

        # T,H,W,C to T,C,H,W
        sample["video"] = sample["video"].permute(0, 3, 1, 2)
        sample["video_masked"] = sample["video_masked"].permute(0, 3, 1, 2)
        # sample["segmenation"] = sample["segmenation"].permute(0, 2, 1)
        # sample["segmentation_masked"] = sample["segmentation_masked"].permute(0, 2, 1)

        # AUDIO NORMALIZATION (if any)
        if self.audio_normalization is not None:
            if self.audio_normalization == "layer_norm":
                sample["audio"] = F.layer_norm(sample["audio"], audio_feats.shape[1:])
            else: 
                raise ValueError(f"Unsupported audio normalization {self.audio_normalization}")

        # normalize landmarks 
        if self.landmark_normalizer is not None:
            if isinstance(self.landmark_normalizer, KeypointScale):
                raise NotImplementedError("Landmark normalization is deprecated")
                self.landmark_normalizer.set_scale(
                    img.shape[0] / input_img_shape[0],
                    img.shape[1] / input_img_shape[1])
            elif isinstance(self.landmark_normalizer, KeypointNormalization):
                self.landmark_normalizer.set_scale(sample["video"].shape[2], sample["video"].shape[3])
            else:
                raise ValueError(f"Unsupported landmark normalizer type: {type(self.landmark_normalizer)}")
            for key in sample["landmarks"].keys():
                sample["landmarks"][key] = self.landmark_normalizer(sample["landmarks"][key])
        return sample


    def _augment(self, img, seg_image, landmark, input_img_shape=None):

        if self.transforms is not None:
            res = self.transforms(image=img.astype(np.float32),
                                  segmentation_maps=seg_image,
                                  keypoints=landmark)
            if seg_image is not None and landmark is not None:
                img, seg_image, landmark = res
            elif seg_image is not None:
                img, seg_image = res
            elif landmark is not None:
                img, _, landmark = res
            else:
                img = res

            img = img.astype(np.float32) / 255.0

        if seg_image is not None:
            seg_image = np.squeeze(seg_image)[..., np.newaxis].astype(np.float32)

        if landmark is not None:
            landmark = np.squeeze(landmark)
            if isinstance(self.landmark_normalizer, KeypointScale):
                self.landmark_normalizer.set_scale(
                    img.shape[0] / input_img_shape[0],
                    img.shape[1] / input_img_shape[1])
            elif isinstance(self.landmark_normalizer, KeypointNormalization):
                self.landmark_normalizer.set_scale(img.shape[0], img.shape[1])
                # self.landmark_normalizer.set_scale(input_img_shape[0], input_img_shape[1])
            else:
                raise ValueError(f"Unsupported landmark normalizer type: {type(self.landmark_normalizer)}")
            landmark = self.landmark_normalizer(landmark)

        return img, seg_image, landmark


    def _augment_sequence_sample(self, sample):
        # get the mediapipe landmarks 
        mediapipe_landmarks = sample["landmarks"]["mediapipe"]
        mediapipe_landmarks_valid = sample["landmarks_validity"]["mediapipe"]
        # mediapipe_landmarks = []
        images = sample["video"]
        segmentation = sample["segmentation"]

        images_masked = np.copy(images)
        segmentation_masked = np.copy(segmentation)

        # compute mouth region bounding box
        if np.random.rand() < self.occlusion_probability_mouth: 
            images_masked, segmentation_masked = self._occlude_sequence(images_masked, segmentation_masked, 
                mediapipe_landmarks, mediapipe_landmarks_valid, "mouth")

        # compute eye region bounding box
        if np.random.rand() < self.occlusion_probability_left_eye:
            images_masked, segmentation_masked = self._occlude_sequence(images_masked, segmentation_masked, 
                mediapipe_landmarks, mediapipe_landmarks_valid, "left_eye")

        if np.random.rand() < self.occlusion_probability_right_eye:
            images_masked, segmentation_masked = self._occlude_sequence(images_masked, segmentation_masked, 
                mediapipe_landmarks, mediapipe_landmarks_valid, "right_eye")

        # compute face region bounding box
        if np.random.rand() < self.occlusion_probability_face:
            images_masked, segmentation_masked = self._occlude_sequence(images_masked, segmentation_masked, 
                mediapipe_landmarks, mediapipe_landmarks_valid, "all")

        sample["video_masked"] = images_masked
        sample["segmentation_masked"] = segmentation_masked

        return sample

    def _occlude_sequence(self, images, segmentation, mediapipe_landmarks, mediapipe_landmarks_valid, region):
        bounding_boxes, sizes = self.occluder.bounding_box_batch(mediapipe_landmarks, region)
        
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

        start_frame = np.random.randint(0, self.sequence_length - occlusion_length + 1)
        end_frame = start_frame + occlusion_length

        images = self.occluder.occlude_batch(images, region, landmarks=None, 
            bounding_box_batch=bounding_boxes, start_frame=start_frame, end_frame=end_frame)
        segmentation = self.occluder.occlude_batch(segmentation, region, landmarks=None, 
            bounding_box_batch=bounding_boxes, start_frame=start_frame, end_frame=end_frame)
        return images, segmentation

    def add_noise(self, wavdata):
        raise NotImplementedError(  )
        noise = np.random.randn(len(wavdata))
        return wavdata + noise

    def __len__(self): 
        return len(self.video_indices)

    def visualize_sample(self, sample_or_index):
        if isinstance(sample_or_index, int):
            index = sample_or_index
            sample = self[index]
        else:
            sample = sample_or_index

        # visualize the video
        video_frames = sample["video"]
        segmentation = sample["segmentation"]
        video_frames_masked = sample["video_masked"]
        segmentation_masked = sample["segmentation_masked"]
        landmarks_mp = sample["landmarks"]["mediapipe"]

        landmarks_mp = self.landmark_normalizer.inv(landmarks_mp)

        # T, C, W, H to T, W, H, C 
        video_frames = video_frames.permute(0, 2, 3, 1)
        video_frames_masked = video_frames_masked.permute(0, 2, 3, 1)
        segmentation = segmentation[..., None]
        segmentation_masked = segmentation_masked[..., None]

        # plot the video frames with plotly
        # horizontally concatenate the frames
        frames = np.concatenate(video_frames.numpy(), axis=1)
        frames_masked = np.concatenate(video_frames_masked.numpy(), axis=1)
        segmentation = np.concatenate(segmentation.numpy(), axis=1)
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

        video_frames_landmarks = np.copy(video_frames)
        for i in range(video_frames_landmarks.shape[0]):
            mp_drawing.draw_landmarks(
                image=video_frames_landmarks[i],
                landmark_list=landmarks_mp_list[i],
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style()
                )
        
        video_frames_landmarks = np.concatenate(video_frames_landmarks, axis=1)

        all_images = [frames, np.tile( segmentation*255, (1,1,3)), 
            frames_masked, np.tile(segmentation_masked*255, (1,1,3)), video_frames_landmarks]
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


def main(): 
    root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs3/extracted")
    output_dir = Path("/ps/scratch/rdanecek/data/lrs3")

    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs2/mvlrs_v1")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs2")

    processed_subfolder = "processed"

    # Create the dataset
    dm = LRS3DataModule(root_dir, output_dir, processed_subfolder)

    # Create the dataloader
    dm.prepare_data() 

    from skvideo.io import vread 

    # frames = vread(str(dm.root_dir / dm.video_list[0]))
    # audio_file = dm._get_path_to_sequence_audio(0)

    # # read audio with scipy 
    # import scipy.io.wavfile as wavfile
    # import scipy.signal as signal
    # import numpy as np
    # #  read audio
    # fs, audio = wavfile.read(audio_file)

    # dm._extract_audio()
    # dm._detect_faces()

    # dm._segment_faces_in_sequence(0)
    idxs = np.arange(dm.num_sequences)
    np.random.seed(0)
    np.random.shuffle(idxs)

    for i in range(dm.num_sequences):
        dm._deep_restore_sequence_sr_res(idxs[i])

    # dm.setup()





if __name__ == "__main__": 
    main()
