from cv2 import imread
import torchaudio
from inferno.datasets.FaceDataModuleBase import FaceDataModuleBase
from inferno.datasets.FaceVideoDataModule import FaceVideoDataModule 
from pathlib import Path
import torch
import torch.nn.functional as F
import os, sys
from inferno.utils.FaceDetector import load_landmark
from inferno.utils.MediaPipeLandmarkDetector import np2mediapipe
from inferno.utils.other import get_path_to_externals
from inferno.utils.MediaPipeFaceOccluder import MediaPipeFaceOccluder, sizes_to_bb, sizes_to_bb_batch
import numpy as np
import pandas as pd
from skvideo.io import vread, vreader
from scipy.io import wavfile
import time
from python_speech_features import logfbank
from inferno.datasets.IO import load_segmentation, process_segmentation, load_segmentation_list
from inferno.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp
from inferno.transforms.imgaug import create_image_augmenter
from inferno.utils.collate import robust_collate
import imgaug
import traceback


class LRS3DataModule(FaceVideoDataModule):

    def __init__(self, root_dir, output_dir, 
                processed_subfolder=None, 
                face_detector='mediapipe', 
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
                include_processed_audio = True,
                include_raw_audio = True,
                preload_videos=False,
                inflate_by_video_size=False,
                landmark_types = None,
                landmark_sources=None,
                segmentation_source=None,
                segmentation_type =None,
                return_mica_images=False,
                ):
        super().__init__(root_dir, output_dir, processed_subfolder, 
            face_detector, face_detector_threshold, image_size, scale, device, 
            unpack_videos=False, save_detection_images=False, 
            # save_landmarks=True,
            save_landmarks=False, # trying out this option
            save_landmarks_one_file=True, 
            save_segmentation_frame_by_frame=False, 
            save_segmentation_one_file=True,

            include_processed_audio = include_processed_audio,
            include_raw_audio = include_raw_audio,
            preload_videos=preload_videos,
            inflate_by_video_size=inflate_by_video_size,
            return_mica_images=return_mica_images,
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
        self.drop_last = drop_last

        # self.occlusion_length_train = occlusion_length_train
        # self.occlusion_length_val = occlusion_length_val
        # self.occlusion_length_test = occlusion_length_test
        self.occlusion_settings_train = occlusion_settings_train or {}
        self.occlusion_settings_val = occlusion_settings_val or {}
        self.occlusion_settings_test = occlusion_settings_test or {}
        self.augmentation = augmentation

        self.landmark_types = landmark_types or ["mediapipe", "fan"]
        self.landmark_sources = landmark_sources or ["original", "aligned"]
        self.segmentation_source = segmentation_source or "aligned"
        self.segmentation_type = segmentation_type or "bisenet"

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

    def _filename2index(self, filename):
        return self.video_list.index(filename)

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

    def _get_path_to_sequence_files(self, sequence_id, file_type, method="", suffix="", assert_=True): 
        if assert_:
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
        from inferno.models.external.GPENFaceRestoration import GPENFaceRestoration
        # except ImportError: 
        #     print("Could not import GPENFaceRestoration. Skipping.") 
        return GPENFaceRestoration(method)

    def _get_jpeg_network(self):
        # try:
        from inferno.models.external.SwinIRTranslation import SwinIRCompressionArtifact
        # except ImportError: 
        #     print("Could not import SwinIRTranslation. Skipping.") 
        return SwinIRCompressionArtifact( 256)

    def _get_superres_network(self, method="swin_ir"):
        # try:
        if method == "swin_ir":
            from inferno.models.external.SwinIRTranslation import SwinIRRealSuperRes
            # except ImportError: 
                # print("Could not import SwinIRTranslation. Skipping.") 
            return SwinIRRealSuperRes( 256)
        elif method == "bsrgan":
            from inferno.models.external.BSRGANSuperRes import BSRSuperRes
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

    def _process_video(self, idx, extract_audio=True, restore_videos=True, 
            detect_landmarks=True, segment_videos=True, reconstruct_faces=False, 
            recognize_emotions=False,):
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
            # rec_methods = ['emoca', 'spectre',]
            rec_methods = ['EMICA-CVT_flame2020',]
            # rec_methods = ['EMICA-CVT_flame2023',]
            self._reconstruct_faces_in_sequence_v2(
                                    idx, reconstruction_net=None, device=None,
                                    save_obj=False, save_mat=True, save_vis=False, save_images=False,
                                    save_video=False, rec_methods=rec_methods, retarget_from=None, retarget_suffix=None)
        if recognize_emotions:
            emo_methods = ['resnet50', ]
            self._extract_emotion_in_sequence(idx, emo_methods=emo_methods)
  

    def _process_shard(self, videos_per_shard, shard_idx, extract_audio=True,
        restore_videos=True, detect_landmarks=True, segment_videos=True, reconstruct_faces=False,
        recognize_emotions=False,
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
            self._process_video(idx, extract_audio=extract_audio, restore_videos=restore_videos,
                detect_landmarks=detect_landmarks, segment_videos=segment_videos, reconstruct_faces=reconstruct_faces, 
                recognize_emotions=recognize_emotions)
            # if extract_audio: 
            #     self._extract_audio_for_video(idx)
            # if restore_videos:
            #     self._deep_restore_sequence_sr_res(idx)
            # if detect_landmarks:
            #     self._detect_faces_in_sequence(idx)
            # if segment_videos:
            #     self._segment_faces_in_sequence(idx)
            #     # raise NotImplementedError()
            # if reconstruct_faces: 
            #     # self._reconstruct_faces_in_sequence(idx, 
            #     #     reconstruction_net=self._get_reconstruction_network('emoca'))
            #     # self._reconstruct_faces_in_sequence(idx, 
            #     #     reconstruction_net=self._get_reconstruction_network('deep3dface'))
            #     # self._reconstruct_faces_in_sequence(idx, 
            #     #     reconstruction_net=self._get_reconstruction_network('deca'))
            #     # rec_methods = ['emoca', 'deep3dface', 'deca']
            #     rec_methods = ['emoca', 'deep3dface',]
            #     # rec_methods = ['emoca',]
            #     for rec_method in rec_methods:
            #         self._reconstruct_faces_in_sequence(idx, reconstruction_net=None, device=None,
            #                            save_obj=False, save_mat=True, save_vis=False, save_images=False,
            #                            save_video=False, rec_method=rec_method, retarget_from=None, retarget_suffix=None)
            
        print("Done processing shard")

    def _get_subsets(self, set_type=None):
        set_type = set_type or "original"
        self.temporal_split = None
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
        elif "pretrain" == set_type:
            if set_type == "pretrain":
                pretrain = []
                trainval = [] 
                test = []
                for i in range(len(self.video_list)): 
                    vid_set = self._video_set(i) 
                    if vid_set == "pretrain": 
                        pretrain.append(i) 
                return pretrain, trainval, test
        elif "random_by_identity_pretrain" in set_type:
            # pretrain_02d_02d, such as pretrain_80_20 
            res = set_type.split("_")
            train = float(res[-2])
            val = float(res[-1])
            train = train / (train + val)
            val = 1 - train
            indices = np.arange(len(self.video_list), dtype=np.int32)
            # get video_clips_by_identity
            video_clips_by_identity = {}
            video_clips_by_identity_indices = {}
            index_counter = 0
            for i in range(len(self.video_list)):
                key = self._video_set(i) + "/" + self._video_supername(i) 
                if key in video_clips_by_identity.keys(): 
                    video_clips_by_identity[key] += [i]
                else: 
                    video_clips_by_identity[key] = [i]
                    video_clips_by_identity_indices[key] = index_counter
                    index_counter += 1
            
            import random
            seed = 4
            # get the list of identities
            identities = list(video_clips_by_identity.keys())
            random.Random(seed).shuffle(identities)
            # identitities randomly shuffled 
            # this determines which identities are for training and which for validation

            # get the list of corresponding indices
            # indices = [] # identity index list shuffled the samte way as the identity list
            # for identity in identities:
            #     indices += [video_clips_by_identity_indices[identity]]

            training = [] 
            validation = [] 
            for i, identity in enumerate(identities): 
                # identity = identities[i] 
                identity_videos = video_clips_by_identity[identity]
                if i < int(train * len(identities)): 
                    training += identity_videos
                else:
                    validation += identity_videos
            training.sort() 
            validation.sort()
            # at this point, the identities are shuffled but per-identity videos have 
            # consecutive indices, for training, shuffle afterwards (set shuffle to True or use a 
            # sampler )
            return training, validation, []

        elif "random_by_video_pretrain" in set_type:
            # pretrain_02d_02d, such as pretrain_80_20 
            res = set_type.split("_")
            if len(res) != 3:
                raise ValueError(f"Unknown set type: {set_type}")
            training = float(res[1])
            val = float(res[2])
            training = training / (training + val)
            val = 1 - training
            indices = np.arange(len(self.video_list), dtype=np.int32)
            np.random.seed(0)
            np.random.shuffle(indices)
            training = indices[:int(training * len(indices))].tolist()
            validation = indices[int(training * len(indices)):].tolist()
            return training, validation, []
        elif "specific_identity" in set_type: 
            res = set_type.split("_")
            identity = res[-1]
            train = float(res[-3])
            val = float(res[-2])
            train = train / (train + val)
            val = 1 - train
            indices = [i for i in range(len(self.video_list)) if self._video_set(i) + "/" + self._video_supername(i) == identity]

            training = indices[:int(train * len(indices))] 
            validation = indices[int(train * len(indices)):]
            import random
            seed = 4
            random.Random(seed).shuffle(training)
            random.Random(seed).shuffle(validation)
            return training, validation, []
        elif set_type == "all":
            pretrain = list(range(len(self.video_list)))
            trainval = list(range(len(self.video_list)))
            test = list(range(len(self.video_list)))
            return pretrain, trainval, test

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
        else: 
            raise ValueError(f"Unknown set type: {set_type}")

    def get_single_video_dataset(self, i):
        dataset = LRS3Dataset(self.root_dir, self.output_dir, 
                self.video_list, self.video_metas, [i], self.audio_metas, 
                # self.sequence_length_val, 
                "all",
                image_size=self.image_size,  
                hack_length=False, 
                occlusion_length=0,
                occlusion_probability_mouth = 0.0,
                occlusion_probability_left_eye = 0.0,
                occlusion_probability_right_eye = 0.0,
                occlusion_probability_face = 0.0,

                landmark_source=self.landmark_sources,
                landmark_types=self.landmark_types,
                segmentation_source=self.segmentation_source,
                segmentation_type=self.segmentation_type,
                return_mica_images=self.return_mica_images,
            )

        return dataset


    def setup(self, stage=None):
        train, val, test = self._get_subsets(self.split)
        training_augmenter = create_image_augmenter(self.image_size, self.augmentation)
        self.training_set = LRS3Dataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, train, 
                self.audio_metas, self.sequence_length_train, image_size=self.image_size, 
                transforms=training_augmenter,
                **self.occlusion_settings_train,
                hack_length='auto', 
                temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,

                landmark_source=self.landmark_sources,
                landmark_types=self.landmark_types,
                segmentation_source=self.segmentation_source,
                segmentation_type=self.segmentation_type,
                return_mica_images=self.return_mica_images,
              )


                    
        self.validation_set = LRS3Dataset(self.root_dir, self.output_dir, 
                self.video_list, self.video_metas, val, self.audio_metas, 
                self.sequence_length_val, image_size=self.image_size,  
                **self.occlusion_settings_val,
                hack_length=False,
                temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,
                
                landmark_source=self.landmark_sources,
                landmark_types=self.landmark_types,
                segmentation_source=self.segmentation_source,
                segmentation_type=self.segmentation_type,
                return_mica_images=self.return_mica_images,
            )


        self.test_set = LRS3Dataset(self.root_dir, self.output_dir, self.video_list, self.video_metas, test, self.audio_metas, 
                self.sequence_length_test, image_size=self.image_size, 
                **self.occlusion_settings_test,
                hack_length=False,
                temporal_split_start=self.temporal_split[0] if self.temporal_split is not None else None,
                temporal_split_end= self.temporal_split[0] + self.temporal_split[1] if self.temporal_split is not None else None,
                preload_videos=self.preload_videos,
                inflate_by_video_size=self.inflate_by_video_size,

                
                landmark_source=self.landmark_sources,
                landmark_types=self.landmark_types,
                segmentation_source=self.segmentation_source,
                segmentation_type=self.segmentation_type,
                return_mica_images=self.return_mica_images,
                )

        if "specific_identity" in self.split: 
            # let's just do this for the small experiments for now
            # just to compute what the losses are is wrt to unoccluded images
            self.validation_set_2 = LRS3Dataset(self.root_dir, self.output_dir, 
                self.video_list, self.video_metas, val, self.audio_metas, 
                self.sequence_length_val, image_size=self.image_size,  
                occlusion_length=0,
                occlusion_probability_mouth = 0.0,
                occlusion_probability_left_eye = 0.0,
                occlusion_probability_right_eye = 0.0,
                occlusion_probability_face = 0.0,

                
                landmark_source=self.landmark_sources,
                landmark_types=self.landmark_types,
                segmentation_source=self.segmentation_source,
                segmentation_type=self.segmentation_type,
                return_mica_images=self.return_mica_images,
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
        dl =  torch.utils.data.DataLoader(self.training_set, shuffle=sampler is None, num_workers=self.num_workers, 
                        pin_memory=True,
                        # pin_memory=False,
                        batch_size=self.batch_size_train, drop_last=self.drop_last, sampler=sampler, 
                        collate_fn=robust_collate,
                        persistent_workers=self.num_workers > 0,
                        )
        return dl

    def val_dataloader(self):
        dl = torch.utils.data.DataLoader(self.validation_set, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.batch_size_val, 
                        #   drop_last=self.drop_last
                          drop_last=False,
                          collate_fn=robust_collate, 
                          persistent_workers=self.num_workers > 0,
                          )
        if hasattr(self, "validation_set_2"): 
            dl2 =  torch.utils.data.DataLoader(self.validation_set_2, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                            batch_size=self.batch_size_val, 
                            # drop_last=self.drop_last, 
                            drop_last=False, 
                            collate_fn=robust_collate,
                            persistent_workers=self.num_workers > 0,
                            )
            return [dl, dl2]
        return dl 

    def test_dataloader(self):
        if hasattr(self, "test_set") and self.test_set is not None:
            return torch.utils.data.DataLoader(self.test_set, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.batch_size_test, 
                          drop_last=False,
                        #   drop_last=self.drop_last,
                          collate_fn=robust_collate
                          )
        return None



from inferno.transforms.keypoints import KeypointNormalization, KeypointScale
from inferno.datasets.VideoDatasetBase import VideoDatasetBase



class LRS3Dataset(VideoDatasetBase):

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
            return_mica_images=False,
    ) -> None:
        landmark_types = landmark_types or "mediapipe"
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
            read_video = read_video,
            read_audio = read_audio,
            reconstruction_type = reconstruction_type,
            return_global_pose = return_global_pose,
            return_appearance = return_appearance,
            average_shape_decode = average_shape_decode,
            emotion_type = emotion_type,
            return_emotion_feature = return_emotion_feature,
            return_mica_images = return_mica_images,
        )



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

