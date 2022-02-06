from gdl.datasets.FaceVideoDataModule import FaceVideoDataModule 
from pathlib import Path
import torch
import torch.nn.functional as F
import os, sys
from gdl.utils.other import get_path_to_externals
import numpy as np


import time


class LRS3DataModule(FaceVideoDataModule):

    def __init__(self, root_dir, output_dir, 
                processed_subfolder=None, face_detector='3fabrec', landmarks_from='sr_res',
                face_detector_threshold=0.9, image_size=224, scale=1.25, device=None):
        super().__init__(root_dir, output_dir, processed_subfolder, 
            face_detector, face_detector_threshold, image_size, scale, device, 
            unpack_videos=False, save_detection_images=False, save_landmarks=True)
        self.detect_landmarks_on_restored_images = landmarks_from

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

    def _get_superres_network(self):
        # try:
        from gdl.models.external.SwinIRTranslation import SwinIRRealSuperRes
        # except ImportError: 
            # print("Could not import SwinIRTranslation. Skipping.") 
        return SwinIRRealSuperRes( 256)


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
                
                restored_images_torch = images_torch.clone()
                for ni, net in enumerate(nets):
                    restored_images_torch = net(restored_images_torch, resize_to_input_size=False)
                
                if resize_to_original:
                    restored_images_torch = F.interpolate(restored_images_torch, 
                        size=(self.video_metas[i]["height"], self.video_metas[i]["width"]), 
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
        self._deep_restore_sequence(sequence_id, nets, output_subfolder = 'sr_res', batch_size=12, resize_to_original=False)
        # self._deep_restore_sequence(sequence_id, [restoration_net], input_videos = 'sr', \
            # output_subfolder = 'sr_res', batch_size=16, resize_to_original=True)

    def _get_num_shards(self, videos_per_shard): 
        num_shards = int(np.ceil( self.num_sequences / videos_per_shard))
        return num_shards


    def _process_shard(self, videos_per_shard, shard_idx, 
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
                # rec_methods = ['deep3dface',]
                rec_methods = ['emoca',]
                for rec_method in rec_methods:
                    self._reconstruct_faces_in_sequence(idx, reconstruction_net=None, device=None,
                                       save_obj=False, save_mat=True, save_vis=False, save_images=False,
                                       save_video=False, rec_method=rec_method, retarget_from=None, retarget_suffix=None)
            
        print("Done processing shard")


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
