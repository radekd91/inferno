from gdl.datasets.FaceVideoDataModule import FaceVideoDataModule 
from pathlib import Path
import torch
import torch.nn.functional as F
import os, sys
from gdl.utils.other import get_path_to_externals
import numpy as np
try:
    from gdl.models.external.GPENFaceRestoration import GPENFaceRestoration
except ImportError: 
    print("Could not import GPENFaceRestoration. Skipping.") 
try:
    from gdl.models.external.SwinIRTranslation import SwinIRRealSuperRes, SwinIRCompressionArtifact
except ImportError: 
    print("Could not import SwinIRTranslation. Skipping.") 
import time


class LRS3DataModule(FaceVideoDataModule):

    def __init__(self, root_dir, output_dir, 
                processed_subfolder=None, face_detector='3fabrec', 
                face_detector_threshold=0.9, image_size=224, scale=1.25, device=None):
        super().__init__(root_dir, output_dir, processed_subfolder, 
            face_detector, face_detector_threshold, image_size, scale, device, 
            unpack_videos=False, save_detection_images=False, save_landmarks=True)

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
        assert file_type in ['videos', 'detections', "landmarks", "landmarks_original", "segmentations", 
            "emotions", "reconstructions",  "audio", "videos_restored" ]
        video_file = self.video_list[sequence_id]
        if len(method) > 0:
            file_type += "_" + method 
        if len(suffix) > 0:
            file_type += suffix

        suffix = Path(file_type) / self._video_set(sequence_id) / self._video_supername(sequence_id) / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder


    def _get_path_to_sequence_restored(self, sequence_id, file_type, method="", suffix=""):
        return self._get_path_to_sequence_files(sequence_id, "videos_restored", method, suffix).with_suffix(".mp4")


    def _get_restoration_network(self, method):
        return GPENFaceRestoration(method)

    def _get_jpeg_network(self):
        return SwinIRCompressionArtifact( 256)

    def _get_superres_network(self):
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


    def _deep_restore_sequence(self, sequence_id, method="GPEN-512"): 
        from skvideo.io import vread, vwrite

        assert method in ['GPEN-512', 'GPEN-256']

        # if method == "GPEN-512": 
        #     im_size = 512
        #     model_name = "GPEN-BFR-512.pth"
        # elif method == 'GPEN-256': 
        #     im_size = 256
        #     model_name = "GPEN-BFR-256.pth"
        # else: 
        #     raise NotImplementedError()

        video_file = self.root_dir / self.video_list[sequence_id]
        restored_video_file = self._get_path_to_sequence_restored(sequence_id, method) 
        # restored_video_file = Path(str(restored_video_file).replace('videos_restored', 'videos_restored_jpg'))
        # restored_video_file = Path(str(restored_video_file).replace('videos_restored', 'videos_restored_jpg_sr'))
        # restored_video_file = Path(str(restored_video_file).replace('videos_restored', 'videos_restored_sr'))
        # restored_video_file = Path(str(restored_video_file).replace('videos_restored', 'videos_restored_jpg_sr_res'))
        # restored_video_file = Path(str(restored_video_file).replace('videos_restored', 'videos_restored_sr_res'))

        # batch_size = 16 
        batch_size = 12
        # batch_size = 6 

        restored_video_file.parent.mkdir(parents=True, exist_ok=True)

        images =  vread(str(video_file))
        restored_images = np.zeros_like(images)

        start_i = 0 
        end_i = min(start_i + batch_size, images.shape[0])

        # jpeg_net = self._get_jpeg_network()
        superres_net = self._get_superres_network()

        restoration_net = self._get_restoration_network(method)

        while start_i < images.shape[0]:
            print(f"Restoring {start_i} to {end_i}")
            images_torch = torch.from_numpy(images[start_i:end_i]).float() / 255.  
            # transpose into torch format 
            images_torch = images_torch.permute(0, 3, 1, 2).cuda()

            with torch.no_grad():
                time_start = time.time()
                # restored_images_torch = jpeg_net(images_torch, resize_to_input_size=False)
                time_jpeg = time.time() 
                restored_images_torch = superres_net(images_torch, resize_to_input_size=False)
                # restored_images_torch = superres_net(restored_images_torch, resize_to_input_size=False)
                time_superres = time.time() 
                # restored_images_torch = restoration_net(images_torch, resize_to_input_size=False)
                restored_images_torch = restoration_net(restored_images_torch, resize_to_input_size=False)
                time_restoration = time.time() 
                restored_images_torch = F.interpolate(restored_images_torch, size=(images_torch.shape[2], images_torch.shape[3]), 
                    mode='bicubic', align_corners=False)
                

            # print times 
            print(f"JPEG: {time_jpeg - time_start}")
            print(f"Superres: {time_superres - time_jpeg}")
            print(f"Restoration: {time_restoration - time_superres}")

            # # RGB to BGR
            # images_torch = images_torch[:, [2, 1, 0], :, :]

            # images_torch = F.interpolate(images_torch, size=(im_size, im_size), mode='bicubic', align_corners=False)
            # images_torch = (images_torch - 0.5) / 0.5
            # images_torch = images_torch.cuda()
            # with torch.no_grad():
            #     restored_images_torch, _ = net.model(images_torch)
            # restored_images_torch = (restored_images_torch * 0.5 + 0.5).clamp(0, 1) * 255.
            # restored_images_torch = F.interpolate(restored_images_torch, size=(images.shape[1], images.shape[2]), mode='bicubic', align_corners=False) 
            # # float to uint 
            # 
            # # BGR to RGB
            # restored_images_torch = restored_images_torch[:, :, :, [2, 1, 0]]

            # back to uint range
            restored_images_torch = restored_images_torch.clamp(0, 1) * 255.

            # back to numpy convention
            restored_images_torch = restored_images_torch.permute(0, 2, 3, 1)
            # cpu and numpy
            restored_images_torch = restored_images_torch.cpu().numpy()
            # to uint
            restored_images_torch = restored_images_torch.astype(np.uint8)
            
            # to video tensor
            restored_images[start_i:end_i] = restored_images_torch

            start_i = end_i
            end_i = min(start_i + batch_size, images.shape[0])
        
        # write the video to file 
        vwrite(str(restored_video_file), restored_images)
        print("Video restored to: ", restored_video_file)


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
        dm._deep_restore_sequence(idxs[i])

    # dm.setup()





if __name__ == "__main__": 
    main()
