from pathlib import Path
from gdl.datasets.FaceDataModuleBase import FaceDataModuleBase
from gdl.datasets.FaceVideoDataModule import FaceVideoDataModule 
import numpy as np



class CelebVHQDataModule(FaceVideoDataModule): 

    def __init__(self, root_dir, output_dir, 
            processed_subfolder=None, 
            face_detector='mediapipe', 
            # landmarks_from='sr_res',
            landmarks_from=None,
            face_detector_threshold=0.5, 
            image_size=224, scale=1.25, 
            processed_video_size=256,
            batch_size_train=16,
            batch_size_val=16,
            batch_size_test=16,
            sequence_length_train=16,
            sequence_length_val=16,
            sequence_length_test=16,
            # occlusion_length_train=0,
            # occlusion_length_val=0,
            # occlusion_length_test=0,            
            bb_center_shift_x=0., # in relative numbers
            bb_center_shift_y=0, # in relative numbers (i.e. -0.1 for 10% shift upwards, ...)
            occlusion_settings_train=None,
            occlusion_settings_val=None,
            occlusion_settings_test=None,
            split = "original", #TODO: does CelebVHQ offer any split?
            num_workers=4,
            device=None,
            augmentation=None,
            drop_last=True,
            ):
        super().__init__(root_dir, output_dir, processed_subfolder, 
            face_detector, face_detector_threshold, image_size, scale, 
            processed_video_size=processed_video_size,
            device=device, 
            unpack_videos=False, save_detection_images=False, 
            # save_landmarks=True,
            save_landmarks=False, # trying out this option
            save_landmarks_one_file=True, 
            save_segmentation_frame_by_frame=False, 
            save_segmentation_one_file=True,
            bb_center_shift_x=bb_center_shift_x, # in relative numbers
            bb_center_shift_y=bb_center_shift_y, # in relative numbers (i.e. -0.1 for 10% shift upwards, ...)
            )
        # self.detect_landmarks_on_restored_images = landmarks_from
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

        self.annotation_json_path = Path(root_dir).parent / "celebvhq_info.json" 
        assert self.annotation_json_path.is_file()

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

    def _gather_data(self, exist_ok=True):
        super()._gather_data(exist_ok)
        
        # vl = [(path.parent / path.stem).as_posix() for path in self.video_list]
        # al = [(path.parent / path.stem).as_posix() for path in self.annotation_list]

        # vl_set = set(vl)
        # al_set = set(al)

        # vl_diff = vl_set.difference(al_set)
        # al_diff = al_set.difference(vl_set)

        # intersection = vl_set.intersection(al_set) 

        # print(f"Video list: {len(vl_diff)}")
        # print(f"Annotation list: {len(al_diff)}")

        # if len(vl_diff) != 0:
        #     print("Video list is not equal to annotation list") 
        #     print("Video list difference:")
        #     print(vl_diff)
        #     raise RuntimeError("Video list is not equal to annotation list")
        
        # if len(al_diff) != 0: 
        #     print("Annotation list is not equal to video list")    
        #     print("Annotation list difference:")
        #     print(al_diff)
        #     raise RuntimeError("Annotation list is not equal to video list")

        # print(f"Intersection: {len(intersection)}")


    def _filename2index(self, filename):
        return self.video_list.index(filename)

    def _get_landmark_method(self):
        return self.face_detector_type

    def _get_segmentation_method(self):
        return "bisenet"

    def _detect_faces(self):
        return super()._detect_faces( )

    def _get_num_shards(self, videos_per_shard): 
        num_shards = int(np.ceil( self.num_sequences / videos_per_shard))
        return num_shards

    def _process_video(self, idx, extract_audio=True, 
            restore_videos=True, 
            detect_landmarks=True, 
            recognize_faces=True,
            cut_out_faces=True,
            segment_videos=True, 
            reconstruct_faces=False,):
        if extract_audio: 
            self._extract_audio_for_video(idx)
        # if restore_videos:
        #     self._deep_restore_sequence_sr_res(idx)
        if detect_landmarks:
            self._detect_faces_in_sequence(idx)
        if recognize_faces: 
            self._recognize_faces_in_sequence(idx)
            self._identify_recognitions_for_sequence(idx)
            self._extract_personal_recognition_sequences(idx)
        if cut_out_faces: 
            self._cut_out_detected_faces_in_sequence(idx)
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
            self._process_video(idx, extract_audio=extract_audio, restore_videos=restore_videos,
                detect_landmarks=detect_landmarks, segment_videos=segment_videos, reconstruct_faces=reconstruct_faces)
            
        print("Done processing shard")

    def _get_path_to_sequence_files(self, sequence_id, file_type, method="", suffix=""): 
        assert file_type in ['videos', 'videos_aligned', 'detections', "landmarks", "landmarks_original", "segmentations", 
            "emotions", "reconstructions", "audio"]
        video_file = self.video_list[sequence_id]
        if len(method) > 0:
            file_type += "_" + method 
        if len(suffix) > 0:
            file_type += suffix

        suffix = Path(file_type) / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder

def main(): 
    root_dir = Path("/ps/project/EmotionalFacialAnimation/data/celebvhq/auto_processed")
    output_dir = Path("/ps/scratch/rdanecek/data/celebvhq/")

    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs2/mvlrs_v1")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs2")

    processed_subfolder = "processed"

    # Create the dataset
    dm = CelebVHQDataModule(root_dir, output_dir, processed_subfolder)

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
