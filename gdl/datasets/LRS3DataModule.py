from gdl.datasets.FaceVideoDataModule import FaceVideoDataModule 
from pathlib import Path


class LRS3DataModule(FaceVideoDataModule):

    def __init__(self, root_dir, output_dir, 
                processed_subfolder=None, face_detector='fan', 
                face_detector_threshold=0.9, image_size=224, scale=1.25, device=None):
        super().__init__(root_dir, output_dir, processed_subfolder, 
            face_detector, face_detector_threshold, image_size, scale, device)

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
        assert file_type in ['videos', 'detections', "landmarks", "segmentations", 
            "emotions", "reconstructions"]
        video_file = self.video_list[sequence_id]
        if len(method) > 0:
            file_type += "_" + method 
        if len(suffix) > 0:
            file_type += suffix

        suffix = Path(file_type) / self._video_set(sequence_id) / self._video_supername(sequence_id) / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder



def main(): 
    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs3")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs3")

    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs2/mvlrs_v1")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs2")

    processed_subfolder = "processed"

    # Create the dataset
    dm = LRS3DataModule(root_dir, output_dir, processed_subfolder)

    # Create the dataloader
    dm.prepare_data() 

    dm.setup()





if __name__ == "__main__": 
    main()
