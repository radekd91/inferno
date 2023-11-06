# from inferno.models.external.Deep3DFace import Deep3DFaceModule # to make sure nvdiffrast is imported without segfaulting
import os, sys 
import numpy as np 
from pathlib import Path
from inferno.datasets.LRS3DataModule import LRS3DataModule


def main(): 
    root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs3/extracted")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs3")
    # output_dir = Path("/is/cluster/work/rdanecek/data/lrs3")
    output_dir = Path("/is/cluster/fast/rdanecek/data/lrs3")

    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs2/mvlrs_v1")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs2")

    # processed_subfolder = "processed"
    processed_subfolder = "processed2"
    # processed_subfolder = None

    # Create the dataset
    dm = LRS3DataModule(root_dir, output_dir, processed_subfolder,
        face_detector_threshold=0.05,
        landmarks_from=None,
        )

    # Create the dataloader
    dm.prepare_data() 
    # sys.exit()

    videos_per_shard = 200 
    shard_idx = 0
    if len(sys.argv) > 1:
        videos_per_shard = int(sys.argv[1])

    if len(sys.argv) > 2:
        shard_idx = int(sys.argv[2])

    print(videos_per_shard, shard_idx)
    print(dm._get_num_shards(videos_per_shard))
    # sys.exit(0)

    if len(sys.argv) > 3:
        extract_audio = bool(int(sys.argv[3]))
    else: 
        extract_audio = False
    if len(sys.argv) > 4:
        restore_videos = bool(int(sys.argv[4]))
    else: 
        restore_videos = False
    if len(sys.argv) > 5:
        detect_landmarks = bool(int(sys.argv[5]))
    else: 
        detect_landmarks = False
    if len(sys.argv) > 6:
        segment_videos = bool(int(sys.argv[6]))
    else: 
        segment_videos = False
    if len(sys.argv) > 7:
        reconstruct_faces = bool(int(sys.argv[7])) 
    else: 
        reconstruct_faces = False
    if len(sys.argv) > 8:
        recognize_emotions = bool(int(sys.argv[8])) 
    else: 
        recognize_emotions = True

    dm._process_shard(videos_per_shard, shard_idx, 
        extract_audio=extract_audio,
        restore_videos=restore_videos, 
        detect_landmarks=detect_landmarks, 
        segment_videos=segment_videos, 
        reconstruct_faces=reconstruct_faces,
        recognize_emotions=recognize_emotions,
    )

    # dm._process_shard(videos_per_shard, shard_idx, 
    #     # extract_audio=True,
    #     extract_audio=False,
    #     # restore_videos=True, 
    #     restore_videos=False, 
    #     detect_landmarks=True, 
    #     # detect_landmarks=False, 
    #     segment_videos=True, 
    #     # segment_videos=False, 
    #     reconstruct_faces=False,
    #     # reconstruct_faces=True,
    # )



if __name__ == "__main__": 
    main()
