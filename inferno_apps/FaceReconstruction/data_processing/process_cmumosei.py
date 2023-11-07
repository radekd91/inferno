import os, sys
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # enable if you get an error about file locking
from pathlib import Path
from inferno.datasets.CmuMoseiDataModule import CmuMoseiDataModule 
import numpy as np


def main(): 
    root_dir = Path("/ps/project/EmotionalFacialAnimation/data/cmu-mosei/downloaded/mosei_videos_cut_25fps")
    output_dir = Path("/is/cluster/fast/rdanecek/data/cmumosei/")


    # processed_subfolder = "processed"
    processed_subfolder = "processed2"
    processed_subfolder = "processed3"

    # Create the dataset
    dm = CmuMoseiDataModule(
            root_dir, output_dir, processed_subfolder,
            scale=1.35, # zooms out the face a little bit s.t. forehead is very likely to be visible and lower part of the chin and a little bit of the neck as well
            bb_center_shift_x=0., # in relative numbers
            bb_center_shift_y=-0.1, # in relative numbers (i.e. -0.1 for 10% shift upwards, ...)
            # processed_video_size=256,
            processed_video_size=384,
    )

    # Create the dataloader
    dm.prepare_data() 



    # videos_per_shard = 50 
    videos_per_shard = 100
    shard_idx = 0
    # shard_idx = 354
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
        # extract_audio = True
        extract_audio = False
    if len(sys.argv) > 4:
        restore_videos = bool(int(sys.argv[4]))
    else: 
        restore_videos = False
    if len(sys.argv) > 5:
        detect_landmarks = bool(int(sys.argv[5]))
    else: 
        detect_landmarks = False
        # detect_landmarks = True
    if len(sys.argv) > 6:
        segment_videos = bool(int(sys.argv[6]))
    else: 
        # segment_videos = True
        segment_videos = False
    if len(sys.argv) > 7:
        detect_aligned_landmarks = bool(int(sys.argv[5]))
    else: 
        detect_aligned_landmarks = False
        # detect_aligned_landmarks = True
    if len(sys.argv) > 8:
        reconstruct_faces = bool(int(sys.argv[7])) 
    else: 
        reconstruct_faces = False

    if len(sys.argv) > 9:
        recognize_emotions = bool(int(sys.argv[9])) 
    else: 
        recognize_emotions = False
        # recognize_emotions = True
    
    if len(sys.argv) > 10:
        create_video = bool(int(sys.argv[10]))
    else:
        # create_video = True
        create_video = False

    if len(sys.argv) > 11:
        segmentations_to_hdf5 = bool(int(sys.argv[10]))
    else:
        segmentations_to_hdf5 = True
        # segmentations_to_hdf5 = False

    dm._process_shard(
        videos_per_shard, 
        shard_idx, 
        extract_audio=extract_audio,
        restore_videos=restore_videos, 
        detect_landmarks=detect_landmarks, 
        segment_videos=segment_videos, 
        detect_aligned_landmarks=detect_aligned_landmarks,
        reconstruct_faces=reconstruct_faces,
        recognize_emotions=recognize_emotions,
        create_video=create_video,
        segmentations_to_hdf5=segmentations_to_hdf5,
    )
    
    # dm.setup()



if __name__ == "__main__": 
    main()
