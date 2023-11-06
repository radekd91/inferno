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

from pathlib import Path
import os, sys
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from inferno.datasets.MEADDataModule import MEADDataModule 
import numpy as np


def main(): 
    # if len(sys.argv) < 3:
    #     print("Usage: python resample_mead.py <downloaded_mead_folder> <output_dir> [videos_per_shard] [shard_idx]")
    #     sys.exit(0)
        
      
    if len(sys.argv) > 1:
        input_data_dir = Path(sys.argv[1])
    else: 
        input_data_dir = "/is/cluster/fast/rdanecek/data/mead_25fps/resampled_videos"    
    if len(sys.argv) > 2:
        output_data_dir = Path(sys.argv[2])
    else:
        output_data_dir = "/is/cluster/fast/rdanecek/data/mead_25fps/"

    processed_subfolder = "processed"

    # Create the dataset
    dm = MEADDataModule(
            input_data_dir, 
            output_data_dir, 
            processed_subfolder,
            scale=1.35, # zooms out the face a little bit s.t. forehead is very likely to be visible and lower part of the chin and a little bit of the neck as well
            bb_center_shift_x=0., # in relative numbers
            bb_center_shift_y=-0.1, # in relative numbers (i.e. -0.1 for 10% shift upwards, ...)
            processed_video_size=384,
    )

    print("Create the dataloader")
    dm.prepare_data() 
    # WARNING: these videos may be missing audio, but MEAD provides them also separately
    ## copy them manually later
    # 'M041/video/front/sad/level_2/020.mp4'
    # 'M041/video/front/sad/level_2/021.mp4'
    # 'M041/video/front/sad/level_2/022.mp4'
    # 'M041/video/front/sad/level_2/023.mp4'

    videos_per_shard = 200 
    shard_idx = 0
    if len(sys.argv) > 3:
        videos_per_shard = int(sys.argv[3])

    if len(sys.argv) > 4:
        shard_idx = int(sys.argv[4])

    print(videos_per_shard, shard_idx)
    print(dm._get_num_shards(videos_per_shard))
    # sys.exit(0)

    if len(sys.argv) > 5:
        extract_audio = bool(int(sys.argv[5]))
    else: 
        extract_audio = False
    if len(sys.argv) > 6:
        restore_videos = bool(int(sys.argv[6]))
    else: 
        restore_videos = False
    if len(sys.argv) > 7:
        detect_landmarks = bool(int(sys.argv[7]))
    else: 
        detect_landmarks = False
    if len(sys.argv) > 8:
        segment_videos = bool(int(sys.argv[8]))
    else: 
        # segment_videos = True
        segment_videos = False
    if len(sys.argv) > 9:
        detect_aligned_landmarks = bool(int(sys.argv[9]))
    else: 
        detect_aligned_landmarks = False
    if len(sys.argv) > 10:
        reconstruct_faces = bool(int(sys.argv[10])) 
    else: 
        # reconstruct_faces = False
        reconstruct_faces = True
    if len(sys.argv) > 11:
        recognize_emotions = bool(int(sys.argv[11])) 
    else: 
        recognize_emotions = False

    # if len(sys.argv) > 12:
    #     segmentations_to_hdf5 = bool(int(sys.argv[10]))
    # else:
    #     # segmentations_to_hdf5 = True
    #     segmentations_to_hdf5 = False

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
        # segmentations_to_hdf5=segmentations_to_hdf5,
    )
    
    dm.setup()
    print("Setup complete")


if __name__ == "__main__": 
    main()
