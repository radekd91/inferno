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
from gdl.datasets.MEADDataModule import MEADDataModule 
import numpy as np


def main(): 
    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/celebvhq/auto_processed")
    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/mead/MEAD")
    # root_dir = Path("/is/cluster/work/rdanecek/data/mead_25fps/resampled_videos")
    root_dir = Path("/is/cluster/fast/rdanecek/data/mead_25fps/resampled_videos")
    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/celebvhq/auto_processed_online")
    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/celebvhq/auto_processed_online_25fps")
    # output_dir = Path("/is/cluster/work/rdanecek/data/mead/")
    # output_dir = Path("/is/cluster/work/rdanecek/data/mead_25fps/")
    output_dir = Path("/is/cluster/fast/rdanecek/data/mead_25fps/")
    # output_dir = Path("/ps/scratch/rdanecek/data/celebvhq/")
    # output_dir = Path("/home/rdanecek/Workspace/Data/celebvhq/")

    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs2/mvlrs_v1")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs2")

    processed_subfolder = "processed"

    # Create the dataset
    dm = MEADDataModule(
            root_dir, output_dir, processed_subfolder,
            scale=1.35, # zooms out the face a little bit s.t. forehead is very likely to be visible and lower part of the chin and a little bit of the neck as well
            bb_center_shift_x=0., # in relative numbers
            bb_center_shift_y=-0.1, # in relative numbers (i.e. -0.1 for 10% shift upwards, ...)
            # processed_video_size=256,
            processed_video_size=384,
    )

    print("Create the dataloader")
    dm.prepare_data() 
    # sys.exit(0)
    # TODO: take care of these #
    # [WARNING] Video file has no audio streams! 'M041/video/front/sad/level_2/020.mp4'
    # [WARNING] Video file has no audio streams! 'M041/video/front/sad/level_2/021.mp4'
    # [WARNING] Video file has no audio streams! 'M041/video/front/sad/level_2/022.mp4'
    # [WARNING] Video file has no audio streams! 'M041/video/front/sad/level_2/023.mp4'

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
        # segment_videos = True
        segment_videos = False
    if len(sys.argv) > 7:
        detect_aligned_landmarks = bool(int(sys.argv[7]))
    else: 
        detect_aligned_landmarks = False
    if len(sys.argv) > 8:
        reconstruct_faces = bool(int(sys.argv[8])) 
    else: 
        # reconstruct_faces = False
        reconstruct_faces = True
    if len(sys.argv) > 9:
        recognize_emotions = bool(int(sys.argv[9])) 
    else: 
        recognize_emotions = False

    if len(sys.argv) > 11:
        segmentations_to_hdf5 = bool(int(sys.argv[10]))
    else:
        # segmentations_to_hdf5 = True
        segmentations_to_hdf5 = False

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
        segmentations_to_hdf5=segmentations_to_hdf5,
    )
    
    dm.setup()
    print("Setup complete")


if __name__ == "__main__": 
    main()