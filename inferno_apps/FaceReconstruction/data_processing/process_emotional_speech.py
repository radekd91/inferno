from pathlib import Path
from inferno.datasets.EmotionSpeechDataModule import EmotionalSpeechDataModule 
import numpy as np
import os, sys

def main(): 
    root_dir = Path("/ps/project/EmotionalFacialAnimation/data/EmotionalSpeech/processed/")
    output_dir = Path("/is/cluster/work/rdanecek/data/EmotionalSpeech/")
    processed_subfolder = "processed"

    # Create the dataset
    dm = EmotionalSpeechDataModule(
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

    videos_per_shard = 20 
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
        # extract_audio = True
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
        # segment_videos = False
        segment_videos = True
    if len(sys.argv) > 7:
        detect_aligned_landmarks = bool(int(sys.argv[7]))
    else: 
        # detect_aligned_landmarks = False
        detect_aligned_landmarks = True
    if len(sys.argv) > 8:
        reconstruct_faces = bool(int(sys.argv[8])) 
    else: 
        # reconstruct_faces = False
        reconstruct_faces = True
    if len(sys.argv) > 9:
        recognize_emotions = bool(int(sys.argv[9])) 
    else: 
        # recognize_emotions = False
        recognize_emotions = True

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
    )
    
    # dm.setup()
    print("Setup complete")


if __name__ == "__main__": 
    main()
