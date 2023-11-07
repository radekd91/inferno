import os, sys 
import numpy as np 
from pathlib import Path
from inferno.datasets.LRS3DataModule import LRS3DataModule
# from tqdm.auto import tqdm


def main(): 
    root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs3/extracted")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs3")
    output_dir = Path("/is/cluster/work/rdanecek/data/lrs3")

    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs2/mvlrs_v1")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs2")

    # processed_subfolder = "processed"
    processed_subfolder = "processed2"
    # processed_subfolder = None

    # Create the dataset
    dm = LRS3DataModule(root_dir, output_dir, processed_subfolder,
        face_detector_threshold=0.05,
        landmarks_from=None,
        split="all", 
        )

    # Create the dataloader
    dm.prepare_data() 
    dm.setup()
    # sys.exit()

    videos_per_shard = 500 
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
        extract_audio = True
    if len(sys.argv) > 4:
        restore_videos = bool(int(sys.argv[4]))
    else: 
        restore_videos = False
    if len(sys.argv) > 5:
        detect_landmarks = bool(int(sys.argv[5]))
    else: 
        detect_landmarks = True
    if len(sys.argv) > 6:
        segment_videos = bool(int(sys.argv[6]))
    else: 
        segment_videos = True
    if len(sys.argv) > 7:
        reconstruct_faces = bool(int(sys.argv[7])) 
    else: 
        reconstruct_faces = False

    dataset = dm.training_set

    start_idx = shard_idx * videos_per_shard
    end_idx = min(start_idx + videos_per_shard, len(dataset))


    # for i in tqdm(range(start_idx, end_idx)):
    for i in range(start_idx, end_idx):
        try:
            sample = dataset._getitem(i)
            print(f"Sample lodaded {i}")
        except Exception as e:
            print(e)
            dm._process_video(i, 
                extract_audio=extract_audio,
                restore_videos=restore_videos, 
                detect_landmarks=detect_landmarks, 
                segment_videos=segment_videos, 
                reconstruct_faces=reconstruct_faces,
            )


if __name__ == "__main__": 
    main()
