import os, sys
from pathlib import Path
import ffmpeg
import subprocess
from tqdm import auto
from skvideo.io import FFmpegReader

def main(): 
    output_fps = 25
    root_dir = Path("/ps/project/EmotionalFacialAnimation/data/mead/MEAD")
    output_dir = Path(f"/is/cluster/work/rdanecek/data/mead_{output_fps}fps/resampled_videos")

    output_dir.mkdir(parents=True, exist_ok=True)

    viewing_angles = ["front"]
    video_list = [] 
    print("Looking for files...")
    for viewing_angle in viewing_angles:
        # video_list = sorted(list(Path(self.root_dir).rglob(f'**/video/{viewing_angle}/**/**/*.mp4')))
        # find video files using bash find command (faster than python glob)
        video_list_ = sorted(subprocess.check_output(f"find {str(root_dir)} -wholename */{viewing_angle}/*/*/*.mp4", shell=True).decode("utf-8").splitlines())
        video_list_ = [Path(path).relative_to(root_dir) for path in video_list_]
        
        video_list += video_list_
    print("Done")

    num_videos = len(video_list)
    if len(sys.argv) > 1:
        videos_per_shard = int(sys.argv[1])
    else:
        videos_per_shard = 200

    if len(sys.argv) > 2:
        shard_idx = int(sys.argv[2])
    else:
        shard_idx = 96



    num_shards = num_videos // videos_per_shard + 1
    print(f"num_shards: {num_shards}")
    print(f"videos_per_shard: {videos_per_shard}")
    print(f"shard_idx: {shard_idx}")

    start_index = shard_idx * videos_per_shard
    end_index = min((shard_idx + 1) * videos_per_shard, num_videos)

    for vi in auto.tqdm(range(start_index, end_index)):
        video_path = video_list[vi]
        # print(f"Processing {video_path} ({vi+1}/{num_videos})")
        output_path = output_dir / video_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.is_file(): 
            print(f"Video already exists: {output_path}")
            continue
        try:
            stream = ffmpeg.input(str(root_dir / video_path))
            stream = ffmpeg.output(stream, str(output_path), r=output_fps, vcodec="h264", crf=17, preset="slow", pix_fmt="yuv420p")
            ffmpeg.run(stream, overwrite_output=True)
        except ffmpeg._run.Error as e:
            print(f"FfmpegError during processing {video_path}")
            continue


        # reader = FFmpegReader(str(output_path))
        # reader.getShape()[0]







if __name__ == "__main__":
    main()