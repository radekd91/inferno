import torch 
import numpy as np
import cv2
from pathlib import Path
import os, sys
from skimage.io import imread, imsave


def combine_video_audio(filename_out, video_in, audio_in):
    import ffmpeg
    video = ffmpeg.input(video_in)
    audio = ffmpeg.input(audio_in)
    out = ffmpeg.output(video, audio, filename_out, vcodec='copy', acodec='aac', strict='experimental')
    out.run()


def save_video_with_audio(video_path, audio_path, video_tensor, fourcc='mp4v', fps=25):
    video_tmp_path = Path(video_path).with_suffix(".tmp.mp4") 
    save_video(video_tmp_path, video_tensor, fourcc=fourcc, fps=fps)
    ffmpeg_cmd = f"ffmpeg -i {str(video_tmp_path)} -i {audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {str(video_path)}"
    os.system(ffmpeg_cmd) 
    # remove out_video_path 
    os.remove(video_tmp_path)



def save_video(video_path, video_tensor, fourcc='mp4v', fps=25):
    if isinstance(video_tensor, torch.Tensor):
        # torch to numpy
        video_tensor_np = video_tensor.cpu().numpy()
        video_tensor_np = video_tensor_np.transpose(0,2,3,1)
    else:
        video_tensor_np = video_tensor
    # video_tensor is in( T, C, H, W)
    # convert to (T, H, W, C)
    if video_tensor_np.dtype == np.float32 or video_tensor_np.dtype == np.float64:
        video_tensor_np = video_tensor_np.clip(0,1)
        # convert to uint8
        video_tensor_np = (video_tensor_np * 255)
    video_tensor_np = video_tensor_np.astype(np.uint8)
    # # rgb to bgr
    video_tensor_np = video_tensor_np[...,::-1]
    # save to video using opencv, framerate 25
    # writer = cv2.VideoWriter(video_path, apiPreference = 0, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=25,  # warning the compression causes quite some differences
    #                             frameSize=(video_tensor_np.shape[1], video_tensor_np.shape[2]))
    # writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'png '), 25, # go lossless instead
    #                         (video_tensor_np.shape[1], video_tensor_np.shape[2])
    #                         )
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*fourcc), fps,
                            # (video_tensor_np.shape[1], video_tensor_np.shape[2])
                            (video_tensor_np.shape[2], video_tensor_np.shape[1])
                            )
    for i, frame in enumerate(video_tensor_np):
        # debug_image_path = video_path.parent / f"frame_{i:05d}.png"
        # imsave(str(debug_image_path), frame)
        writer.write(frame)
    writer.release()


def concatenate_videos(video_list, output_path, horizontal=True, with_audio=True): 
    assert len(video_list) > 0, "No videos to concatenate"
    # video_list_str = " ".join([str(video_path) for video_path in video_list])
    # output_path = Path("/is/cluster/work/rdanecek/emoca/finetune_deca/video_output") / video_name / "video_geometry_coarse_with_sound.mp4"
    # save video list into a text file
    video_list_path = Path(output_path).with_suffix(".txt")
    with open(video_list_path, "w") as f:
        f.write("\n".join([str(video_path) for video_path in video_list]))
    print("Done")
    # stack the videos and keep the audio from the first file 
    video_list_str = "-i " + " -i ".join([str(video_path) for video_path in video_list])
    filter_str = [f"[{n}:v]" for n in range(len(video_list))]
    filter_str = "".join(filter_str)
    if horizontal:
        keyword = "hstack"
    else:
        keyword = "vstack"
    if with_audio:
        audio = "-map 1:a"
    else: 
        audio = ""
    cmd = f'ffmpeg -n {video_list_str} -filter_complex "{filter_str}{keyword}=inputs={len(video_list)}[v]" -map "[v]" {audio} {output_path}'
    print(cmd)
    os.system(cmd)
    # os.system(f"ffmpeg {video_list_str} -filter_complex hstack=inputs={2} -map 1:a {output_path}")

