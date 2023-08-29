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


def concatenate_videos(video_list, output_path, horizontal=True, with_audio=True, overwrite=None):
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
    if overwrite is True: 
        overwrite_str = "-y"
    elif overwrite is False: 
        overwrite_str = "-n"
    else:
        overwrite_str = ""
    cmd = f'ffmpeg {overwrite_str} {video_list_str} -filter_complex "{filter_str}{keyword}=inputs={len(video_list)}[v]" -map "[v]" {audio} {output_path}'
    print(cmd)
    os.system(cmd)
    # os.system(f"ffmpeg {video_list_str} -filter_complex hstack=inputs={2} -map 1:a {output_path}")


def resample_video(input_video_file, output_video_file, fps=25, audio_file=None, overwrite=None):
    # keep the audio if any
    audio_str = ""
    if audio_file is not None:
        audio_str = f"-i {audio_file} -c:a copy"
    if overwrite is True: 
        overwrite_str = "-y"
    elif overwrite is False: 
        overwrite_str = "-n"
    else:
        overwrite_str = ""
    cmd = f"ffmpeg {overwrite_str} -i {input_video_file} -r {fps} -c:a copy {output_video_file}"
    print(cmd)
    os.system(cmd)


def customAddWeighted(src1, alpha, src2, beta, gamma=0):
    # Check if the images have the same size
    if src1.shape != src2.shape:
        raise ValueError("Input images must have the same size.")
    # Perform alpha blending
    blended_image = np.clip(src1 * alpha[:, :, np.newaxis] + src2 * beta[:, :, np.newaxis] + gamma, 0, 255).astype(np.uint8)
    return blended_image


def create_video_from_images_opencv(image_folder, video_path, framerate=30, background_color=(0, 0, 0), audio_path=None):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")]
    images.sort()

    if not images:
        print("No images found in the specified directory!")
        return

    img_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(video_path), fourcc, framerate, (width, height))

    for i in range(len(images)):
        img_path = os.path.join(image_folder, images[i])
        frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if frame.shape[2] == 4:
            # Create a background image
            background = np.zeros((height, width, 3), dtype=np.uint8)
            background[:] = background_color

            # Extract the alpha channel and normalize it
            alpha_channel = frame[:, :, 3] / 255.0

            # Blend image and background
            frame = customAddWeighted(frame[:, :, :3], alpha_channel, background, 1 - alpha_channel)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()
    # cv2.destroyAllWindows()

    # if audio_path:
    #     audio = AudioSegment.from_file(audio_path, format="mp3")
    #     audio.export(video_path, format="mp4")

def create_video_from_images(input_folder, output_video_file, fps=25, audio_file=None, background_color="black", img_fmt_str="frame_%05d.png", overwrite=None):
    # keep the audio if any
    audio_str = ""
    if audio_file is not None:
        audio_str = f"-i {audio_file} -c:a aac "
    background_str = ""
    if len(background_color) > 0:
        # find and read the first image 
        img_path = os.path.join(input_folder, img_fmt_str % 0)
        frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        height, width, _ = frame.shape 
    
        if background_color == "black":
            # background_str =   f"color=black:s={width}x{height},format=rgb24[c];[c][0]scale2ref[c][i];[c][i]overlay=format=auto:shortest=1,setsar=1"
            background_str =   f'-filter_complex "color=black:s={width}x{height},format=rgb24[c];[0:v][c]scale2ref[vid][bg];[bg][vid]overlay=format=auto:shortest=1,setsar=1"'
        elif background_color == "white":
            # background_str =   f"color=white:s={width}x{height},format=rgb24[c];[c][0]scale2ref[c][i];[c][i]overlay=format=auto:shortest=1,setsar=1"
            background_str =   f'-filter_complex "color=white:s={width}x{height},format=rgb24[c];[0:v][c]scale2ref[vid][bg];[bg][vid]overlay=format=auto:shortest=1,setsar=1"'
            
            # background_str =   f"color=white:s={width}x{height},format=rgb24[c];[0][c]scale2ref[vid][bg];[bg][vid]overlay=format=auto:shortest=1,setsar=1"
            # background_str =   f"color=white:s={width}x{height},format=rgb24[c];[0][c]scale2ref[vid][bg];[bg][vid]overlay=format=auto:shortest=1,setsar=1',"
           
            # background_str =  "[0:v]scale2ref=oh*mdar:ih[bg][i];[bg]color=white:oh*mdar[bg];[bg][i]overlay=format=auto,setsar=1"
        else: 
            # background_str = f"-vf pad=ceil(iw/2)*2:ceil(ih/2)*2:color={background_color}"
            background_str =    f'-filter_complex "color={background_color}:s={width}x{height},format=rgb24[c];[0:v][c]scale2ref[vid][bg];[bg][vid]overlay=format=auto:shortest=1,setsar=1"'
            
    # cmd = f"ffmpeg -n -framerate {fps} {audio_str} -i {str(input_folder)}/{img_fmt_str} -c:v libx264 -pix_fmt yuv420p -report {output_video_file}"
    # cmd = f"ffmpeg -n {audio_str} -i {str(input_folder)}/{img_fmt_str} -c:v libx264 -vf fps={fps} -pix_fmt yuv420p {output_video_file}"
    if overwrite is True: 
        overwrite_str = "-y"
    elif overwrite is False: 
        overwrite_str = "-n"
    else:
        overwrite_str = ""
    cmd = f"ffmpeg {overwrite_str} -framerate {fps} -i {str(input_folder)}/{img_fmt_str}  {audio_str}  {background_str}  -c:v libx264 -pix_fmt yuv420p {output_video_file}"
    print(cmd)
    os.system(cmd)


def dump_video_frames(input_video_file, output_folder): 
    # dump the videos frames into a folder, each frame is a png file
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {input_video_file} {output_folder}/frame_%05d.png"
    print(cmd)
    os.system(cmd)
