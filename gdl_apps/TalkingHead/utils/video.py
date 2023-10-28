import cv2
import numpy as np
import torch

def save_video(video_path, video_tensor, fourcc='png ', fps=25):
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
                            (video_tensor_np.shape[1], video_tensor_np.shape[2])
                            )
    for frame in video_tensor_np:
        writer.write(frame)
    writer.release()

    # import skvideo.io as skvio
    # video_tensor_np_rec = skvio.vread(video_path)[..., ::-1]
    # diff_ = video_tensor_np_rec.astype(np.float64) - video_tensor_np.astype(np.float64)
    # diff_abs = np.abs(diff_)  ## this seems to check out with the png config

    # # to uint 8
    # diff_abs = (diff_abs).astype(np.uint8)
    # # bgr to rgb
    # diff_abs = diff_abs[..., ::-1] 
    # # save to video using opencv, framerate 25
    # writer = cv2.VideoWriter(video_path.replace(".mp4", "_diff.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 25, 
    #                             (diff_abs.shape[1], diff_abs.shape[2]))
    # for frame in diff_abs:
    #     writer.write(frame)
    # writer.release()
    # return
