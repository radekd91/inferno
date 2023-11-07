from skimage.io import imsave
from skvideo.io import vwrite
from pathlib import Path
from wandb import Image, Video
import numpy as np
import soundfile as sf
from inferno.utils.video import combine_video_audio


def _fix_image(image):
    if image.max() < 30.: #ugly hack just to find out if range is [0-1] or [0-255]
        image = image * 255.
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _log_wandb_image(path, image, caption=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _fix_image(image)
    imsave(path, image)
    if caption is not None:
        caption_file = Path(path).parent / (Path(path).stem + ".txt")
        with open(caption_file, "w") as f:
            f.write(caption)
    wandb_image = Image(str(path), caption=caption)
    return wandb_image


def _log_array_image(path, image, caption=None):
    image = _fix_image(image)
    if path is not None:
        imsave(path, image)
    return image


def _torch_image2np(torch_image):
    image = torch_image.detach().cpu().numpy()
    if len(image.shape) == 4:
        image = image.transpose([0, 2, 3, 1])
    elif len(image.shape) == 3:
        image = image.transpose([1, 2, 0])
    return image


def _log_wandb_video(path, video_frames, fps, audio=None, audio_samplerate=None, caption=None):
    _log_array_video(path, video_frames, fps, audio=audio, audio_samplerate=audio_samplerate, caption=caption)
    wandb_vid = Video(str(path), caption=caption)
    return wandb_vid


def _log_array_video(path, video_frames, fps, audio=None, audio_samplerate=None, caption=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    video_frames = _fix_image(video_frames)
    # write video with the given fps and audio (if any)
    vwrite(path, video_frames, inputdict={'-r': str(fps)}, outputdict={'-r': str(fps)})

    if audio is not None:
        audio_path = Path(path).parent / (Path(path).stem + ".wav")
        sf.write(audio_path, audio, samplerate=audio_samplerate)

        video_with_sound_path = Path(path).parent / (Path(path).stem + "_with_sound.mp4")
        combine_video_audio(str(video_with_sound_path), str(audio_path), str(path))
        # remove audio file
        audio_path.unlink()
        # remove video file without audio
        path.unlink()
        # rename video file with audio
        video_with_sound_path.rename(path)

    if caption is not None:
        caption_file = Path(path).parent / (Path(path).stem + ".txt")
        with open(caption_file, "w") as f:
            f.write(caption)

    return video_frames

