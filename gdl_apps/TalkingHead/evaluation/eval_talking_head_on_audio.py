from gdl_apps.TalkingHead.evaluation.eval_lip_reading import TalkingHeadWrapper, dict_to_device, save_video
from pathlib import Path
import librosa
import numpy as np
from gdl.utils.collate import robust_collate
import torch
import os

def create_condition(talking_head, sample):
    # condition = []
    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_label', False): # mead GT expression label
        # T = sample["gt_vertices"].shape[1]
        sample["gt_expression_label"] = np.array([1])
        # expressions = torch.nn.functional.one_hot(sample["gt_expression_label"], 
        #                                             num_classes=talking_head.cfg.n_expression).to(device=sample["gt_expression_label"].device)
        # # if expressions.ndim == 3: # B, T, num expressions
        # #     expressions = expressions.unsqueeze(1).expand(-1, T, -1)
        # if expressions.ndim == 2: # B, num expressions, missing temporal dimension -> expand
        #     expressions = expressions.unsqueeze(1).expand(-1, T, -1)
        # expressions = expressions.to(dtype=torch.float32)
        # assert expressions.ndim == 3, "Expressions must have 3 dimensions"
        # condition += [expressions]
    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_label', False): # mead GT expression intensity
        # T = sample["gt_vertices"].shape[1]
        sample["gt_expression_intensity"] = np.array([3])
        # intensities = torch.nn.functional.one_hot(sample["gt_expression_intensity"] -1, 
        #     num_classes=talking_head.cfg.n_intensities).to(device=sample["gt_expression_intensity"].device)
        # if intensities.ndim == 2: # B, num intensities, missing temporal dimension -> expand
        #     intensities = intensities.unsqueeze(1).expand(-1, T, -1)
        # intensities = intensities.to(dtype=torch.float32)
        # assert intensities.ndim == 3, "Intensities must have 3 dimensions"
        # condition += [intensities]

    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_label', False): 
        # T = sample["gt_vertices"].shape[1]
        sample["gt_expression_identity"] = np.array([0])
        # identities = torch.nn.functional.one_hot(sample["gt_expression_identity"], 
        #                                             num_classes=talking_head.cfg.n_identities).to(device=sample["gt_expression_identity"].device)
        # if identities.ndim == 2: # B, num identities, missing temporal dimension -> expand
        #     identities = identities.unsqueeze(1).expand(-1, T, -1)
        # identities = identities.to(dtype=torch.float32)
        # assert identities.ndim == 3, "Identities must have 3 dimensions"
        # condition += [identities]

    return sample


def eval_talking_head_on_audio(talking_head, audio_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    talking_head = talking_head.to(device)
    # talking_head.talking_head_model.preprocessor.to(device) # weird hack

    wavdata, sampling_rate = read_audio(audio_path)
    sample = process_audio(wavdata, sampling_rate, video_fps=25)

    T = sample["raw_audio"].shape[0]

    sample["reconstruction"] = {}
    sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]] = {} 
    sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]]["gt_exp"] = np.zeros((T, 50), dtype=np.float32)
    sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]]["gt_shape"] = np.zeros((300), dtype=np.float32)
    # sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]]["gt_shape"] = np.zeros((T, 300), dtype=np.float32)
    sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]]["gt_jaw"] = np.zeros((T, 3), dtype=np.float32)
    sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]]["gt_tex"] = np.zeros((50), dtype=np.float32)
    
    sample = create_condition(talking_head, sample)

    batch = robust_collate([sample])
    batch = dict_to_device(batch, device)
    with torch.no_grad():
        batch = talking_head(batch)

    predicted_mouth_video = batch["predicted_video"]["front"][0]
    
    out_video_path = "predicted_video.mp4"
    save_video(out_video_path, predicted_mouth_video)
    
    out_video_with_audio_path = "predicted_video_with_audio.mp4"

    # attach audio to video with ffmpeg
    # ffmpeg -i predicted_video.mp4 -i audio.wav -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 predicted_video_with_audio.mp4

    ffmpeg_cmd = f"ffmpeg -i {out_video_path} -i {audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {out_video_with_audio_path}"
    print(ffmpeg_cmd)
    os.system(ffmpeg_cmd)

    print("Done")



def read_audio(audio_path):
    sampling_rate = 16000
    wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    # wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    if wavdata.ndim > 1:
        wavdata = librosa.to_mono(wavdata)
    wavdata = (wavdata.astype(np.float64) * 32768.0).astype(np.int16)
    return wavdata, sampling_rate


def process_audio(wavdata, sampling_rate, video_fps):
    assert sampling_rate % video_fps == 0 
    wav_per_frame = sampling_rate // video_fps 

    num_frames = wavdata.shape[0] // wav_per_frame

    wavdata_ = np.zeros((num_frames, wav_per_frame), dtype=wavdata.dtype) 
    wavdata_ = wavdata_.reshape(-1)
    if wavdata.size > wavdata_.size:
        wavdata_[...] = wavdata[:wavdata_.size]
    else: 
        wavdata_[:wavdata.size] = wavdata
    wavdata_ = wavdata_.reshape((num_frames, wav_per_frame))
    # wavdata_ = wavdata_[start_frame:(start_frame + num_read_frames)] 
    # if wavdata_.shape[0] < sequence_length:
    #     # concatente with zeros
    #     wavdata_ = np.concatenate([wavdata_, 
    #         np.zeros((sequence_length - wavdata_.shape[0], wavdata_.shape[1]),
    #         dtype=wavdata_.dtype)], axis=0)
    # wavdata_ = wavdata_.astype(np.float64) / np.int16(np.iinfo(np.int16).max)

    # wavdata_ = np.zeros((sequence_length, samplerate // video_fps), dtype=wavdata.dtype)
    # wavdata_ = np.zeros((n * frames.shape[0]), dtype=wavdata.dtype)
    # wavdata_[:wavdata.shape[0]] = wavdata 
    # wavdata_ = wavdata_.reshape((frames.shape[0], -1))
    sample = {}
    sample["raw_audio"] = wavdata_ 
    sample["samplerate"] = sampling_rate
    return sample





def main(): 
    root = "/is/cluster/work/rdanecek/talkinghead/trainings/"
    resume_folders = []
    resume_folders += ["2023_05_04_13-04-51_-8462650662499054253_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_predEJ_LVm"]
    # resume_folders += ["2023_05_04_18-22-17_5674910949749447663_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]

    # audio = Path('/ps/project/EmotionalFacialAnimation/data/lrs3/extracted/test/0Fi83BHQsMA/00002.mp4')
    audio = Path('/is/cluster/fast/rdanecek/data/lrs3/processed2/audio/trainval/0af00UcTOSc/50001.wav')

    for resume_folder in resume_folders:
        model_path = Path(root) / resume_folder  

        talking_head = TalkingHeadWrapper(model_path)

        eval_talking_head_on_audio(talking_head, audio)
    

if __name__=="__main__": 
    main()
