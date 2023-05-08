from gdl_apps.TalkingHead.evaluation.eval_lip_reading import TalkingHeadWrapper, dict_to_device, save_video
from pathlib import Path
import librosa
import numpy as np
from gdl.utils.collate import robust_collate
import torch
import os
from gdl.utils.PyRenderMeshSequenceRenderer import PyRenderMeshSequenceRenderer
from tqdm.auto import tqdm
from gdl.datasets.AffectNetAutoDataModule import AffectNetExpressions



def create_condition(talking_head, sample, emotions=None, intensities=None, identities=None):
    # condition = []
    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_label', False): # mead GT expression label
        # T = sample["gt_vertices"].shape[1]
        # sample["gt_expression_label"] = np.array([1])
        if emotions is None:
            emotions = [AffectNetExpressions.Neutral.value]
        sample["gt_expression_label_condition"] = torch.nn.functional.one_hot(torch.tensor(emotions), 
            num_classes=talking_head.get_num_emotions()).numpy()

    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_label', False): # mead GT expression intensity
        # T = sample["gt_vertices"].shape[1]
        if intensities is None:
            intensities = [2]
        sample["gt_expression_intensity_condition"] = torch.nn.functional.one_hot(torch.tensor(intensities), 
            num_classes=talking_head.get_num_intensities()).numpy()

    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_label', False): 
        # T = sample["gt_vertices"].shape[1]
        if identities is None:
            identities = [0]
        sample["gt_expression_identity_condition"] = torch.nn.functional.one_hot(torch.tensor(identities), 
            num_classes=talking_head.get_num_identities()).numpy()

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
    
    samples = []
    # for emo_idx in range(8):
    # for emo_idx in range(0,4):

    for identity_idx in range(0, talking_head.get_num_identities()):
        for emo_idx in range(0, talking_head.get_num_emotions()):
            for int_idx in range(0, talking_head.get_num_intensities()):
                sample_copy = sample.copy()
                sample_copy = create_condition(talking_head, sample_copy, 
                                               emotions=[emo_idx], 
                                               identities=[identity_idx], 
                                               intensities=[int_idx])
                samples.append(sample_copy)

    batch_size = 4

    template_mesh_path = Path(talking_head.cfg.model.sequence_decoder.flame.flame_lmk_embedding_path).parent / "FLAME_sample.ply"        
    renderer = PyRenderMeshSequenceRenderer(template_mesh_path)
    D = len(samples)
    BD = int(np.ceil(D / batch_size))
    training_subjects = talking_head.get_subject_labels('training')

    overwrite = False

    for bd in tqdm(range(BD)):

        samples_batch = samples[bd*batch_size:(bd+1)*batch_size]
        batch = robust_collate(samples_batch)
        batch = dict_to_device(batch, device)

        with torch.no_grad():
            batch = talking_head(batch)

        B = batch["predicted_vertices"].shape[0]
        for b in range(B):

            output_video_dir = Path(talking_head.cfg.inout.full_run_dir) / "test_videos" / (audio_path.parent.name + "_" + audio_path.stem)
            output_video_dir.mkdir(exist_ok=True, parents=True)

            intensity = batch["gt_expression_intensity_condition"][b].argmax().item()
            emotion = batch["gt_expression_label_condition"][b].argmax().item()
            identity = batch["gt_expression_identity_condition"][b].argmax().item()

            emotion = AffectNetExpressions(emotion).name
            identity = training_subjects[identity]

            suffix = f"_{identity}_{emotion}_{intensity}"

            if talking_head.render_results:
                predicted_mouth_video = batch["predicted_video"]["front"][b]


                out_video_path = output_video_dir / f"predicted_video_{suffix}.mp4"
                save_video(out_video_path, predicted_mouth_video)
                
                out_video_with_audio_path = output_video_dir / f"predicted_video_with_audio_{suffix}.mp4"

                # attach audio to video with ffmpeg

                ffmpeg_cmd = f"ffmpeg -i {out_video_path} -i {audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {out_video_with_audio_path}"
                print(ffmpeg_cmd)
                os.system(ffmpeg_cmd)


            predicted_vertices = batch["predicted_vertices"][b]
            T = predicted_vertices.shape[0]


            out_video_path = output_video_dir / f"pyrender_predicted_video_{suffix}.mp4"
            out_video_with_audio_path = output_video_dir / f"pyrender_predicted_video_with_audio_{suffix}.mp4"

            if out_video_with_audio_path.exists() and not overwrite:
                continue

            pred_images = []
            for t in tqdm(range(T)):
                pred_vertices = predicted_vertices[t].detach().cpu().view(-1,3).numpy()
                pred_image = renderer.render(pred_vertices)
                pred_images.append(pred_image)

            pred_images = np.stack(pred_images, axis=0)

            save_video(out_video_path, pred_images)

            
            ffmpeg_cmd = f"ffmpeg -y -i {out_video_path} -i {audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {out_video_with_audio_path}"
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
    # audio = Path('/is/cluster/fast/rdanecek/data/lrs3/processed2/audio/pretrain/0akiEFwtkyA/00031.wav')

    for resume_folder in resume_folders:
        model_path = Path(root) / resume_folder  

        talking_head = TalkingHeadWrapper(model_path, render_results=False)

        eval_talking_head_on_audio(talking_head, audio)
    

if __name__=="__main__": 
    main()
