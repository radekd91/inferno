from gdl_apps.TalkingHead.evaluation.eval_talking_head_on_audio import *
import glob
import os


def eval_talking_head_on_audio(talking_head, audio_path, emotion_index_list=None, output_path=None, 
                               silent_frames_start=30, 
                               silent_frames_end=30, 
                               silent_emotion_start=0, 
                               silent_emotion_end=0
                               ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    talking_head = talking_head.to(device)
    # talking_head.talking_head_model.preprocessor.to(device) # weird hack
    sample = create_base_sample(talking_head, audio_path, silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end)
    # samples = create_id_emo_int_combinations(talking_head, sample)
    # styles = ['M003', 'M009', 'M022', 'M028', 'W014', 'W028']
    styles = ['M003', 'M009', 'M022']
    style_indices = [training_ids.index(s) for s in styles]
    samples = []
    for i in style_indices:
        samples += create_high_intensity_emotions(talking_head, sample, identity_idx=i, emotion_index_list=emotion_index_list, 
                                            silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end, 
                                            silent_emotion_start = silent_emotion_start, silent_emotion_end = silent_emotion_end)
        
    run_evalutation(talking_head, samples, audio_path, out_folder=output_path, pyrender_videos=False, save_meshes=True, 
                    silent_start=silent_frames_start, silent_end=silent_frames_end)
                    
    print("Done")


def run(resume_folder, audio_folder, emotion_index_list=None, output_folder = None):
    root = "/is/cluster/work/rdanecek/talkinghead/trainings/"
    model_path = Path(root) / resume_folder  
    talking_head = TalkingHeadWrapper(model_path, render_results=False)

    ## find all files in audio_folder
    audio_files = []
    if audio_folder.is_dir():
        audio_files = sorted(list(glob.glob(str(audio_folder) + "/**/*.wav", recursive=True)))

    for audio in audio_files:
        # print("audio: ", audio)
        audio = Path(audio)
        # output_dir = Path("/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/") / \
        if output_folder is None:  
            output_dir = Path("/is/cluster/work/rdanecek/testing/enspark/ablations/") / Path(talking_head.cfg.inout.full_run_dir).name / "mturk_videos_lrs3" 
        else: 
            output_dir = Path(output_folder) / Path(talking_head.cfg.inout.full_run_dir).name / "videos" 
            
        output_dir = output_dir / audio.parents[1].name / (audio.parent.name + "/" + audio.stem)
        # output_dir = Path(talking_head.cfg.inout.full_run_dir) / "mturk_videos_lrs3" / audio.parents[1].name / (audio.parent.name + "/" + audio.stem)
        eval_talking_head_on_audio(talking_head, audio, output_path=output_dir, emotion_index_list=emotion_index_list, 
                                silent_frames_start=30, 
                                silent_frames_end=30, 
                                silent_emotion_start=0, 
                                silent_emotion_end=0,
                                   )

        chmod_cmd = f"find {str(output_dir)} -print -type d -exec chmod 775 {{}} +"
        os.system(chmod_cmd)



def main(): 
    # resume_folders = []
    # resume_folders += ["2023_05_04_13-04-51_-8462650662499054253_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_predEJ_LVm"]
    # resume_folders += ["2023_05_04_18-22-17_5674910949749447663_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]

    if len(sys.argv) > 1:
        resume_folder = sys.argv[1]
    else:
        # good model with disentanglement
        # resume_folder = "2023_05_08_20-36-09_8797431074914794141_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"
        # final models
        resume_folder = "2023_05_18_01-26-32_-6224330163499889169_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"

    if len(sys.argv) > 2:
        audio_folder = Path(sys.argv[2])
    else:
        # audio = Path('/ps/project/EmotionalFacialAnimation/data/lrs3/extracted/test/0Fi83BHQsMA/00002.mp4')
        # audio_folder = Path('/is/cluster/fast/rdanecek/data/lrs3_enspark_testing')
        # audio_folder = Path('/is/cluster/fast/rdanecek/data/lrs3_enspark_testing/test')
        # audio_folder = Path('/is/cluster/work/rdanecek/data/lrs3_enspark_testing_v2')
        # audio_folder = Path('/is/cluster/work/rdanecek/data/lrs3_enspark_testing_silence')
        audio_folder = Path('/ps/project/EmotionalFacialAnimation/emote_fastforward')
        # audio = Path('/is/cluster/fast/rdanecek/data/lrs3/processed2/audio/pretrain/0akiEFwtkyA/00031.wav')

    output_folder = None
    if len(sys.argv) > 3: 
        output_folder = Path(sys.argv[3])
    else: 
        output_folder = Path('/is/cluster/fast/rdanecek/testing/enspark/fastforward_results')

    emotion_index_list = None
    if len(sys.argv) > 3:
        emotion_index_list = [int(i) for i in sys.argv[3].split(",")]

    run(resume_folder, audio_folder, emotion_index_list=emotion_index_list, output_folder=output_folder)
    


if __name__ == "__main__":
    main()
