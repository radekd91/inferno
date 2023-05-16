from gdl_apps.TalkingHead.evaluation.eval_talking_head_on_audio import *
import glob


def eval_talking_head_on_audio(talking_head, audio_path, output_path=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    talking_head = talking_head.to(device)
    # talking_head.talking_head_model.preprocessor.to(device) # weird hack
    sample = create_base_sample(talking_head, audio_path)
    # samples = create_id_emo_int_combinations(talking_head, sample)
    samples = create_high_intensity_emotions(talking_head, sample)
    run_evalutation(talking_head, samples, audio_path, out_folder=output_path, pyrender_videos=False, save_meshes=True)
    print("Done")




def main(): 
    root = "/is/cluster/work/rdanecek/talkinghead/trainings/"
    # resume_folders = []
    # resume_folders += ["2023_05_04_13-04-51_-8462650662499054253_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_predEJ_LVm"]
    # resume_folders += ["2023_05_04_18-22-17_5674910949749447663_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]

    if len(sys.argv) > 1:
        resume_folder = sys.argv[1]
    else:
        # good model with disentanglement
        resume_folder = "2023_05_08_20-36-09_8797431074914794141_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"

    if len(sys.argv) > 2:
        audio_folder = sys.argv[2]
    else:
        # audio = Path('/ps/project/EmotionalFacialAnimation/data/lrs3/extracted/test/0Fi83BHQsMA/00002.mp4')
        audio_folder = Path('/is/cluster/fast/rdanecek/data/lrs3_enspark_testing')
        # audio = Path('/is/cluster/fast/rdanecek/data/lrs3/processed2/audio/pretrain/0akiEFwtkyA/00031.wav')

    model_path = Path(root) / resume_folder  
    talking_head = TalkingHeadWrapper(model_path, render_results=False)

    ## find all files in audio_folder
    audio_files = []
    if audio_folder.is_dir():
        audio_files = sorted(list(glob.glob(str(audio_folder) + "/**/*.wav", recursive=True)))

    for audio in audio_files:
        # print("audio: ", audio)
        audio = Path(audio)
        output_dir = Path(talking_head.cfg.inout.full_run_dir) / "mturk_videos_lrs3" / audio.parents[1].name / (audio.parent.name + "/" + audio.stem)
        eval_talking_head_on_audio(talking_head, audio, output_path=output_dir)


if __name__ == "__main__":
    main()
