from gdl_apps.TalkingHead.evaluation.eval_talking_head_on_audio import *


def eval_talking_head_interpolated_conditions(talking_head, audio_path, speaking_style_index=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    talking_head = talking_head.to(device)
    # talking_head.talking_head_model.preprocessor.to(device) # weird hack
    sample = create_base_sample(talking_head, audio_path)
    samples = []


    # neutral to all other emotions
    end_emotions = list(range(8))
    end_emotions.remove(AffectNetExpressions.Neutral.value)
    start_emotions = [AffectNetExpressions.Neutral.value] * len(end_emotions)

    samples += create_high_intensity_emotions(talking_head, sample)


    start_intensities = [2] * len(end_emotions)
    end_intensities = [2] * len(end_emotions)

    start_indentities = [speaking_style_index] * len(end_emotions)
    end_indentities = [speaking_style_index] * len(end_emotions)

    samples += create_emo_interpolations(talking_head, sample, 
                              start_indentities, end_indentities,
                              start_emotions, end_emotions,
                              start_intensities, end_intensities,
                              )

    # # intensity 0 to intensity 2 for all emotions 
    # start_intensities = [0] * len(end_emotions)
    # end_intensities = [2] * len(end_emotions)

    # samples += create_emo_interpolations(talking_head, sample,
    #                                         start_indentities, end_indentities,
    #                                         end_emotions, end_emotions,
    #                                         start_intensities, end_intensities,
    #                                         )
    
    # happy to sad
    start_emotions = [AffectNetExpressions.Happy.value]
    end_emotions = [AffectNetExpressions.Sad.value]
    start_intensities = [2]
    end_intensities = [2]
    start_indentities = [speaking_style_index]
    end_indentities = [speaking_style_index]

    samples += create_emo_interpolations(talking_head, sample,
                                            start_indentities, end_indentities,
                                            start_emotions, end_emotions,
                                            start_intensities, end_intensities,
                                            )
    
    start_emotions = [AffectNetExpressions.Anger.value]
    end_emotions = [AffectNetExpressions.Happy.value]
    start_intensities = [2]
    end_intensities = [2]
    start_indentities = [speaking_style_index]
    end_indentities = [speaking_style_index]

    samples += create_emo_interpolations(talking_head, sample,
                                            start_indentities, end_indentities,
                                            start_emotions, end_emotions,
                                            start_intensities, end_intensities,
                                    )

    # surprised to disgusted
    start_emotions = [AffectNetExpressions.Surprise.value]
    end_emotions = [AffectNetExpressions.Disgust.value]
    start_intensities = [2]
    end_intensities = [2]
    start_indentities = [speaking_style_index]
    end_indentities = [speaking_style_index]

    samples += create_emo_interpolations(talking_head, sample,
                                            start_indentities, end_indentities,
                                            start_emotions, end_emotions,
                                            start_intensities, end_intensities,
                                            )
    
    start_emotions = [AffectNetExpressions.Sad.value]
    end_emotions = [AffectNetExpressions.Anger.value]
    start_intensities = [2]
    end_intensities = [2]
    start_indentities = [speaking_style_index]
    end_indentities = [speaking_style_index]

    samples += create_emo_interpolations(talking_head, sample,
                                            start_indentities, end_indentities,
                                            start_emotions, end_emotions,
                                            start_intensities, end_intensities,
                                            )
    
    start_emotions = [AffectNetExpressions.Sad.value]
    end_emotions = [AffectNetExpressions.Fear.value]
    start_intensities = [2]
    end_intensities = [2]
    start_indentities = [speaking_style_index]
    end_indentities = [speaking_style_index]

    samples += create_emo_interpolations(talking_head, sample,
                                            start_indentities, end_indentities,
                                            start_emotions, end_emotions,
                                            start_intensities, end_intensities,
                                            )
    

    start_emotions = [AffectNetExpressions.Sad.value]
    end_emotions = [AffectNetExpressions.Contempt.value]
    start_intensities = [2]
    end_intensities = [2]
    start_indentities = [speaking_style_index]
    end_indentities = [speaking_style_index]

    samples += create_emo_interpolations(talking_head, sample,
                                            start_indentities, end_indentities,
                                            start_emotions, end_emotions,
                                            start_intensities, end_intensities,
                                            )




    # surprised to disgusted
    start_emotions = [AffectNetExpressions.Fear.value]
    end_emotions = [AffectNetExpressions.Happy.value]
    start_intensities = [2]
    end_intensities = [2]
    start_indentities = [speaking_style_index]
    end_indentities = [speaking_style_index]

    samples += create_emo_interpolations(talking_head, sample,
                                            start_indentities, end_indentities,
                                            start_emotions, end_emotions,
                                            start_intensities, end_intensities,
                                            )

    
    run_evalutation(talking_head, samples, audio_path, pyrender_videos=False, save_meshes=True)
    print("Done")


def create_emo_interpolations(talking_head, sample, 
                              start_identities, end_identities,
                              start_emos, end_emos,
                              start_intensities, end_intensities,
                              ):
    samples = []
    assert len(start_emos) == len(end_emos)
    assert len(start_identities) == len(end_identities)
    assert len(start_intensities) == len(end_intensities)
    assert len(start_emos) == len(start_identities)

    training_subjects = talking_head.get_subject_labels('training')
    
    for i in range(len(start_emos)):
        sample_copy = sample.copy()
        sample_1 = create_condition(talking_head, sample_copy, 
                                       emotions=[start_emos[i]], 
                                       identities=[start_identities[i]],
                                       intensities=[start_intensities[i]])
        sample_copy = sample.copy()
        sample_2 = create_condition(talking_head, sample_copy,
                                    emotions=[end_emos[i]],
                                    identities=[end_identities[i]],
                                    intensities=[end_intensities[i]])
        T = sample_1["raw_audio"].shape[0]
        sample_interpolated_lin = interpolate_condition(sample_1, sample_2, T, interpolation_type="linear")
        sample_interpolated_lin["output_name"] = create_interpolation_name(
                                start_intensities[i], end_intensities[i], 
                                start_emos[i], end_emos[i], 
                                start_identities[i], end_identities[i], 
                                training_subjects, interpolation_type="linear")
        sample_interpolated_nn = interpolate_condition(sample_1, sample_2, T, interpolation_type="nn")
        sample_interpolated_nn["output_name"] = create_interpolation_name(
                                start_intensities[i], end_intensities[i], 
                                start_emos[i], end_emos[i], 
                                start_identities[i], end_identities[i], 
                                training_subjects, interpolation_type="nn")
        # samples += [sample_1, sample_2, sample_interpolated_lin, sample_interpolated_nn]
        samples += [sample_interpolated_lin, sample_interpolated_nn]
    return samples


def run(resume_folder, audio_path, speaking_style_index):
    root = "/is/cluster/work/rdanecek/talkinghead/trainings/"
    # resume_folders = []
    # resume_folder = "2023_05_04_13-04-51_-8462650662499054253_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_predEJ_LVm"
    # resume_folders += ["2023_05_04_18-22-17_5674910949749447663_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]
    model_path = Path(root) / resume_folder  
    talking_head = TalkingHeadWrapper(model_path, render_results=False)
    eval_talking_head_interpolated_conditions(talking_head, audio_path, speaking_style_index)


def main(): 
    if len(sys.argv) > 1:
        resume_folder = sys.argv[1]
    else:
        # good model with disentanglement
        resume_folder = "2023_05_08_20-36-09_8797431074914794141_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"

    if len(sys.argv) > 2:
        audio = Path(sys.argv[2])
    else:
        # audio = Path('/ps/project/EmotionalFacialAnimation/data/lrs3/extracted/test/0Fi83BHQsMA/00002.mp4')
        audio = Path('/is/cluster/fast/rdanecek/data/lrs3/processed2/audio/trainval/0af00UcTOSc/50001.wav')
        # audio = Path('/is/cluster/fast/rdanecek/data/lrs3/processed2/audio/pretrain/0akiEFwtkyA/00031.wav')

    if len(sys.argv) > 3:
        identity_idx = int(sys.argv[3])
    else:
        identity_idx = 0
    

    print("Running with resume_folder: ", resume_folder)
    print("Running with audio: ", audio)
    print("Running with identity_idx: ", identity_idx)
    run(resume_folder, audio, identity_idx)

if __name__=="__main__": 
    main()

