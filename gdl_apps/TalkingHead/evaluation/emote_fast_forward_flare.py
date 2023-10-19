"""
Author: Radek Danecek
Copyright (c) 2023, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emote@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

from gdl_apps.TalkingHead.evaluation.TalkingHeadWrapper import TalkingHeadWrapper
from gdl_apps.TalkingHead.evaluation.eval_talking_head_on_audio import *


def eval_talking_head_on_audio(talking_head, audio_path, silent_frames_start=0, silent_frames_end=0, 
    silent_emotion_start = 0, silent_emotion_end = 0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    talking_head = talking_head.to(device)
    # talking_head.talking_head_model.preprocessor.to(device) # weird hack
    sample = create_base_sample(talking_head, audio_path, silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end)
    # samples = create_id_emo_int_combinations(talking_head, sample)
    emo_split = Path(audio_path).stem.split("_")
    if len(emo_split) > 1:
        emo = [1]
        emo = emo[0].upper() + emo[1:]
        emo_idx = [AffectNetExpressions[emo].value]
    else: 
        emo_idx = list(range(8))
    samples = create_high_intensity_emotions(talking_head, sample, 
                                             emotion_index_list = emo_idx,
                                             silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end, 
                                            silent_emotion_start = silent_emotion_start, silent_emotion_end = silent_emotion_end)

    num_frames_to_open_mouth = 5
    silent_intervals = [(0,silent_frames_start-num_frames_to_open_mouth),(-silent_frames_end+num_frames_to_open_mouth, -1)]
    manual_mouth_openting_intervals = [(silent_frames_start-num_frames_to_open_mouth, silent_frames_start)]
    manual_mouth_closure_intervals = [(-silent_frames_end, -silent_frames_end+num_frames_to_open_mouth)]

    orig_audio, sr = librosa.load(audio_path) 
    ## prepend the silent frames
    if silent_frames_start > 0:
        orig_audio = np.concatenate([np.zeros(int(silent_frames_start * sr / 25), dtype=orig_audio.dtype), orig_audio], axis=0)
    if silent_frames_end > 0:
        orig_audio = np.concatenate([orig_audio, np.zeros(int(silent_frames_end * sr / 25 , ), dtype=orig_audio.dtype)], axis=0)
    
    orig_audios = [(orig_audio, sr)]*len(samples)


    run_evalutation(talking_head, samples, audio_path,  
                    # silent_start=silent_frames_start, silent_end=silent_frames_end, 
                    # manual_mouth_closure_start=5, manual_mouth_closure_end=5, 
                    mouth_opening_intervals=manual_mouth_openting_intervals,
                    mouth_closure_intervals=manual_mouth_closure_intervals,
                    silent_intervals=silent_intervals,
                    save_flame=False, 
                    pyrender_videos=False,
                    original_audios=orig_audios,
                    )
    print("Done")


def main(): 
    root = "/is/cluster/work/rdanecek/talkinghead/trainings/"

    if len(sys.argv) > 1:
        resume_folder = sys.argv[1]
    else:
        # NEW MAIN MODEL 
        resume_folder = "2023_05_18_01-26-32_-6224330163499889169_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"
    if len(sys.argv) > 2:
        audio = Path(sys.argv[2])
    else:
        # audio = Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/Audio_FLARE_ShrishaBharadwaj/1_happy.wav')
        # audio = Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/Audio_FLARE_ShrishaBharadwaj/2_sad.wav')
        # audio = Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/Audio_FLARE_ShrishaBharadwaj/3_happy.wav')
        # audio = Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/Audio_FLARE_ShrishaBharadwaj/4_sad.wav')
        # audio = Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/Audio_FLARE_ShrishaBharadwaj/5_happy.wav')
        # audio = Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/Audio_FLARE_ShrishaBharadwaj/6_happy.wav')
        # audio = Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/Audio_FLARE_ShrishaBharadwaj/7_happy.wav')
        # audio = Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/Audio_FLARE_ShrishaBharadwaj/8_neutral.wav')
        
        audio = []
        audio += [Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/MJB_Audio/1.m4a')]
        audio += [Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/MJB_Audio/2.m4a')]
        audio += [Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/MJB_Audio/3.m4a')]
        audio += [Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/MJB_Audio/4.m4a')]

        audio += [Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/YZ_Audio/1.m4a')]
        audio += [Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/YZ_Audio/2.m4a')]
        audio += [Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/YZ_Audio/3.m4a')]
        audio += [Path('/ps/project/EmotionalFacialAnimation/flare_fastforward/YZ_Audio/4.m4a')]

    if not isinstance(audio, list):
        audio = [audio]

    model_path = Path(root) / resume_folder  
    talking_head = TalkingHeadWrapper(model_path, render_results=False)
    
    for a in audio:
        assert a.exists(), f"{a} does not exist"
        eval_talking_head_on_audio(talking_head, a, 
                                silent_frames_start=30, 
                                silent_frames_end=30, 
                                silent_emotion_start=0, 
                                silent_emotion_end=0,
        )
    


if __name__=="__main__": 
    main()
