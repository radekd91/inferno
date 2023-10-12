import os, sys 
from pathlib import Path
from gdl_apps.TalkingHead.evaluation.eval_talking_head_on_audio import (create_base_sample, create_high_intensity_emotions, 
                                                                        run_evalutation, AffectNetExpressions, TalkingHeadWrapper, training_ids)
from collections import OrderedDict
import torch
import numpy as np
import librosa
import soundfile as sf


def temporal_concatenation(samples, keys_to_ignore=None):
    keys_to_ignore = keys_to_ignore or []
    sample = samples[0]
    new_sample = {}
    for key in sample.keys():
        if key in keys_to_ignore:
            new_sample[key] = sample[key]
            continue
        if isinstance(sample[key], dict):
            new_sample[key] = temporal_concatenation([s[key] for s in samples], keys_to_ignore)
            continue
        tensors = []
        for s in samples:
            tensors += [s[key]]
        new_sample[key] = np.concatenate(tensors, axis=0)
    return new_sample


def eval_talking_head_on_audio(talking_head, 
                               samples_to_emotion, 
                               samples_to_style,
                               samples_to_silent_start_ends, 
                               samples_silence, 
                               output_path=None,):
    
    all_samples = []
    original_audios = []
    silent_intervals = []
    mouth_opening_intervals = []
    mouth_closure_intervals = []
    frame_counter = 0
    num_frames_to_open_mouth = 5
    for ai, audio_path in enumerate(samples_to_emotion.keys()):
        emotion = samples_to_emotion[audio_path]
        style = samples_to_style[audio_path]
        silent_frames_start, silent_frames_end, silent_emotion_start, silent_emotion_end = samples_to_silent_start_ends[audio_path]

        silent = samples_silence[audio_path]
        sample = create_base_sample(talking_head, audio_path, 
                                    silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end, silence_all=silent)

        samples = create_high_intensity_emotions(talking_head, sample, identity_idx=style, emotion_index_list=emotion, 
                                            silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end, 
                                            silent_emotion_start = silent_emotion_start, silent_emotion_end = silent_emotion_end)
        all_samples += samples

        T = samples[0]["raw_audio"].shape[0]
        if ai == 0:
            first_silent_frames_start = frame_counter + silent_frames_start
        if ai == len(samples_to_emotion.keys()) - 2:
            last_silent_frames_start = frame_counter + T - silent_frames_end
            last_silent_frames_end = frame_counter + T 

        orig_audio, sr = librosa.load(audio_path) 
        if silent_frames_start > 0:
            orig_audio = np.concatenate([np.zeros(int(silent_frames_start * sr / 25), dtype=orig_audio.dtype), orig_audio], axis=0)
        if silent_frames_end > 0:
            orig_audio = np.concatenate([orig_audio, np.zeros(int(silent_frames_end * sr / 25 , ), dtype=orig_audio.dtype)], axis=0)
        original_audios += [orig_audio]


        if silent: 
            silent_intervals += [[frame_counter, frame_counter + T]]
            
        if silent_frames_start > 0:
            mouth_opening_start = frame_counter + silent_frames_start - num_frames_to_open_mouth
            mouth_opening_end = frame_counter + silent_frames_start
            if mouth_opening_end - mouth_opening_start > 0:
                mouth_opening_intervals += [(mouth_opening_start, mouth_opening_end)]

        if silent_frames_end > 0:
            mouth_closure_start = frame_counter + T - silent_frames_end
            mouth_closure_end = frame_counter + T - silent_frames_end + num_frames_to_open_mouth
            if mouth_closure_end - mouth_closure_start > 0:
                mouth_closure_intervals += [(mouth_closure_start, mouth_closure_end)]


        frame_counter += T
    # add the beginnings
    # silent_intervals = [(0,first_silent_frames_start-num_frames_to_open_mouth)] + silent_intervals + [(-last_silent_frames_end+num_frames_to_open_mouth, frame_counter)]
    silent_intervals = [(0,first_silent_frames_start-num_frames_to_open_mouth)] + silent_intervals + [(last_silent_frames_start+num_frames_to_open_mouth, frame_counter)]
    
    keys_to_ignore = ["output_name", "gt_shape", "gt_tex", "samplerate"]
    sample = temporal_concatenation(all_samples, keys_to_ignore=keys_to_ignore)
    original_audios = np.concatenate(original_audios, axis=0)
    sf.write("test.wav", original_audios, sr)

    run_evalutation(talking_head, [sample], audio_path, out_folder=output_path,
                    # pyrender_videos=False, 
                    pyrender_videos=True, 
                    save_meshes=True, 
                    # save_meshes=False, 
                    save_flame=True,
                    # silent_start=first_silent_frames_start, 
                    # silent_end=last_silent_frames_end, 
                    original_audios=[[original_audios, sr]], 
                    mouth_opening_intervals=mouth_opening_intervals,
                    mouth_closure_intervals=mouth_closure_intervals,
                    silent_intervals=silent_intervals,
                    )



def main(): 
    audio_path = Path('/ps/project/EmotionalFacialAnimation/emote_fastforward')
    
    samples_to_emotion = OrderedDict({ 
        audio_path / '01b-1c.wav' :             [AffectNetExpressions.Happy.value,],
        audio_path / '01b-2_gday.wav' :         [AffectNetExpressions.Contempt.value,],
        audio_path / 'exactly.wav' :            [AffectNetExpressions.Neutral.value,],
        audio_path / '03_true_blue.wav':        [AffectNetExpressions.Sad.value,], 
        audio_path / 'right.wav':               [AffectNetExpressions.Neutral.value,], 
        # audio_path / '04_oi.wav':             [AffectNetExpressions.Anger.value,],
        audio_path / '04b_oi.wav':              [AffectNetExpressions.Anger.value,],
        audio_path / 'stays.wav':               [AffectNetExpressions.Anger.value,],
        audio_path / '05c_fair_dinkum.wav':     [AffectNetExpressions.Surprise.value,],
        audio_path / 'interested.wav':          [AffectNetExpressions.Happy.value,],
    })

    samples_to_style =  OrderedDict({ 
        audio_path / '01b-1c.wav' :         training_ids.index("M003"),
        audio_path / '01b-2_gday.wav' :     training_ids.index("M003"),
        audio_path / 'exactly.wav' :        training_ids.index("M003"),
        audio_path / '03_true_blue.wav':    training_ids.index("M003"),
        audio_path / 'right.wav':           training_ids.index("M003"),
        # audio_path / '04_oi.wav':           training_ids.index("M003"),
        audio_path / '04b_oi.wav':           training_ids.index("M003"),
        audio_path / 'stays.wav':           training_ids.index("M003"),
        audio_path / '05c_fair_dinkum.wav': training_ids.index("M003"),
        audio_path / 'interested.wav':      training_ids.index("M003"),
    })


    # samples_to_silent_start_ends =  OrderedDict({
    #     audio_path / '01b-1c.wav' :         (30, 0, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
    #     audio_path / '01b-2_gday.wav' :     (0, 30, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
    #     audio_path / 'exactly.wav' :        (0, 0, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
    #     audio_path / '03_true_blue.wav':    (5, 5, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
    #     audio_path / 'right.wav':           (0, 0, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
    #     audio_path / '04_oi.wav':           (5, 5, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
    #     audio_path / 'stays.wav':           (0, 0, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
    #     audio_path / '05c_fair_dinkum.wav': (5, 5, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
    #     audio_path / 'interested.wav':      (0, 0, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
    # })

    samples_to_silent_start_ends =  OrderedDict({
        audio_path / '01b-1c.wav' :         (10, 0, AffectNetExpressions.Happy.value, AffectNetExpressions.Happy.value),
        audio_path / '01b-2_gday.wav' :     (0, 5, AffectNetExpressions.Happy.value, AffectNetExpressions.Happy.value),
        audio_path / 'exactly.wav' :        (0, 0, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
        audio_path / '03_true_blue.wav':    (5, 5, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
        audio_path / 'right.wav':           (0, 0, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
        # audio_path / '04_oi.wav':           (5, 5, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
        audio_path / '04b_oi.wav':          (5, 5, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
        audio_path / 'stays.wav':           (0, 0, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
        audio_path / '05c_fair_dinkum.wav': (5, 10, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
        audio_path / 'interested.wav':      (0, 0, AffectNetExpressions.Neutral.value, AffectNetExpressions.Neutral.value),
    })


    samples_to_silent_all=  OrderedDict({
        audio_path / '01b-1c.wav' :     False,
        audio_path / '01b-2_gday.wav' :     False,
        audio_path / 'exactly.wav' :        True,
        audio_path / '03_true_blue.wav':    False,
        audio_path / 'right.wav':           True,
        # audio_path / '04_oi.wav':           False,
        audio_path / '04b_oi.wav':           False,
        audio_path / 'stays.wav':           True,
        audio_path / '05c_fair_dinkum.wav': False,
        audio_path / 'interested.wav':      True,
    })

    model_path = Path("/is/cluster/work/rdanecek/talkinghead/trainings/2023_05_18_01-26-32_-6224330163499889169_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm")
    
    talking_head = TalkingHeadWrapper(model_path, render_results=False)
    # talking_head = None
    eval_talking_head_on_audio(
        talking_head, 
        samples_to_emotion, 
        samples_to_style,
        samples_to_silent_start_ends,
        samples_to_silent_all
        )
    






if __name__ == "__main__": 
    main()
