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
import argparse
from inferno_apps.TalkingHead.evaluation.TalkingHeadWrapper import TalkingHeadWrapper
from inferno_apps.TalkingHead.evaluation.evaluation_functions import *


def eval_talking_head_on_audio(
    talking_head, 
    audio_path, 
    silent_frames_start=0, 
    silent_frames_end=0, 
    silent_emotion_start = 0, 
    silent_emotion_end = 0, 
    outfolder=None,
    identity_idx=0,
    emotion_index_list=None,
    intensity_list=None,
    save_flame=False,
    save_meshes=False,
    save_videos=False,
    neutral_mesh_path=None,
    ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    talking_head = talking_head.to(device)
    # talking_head.talking_head_model.preprocessor.to(device) # weird hack
    sample = create_base_sample(talking_head, audio_path, silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end)
    # samples = create_id_emo_int_combinations(talking_head, sample)
    samples = create_high_intensity_emotions(talking_head, sample, 
                                             identity_idx=identity_idx,
                                             emotion_index_list=emotion_index_list,
                                             intensity_list=intensity_list,
                                             silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end, 
                                             silent_emotion_start = silent_emotion_start, silent_emotion_end = silent_emotion_end)
    silent_intervals = []
    if silent_frames_start > 0:
        num_frames_to_open_mouth = 5
        silent_intervals += [(0,silent_frames_start-num_frames_to_open_mouth)]
        manual_mouth_opening_intervals = [(silent_frames_start-num_frames_to_open_mouth, silent_frames_start)]
    else: 
        num_frames_to_open_mouth = 0
        manual_mouth_opening_intervals = []
    if silent_frames_end > 0:
        num_frames_to_close_mouth = 5
        silent_intervals += [(-silent_frames_end+num_frames_to_close_mouth, -1)]    
        manual_mouth_closure_intervals = [(-silent_frames_end, -silent_frames_end+num_frames_to_close_mouth)]
    else:
        num_frames_to_close_mouth = 0
        manual_mouth_closure_intervals = []
    
    orig_audio, sr = librosa.load(audio_path) 
    ## prepend the silent frames
    if silent_frames_start > 0:
        orig_audio = np.concatenate([np.zeros(int(silent_frames_start * sr / 25), dtype=orig_audio.dtype), orig_audio], axis=0)
    if silent_frames_end > 0:
        orig_audio = np.concatenate([orig_audio, np.zeros(int(silent_frames_end * sr / 25 , ), dtype=orig_audio.dtype)], axis=0)
    
    orig_audios = [(orig_audio, sr)]*len(samples)

    run_evalutation(talking_head, 
                    samples, 
                    audio_path,  
                    mouth_opening_intervals=manual_mouth_opening_intervals,
                    mouth_closure_intervals=manual_mouth_closure_intervals,
                    silent_intervals=silent_intervals,
                    original_audios=orig_audios,
                    out_folder=outfolder,
                    save_flame=save_flame,
                    save_meshes=save_meshes,
                    pyrender_videos=save_videos,
                    neutral_mesh_path=neutral_mesh_path,
                    )
    print("Done")


training_ids = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029'
                ]
# val_ids = ['M032', 'M033', 'M034', 'M035', 'W033', 'W035', 'W036'] 
# test_ids = ['M037', 'M039', 'M040', 'M041', 'M042', 'W037', 'W038', 'W040']

def main(): 
    
    parser = argparse.ArgumentParser()
    # add the input folder arg 
    parser.add_argument('--path_to_audio', type=str, default= str(get_path_to_assets() / "data/EMOTE_test_example_data/01_gday.wav"))
    parser.add_argument('--output_folder', type=str, default="results", help="Output folder to save the results to.")
    # parser.add_argument('--model_name', type=str, default='EMOTE', help='Name of the model to use.')
    parser.add_argument('--model_name', type=str, default='EMOTE_v2', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str, default=str(get_path_to_assets() / "TalkingHead/models"))
    parser.add_argument('--save_video', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_flame', type=bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=True, help="If true, output meshes will be saved.")
    parser.add_argument('--silent_frames_start', type=int, default=0, help="Number of silent frames to prepend to the audio.")
    parser.add_argument('--silent_frames_end', type=int, default=0, help="Number of silent frames to append to the audio.")
    parser.add_argument('--silent_emotion_start', type=int, default=0, help="Which emotion will be silent at the beginning of the audio.")
    parser.add_argument('--silent_emotion_end', type=int, default=0, help="Which emotion will be silent at the end of the audio.")
    parser.add_argument('--subject_style', type=str, default='M003', help=f"Which subject style to use. Styles available: \n{training_ids}")
    parser.add_argument('--neutral_mesh_path', type=str, default='', help="Path to the neutral mesh. If blank, the default FLAME mean face will be used")
    parser.add_argument('--emotion', type=str, default='all', help="The emotion to generate. One of: neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt. If 'all', all emotions will be generated.")
    parser.add_argument('--intensity', type=str, default='2', help="The emotion intentsity. One of: 0, 1, 2. If 'all', all emotions will be generated.")

    args = parser.parse_args()

    root = args.path_to_models
    resume_folder = args.model_name
    audio = args.path_to_audio
    
    model_path = Path(root) / resume_folder  
    talking_head = TalkingHeadWrapper(model_path, render_results=False)
    
    subject_id = training_ids.index(args.subject_style)
    
    if args.emotion == 'all':
        emotion_index_list = list(range(8))
    else:
        emotions = args.emotion.split(',')
        emotion_index_list = []
        for e in emotions:
            emotion_name = e[0].upper() + e[1:].lower()
            emotion_index_list += [AffectNetExpressions.index(emotion_name)]
    
    if args.intensity == 'all': 
        intensity_list = list(range(3))
    else:
        intensities = args.intensity.split(',')
        intensity_list = []
        for i in intensities:
            intensity_list += [int(i)]
    
    eval_talking_head_on_audio(
        talking_head, audio, 
        silent_frames_start=args.silent_frames_start,
        silent_frames_end=args.silent_frames_end, 
        silent_emotion_start=args.silent_emotion_start,
        silent_emotion_end=args.silent_emotion_end,
        outfolder=str(Path(args.output_folder) / args.model_name),
        identity_idx = subject_id, 
        save_flame=args.save_flame,
        save_meshes=args.save_mesh,
        save_videos=args.save_video,
        neutral_mesh_path = args.neutral_mesh_path if args.neutral_mesh_path != '' else None,
        emotion_index_list=emotion_index_list,
        intensity_list=intensity_list
    )
    


if __name__=="__main__": 
    main()
