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

from inferno_apps.TalkingHead.evaluation.TalkingHeadWrapper import TalkingHeadWrapper
from inferno_apps.TalkingHead.utils.video import save_video
from inferno.datasets.FaceVideoDataModule import dict_to_device
from pathlib import Path
import librosa
import numpy as np
from inferno.utils.collate import robust_collate
import torch
import os, sys
from inferno.utils.PyRenderMeshSequenceRenderer import PyRenderMeshSequenceRenderer
from tqdm.auto import tqdm
from inferno.datasets.AffectNetAutoDataModule import AffectNetExpressions
import trimesh
import copy
import soundfile as sf
from psbody.mesh import Mesh
from inferno.utils.other import get_path_to_assets
import omegaconf


def create_condition(talking_head, sample, emotions=None, intensities=None, identities=None):
    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_label', False): # mead GT expression label
        if emotions is None:
            emotions = [AffectNetExpressions.Neutral.value]
        sample["gt_expression_label_condition"] = torch.nn.functional.one_hot(torch.tensor(emotions), 
            num_classes=talking_head.get_num_emotions()).numpy()

    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_intensity', False): # mead GT expression intensity
        if intensities is None:
            intensities = [2]
        sample["gt_expression_intensity_condition"] = torch.nn.functional.one_hot(torch.tensor(intensities), 
            num_classes=talking_head.get_num_intensities()).numpy()

    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_identity', False): 
        if identities is None:
            identities = [0]
        sample["gt_expression_identity_condition"] = torch.nn.functional.one_hot(torch.tensor(identities), 
            num_classes=talking_head.get_num_identities()).numpy()
    return sample


def interpolate_condition(sample_1, sample_2, length, interpolation_type="linear"): 
    keys_to_interpolate = ["gt_expression_label_condition", "gt_expression_intensity_condition", "gt_expression_identity_condition"]

    sample_result = copy.deepcopy(sample_1)
    for key in keys_to_interpolate:
        condition_1 = sample_1[key]
        condition_2 = sample_2[key]

        # if temporal dimension is missing, add it
        if len(condition_1.shape) == 1:
            condition_1 = np.expand_dims(condition_1, axis=0)
            # condition_1 = condition_1.unsqueeze(0)
        if len(condition_2.shape) == 1:
            condition_2 = np.expand_dims(condition_2, axis=0)
            # conditions_2 = condition_2.unsqueeze(0)

        
        # if temporal dimension 1, repeat it
        if condition_1.shape[0] == 1:
            condition_1 = condition_1.repeat(length, axis=0)
            # condition_1 = condition_1.repeat(length)
        if condition_2.shape[0] == 1:
            condition_2 = condition_2.repeat(length, axis=0)
            # condition_2 = condition_2.repeat(length)

        # interpolate
        if interpolation_type == "linear":
            # interpolate from condition_1 to condition_2 along the length
            weights = np.linspace(0, 1, length)[..., np.newaxis]
        elif interpolation_type == "nn":
            # interpolate from condition_1 to condition_2 along the length
            weights = np.linspace(0, 1, length)[..., np.newaxis]
            weights = np.round(weights)
        else:
            raise ValueError(f"Unknown interpolation type {interpolation_type}")
        
        interpolated_condition = condition_1 * (1 - weights) + condition_2 * weights
        sample_result[key] = interpolated_condition

    return sample_result



# def eval_talking_head_on_audio(talking_head, audio_path, silent_frames_start=0, silent_frames_end=0, 
#     silent_emotion_start = 0, silent_emotion_end = 0):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     talking_head = talking_head.to(device)
#     # talking_head.talking_head_model.preprocessor.to(device) # weird hack
#     sample = create_base_sample(talking_head, audio_path, silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end)
#     # samples = create_id_emo_int_combinations(talking_head, sample)
#     samples = create_high_intensity_emotions(talking_head, sample, 
#                                              silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end, 
#                                             silent_emotion_start = silent_emotion_start, silent_emotion_end = silent_emotion_end)
#     num_frames_to_open_mouth = 5
#     silent_intervals = [(0,silent_frames_start-num_frames_to_open_mouth),(-silent_frames_end+num_frames_to_open_mouth, -1)]
#     manual_mouth_openting_intervals = [(silent_frames_start-num_frames_to_open_mouth, silent_frames_start)]
#     manual_mouth_closure_intervals = [(-silent_frames_end, -silent_frames_end+num_frames_to_open_mouth)]

#     orig_audio, sr = librosa.load(audio_path) 
#     ## prepend the silent frames
#     if silent_frames_start > 0:
#         orig_audio = np.concatenate([np.zeros(int(silent_frames_start * sr / 25), dtype=orig_audio.dtype), orig_audio], axis=0)
#     if silent_frames_end > 0:
#         orig_audio = np.concatenate([orig_audio, np.zeros(int(silent_frames_end * sr / 25 , ), dtype=orig_audio.dtype)], axis=0)
    
#     orig_audios = [(orig_audio, sr)]*len(samples)

#     run_evalutation(talking_head, 
#                     samples, 
#                     audio_path,  
#                     mouth_opening_intervals=manual_mouth_openting_intervals,
#                     mouth_closure_intervals=manual_mouth_closure_intervals,
#                     silent_intervals=silent_intervals,
#                     pyrender_videos=True,
#                     save_flame=False,
#                     save_meshes=False,
#                     original_audios=orig_audios
#                     )
#     print("Done")


def create_base_sample(talking_head, audio_path, smallest_unit=1, silent_frames_start=0, silent_frames_end=0, silence_all=False):
    wavdata, sampling_rate = read_audio(audio_path)
    sample = process_audio(wavdata, sampling_rate, video_fps=25)
    # pad the audio such that it is a multiple of the smallest unit
    sample["raw_audio"] = np.pad(sample["raw_audio"], (0, smallest_unit - sample["raw_audio"].shape[0] % smallest_unit))
    if silent_frames_start > 0:
        sample["raw_audio"] = np.concatenate([np.zeros((silent_frames_start, sample["raw_audio"].shape[1]), dtype=sample["raw_audio"].dtype), sample["raw_audio"]], axis=0)
    if silent_frames_end > 0:
        sample["raw_audio"] = np.concatenate([sample["raw_audio"], np.zeros((silent_frames_end, sample["raw_audio"].shape[1]), dtype=sample["raw_audio"].dtype)], axis=0)
    if silence_all:
        sample["raw_audio"] = np.zeros_like(sample["raw_audio"])
    T = sample["raw_audio"].shape[0]
    reconstruction_type = talking_head.cfg.data.reconstruction_type[0] if isinstance(talking_head.cfg.data.reconstruction_type, (list, omegaconf.ListConfig)) else talking_head.cfg.data.reconstruction_type
    sample["reconstruction"] = {}
    sample["reconstruction"][reconstruction_type] = {} 
    sample["reconstruction"][reconstruction_type]["gt_exp"] = np.zeros((T, 50), dtype=np.float32)
    sample["reconstruction"][reconstruction_type]["gt_shape"] = np.zeros((300), dtype=np.float32)
    # sample["reconstruction"][reconstruction_type]["gt_shape"] = np.zeros((T, 300), dtype=np.float32)
    sample["reconstruction"][reconstruction_type]["gt_jaw"] = np.zeros((T, 3), dtype=np.float32)
    sample["reconstruction"][reconstruction_type]["gt_tex"] = np.zeros((50), dtype=np.float32)
    sample = create_condition(talking_head, sample)
    return sample


def create_name(int_idx, emo_idx, identity_idx, training_subjects): 
    intensity = int_idx
    emotion = emo_idx
    identity = identity_idx

    emotion = AffectNetExpressions(emotion).name
    identity = training_subjects[identity]
    suffix = f"_{identity}_{emotion}_{intensity}"
    return suffix


def create_interpolation_name(start_int_idx, end_int_idx, 
                              start_emo_idx, end_emo_idx, 
                              start_identity_idx, end_identity_idx, 
                              training_subjects, interpolation_type): 
    
    start_emotion = AffectNetExpressions(start_emo_idx).name
    end_emotion = AffectNetExpressions(end_emo_idx).name
    start_identity = training_subjects[start_identity_idx]
    end_identity = training_subjects[end_identity_idx]
    suffix = f"_{start_identity}to{end_identity}_{start_emotion}2{end_emotion}_{start_int_idx}to{end_int_idx}_{interpolation_type}"
    return suffix


def create_id_emo_int_combinations(talking_head, sample):
    samples = []
    training_subjects = talking_head.get_subject_labels('training')
    for identity_idx in range(0, talking_head.get_num_identities()):
        for emo_idx in range(0, talking_head.get_num_emotions()):
            for int_idx in range(0, talking_head.get_num_intensities()):
                sample_copy = copy.deepcopy(sample)
                sample_copy = create_condition(talking_head, sample_copy, 
                                               emotions=[emo_idx], 
                                               identities=[identity_idx], 
                                               intensities=[int_idx])

                sample_copy["output_name"] = create_name(int_idx, emo_idx, identity_idx, training_subjects)

                samples.append(sample_copy)
    return samples


def create_neutral_emotions(talking_head, sample, 
                            identity_idx=None,
                            silent_frames_start=0, silent_frames_end=0, 
                            silent_emotion_start = 0, silent_emotion_end = 0):

    return create_high_intensity_emotions(talking_head, sample, 
                                        identity_idx=identity_idx, 
                                        emotion_index_list=[0], 
                                        silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end, 
                                        silent_emotion_start = silent_emotion_start, silent_emotion_end = silent_emotion_end)
                                          

def create_high_intensity_emotions(talking_head, sample, identity_idx=None, 
                                   emotion_index_list=None, 
                                   intensity_list=None,
                                   silent_frames_start=0, silent_frames_end=0, 
                                   silent_emotion_start = 0, silent_emotion_end = 0):
    samples = []
    training_subjects = talking_head.get_subject_labels('training')
    # for identity_idx in range(0, talking_head.get_num_identities()): 
    identity_idx = identity_idx or 0
    emotion_index_list = emotion_index_list or list(range(0, talking_head.get_num_emotions()))
    for emo_idx in emotion_index_list:
        # for int_idx in range(0, talking_head.get_num_intensities()):
        if intensity_list is None:
            if emotion_index_list == [0]:
            # if emotion_index_list == 0:
                int_idx = 0
            else:
                int_idx = talking_head.get_num_intensities() - 1
            intensity_list_ = [int_idx]
        else:
            intensity_list_ = intensity_list.copy()
        
        for int_idx in intensity_list_:
            sample_copy = copy.deepcopy(sample)
            sample_copy = create_condition(talking_head, sample_copy, 
                                            emotions=[emo_idx], 
                                            identities=[identity_idx], 
                                            intensities=[int_idx])
            # if silent_frames_start > 0:
            T = sample_copy["raw_audio"].shape[0]
            for key in ["gt_expression_label_condition", "gt_expression_identity_condition", "gt_expression_intensity_condition"]:
                # cond = sample_copy["gt_expression_label_condition"]
                cond = sample_copy[key]
                if cond.shape[0] == 1:
                    cond = cond.repeat(T, axis=0)
                    if key == "gt_expression_label_condition":
                        cond[:silent_frames_start] = 0 
                        cond[:silent_frames_start, silent_emotion_start] = 1
                # sample_copy["gt_expression_label_condition"]= cond
                sample_copy[key]= cond

            # if silent_frames_end > 0:
            T = sample_copy["raw_audio"].shape[0]

            for key in ["gt_expression_label_condition", "gt_expression_identity_condition", "gt_expression_intensity_condition"]:
                # cond = sample_copy["gt_expression_label_condition"]
                cond = sample_copy[key]
                if cond.shape[0] == 1:
                    cond = cond.repeat(T, axis=0)
                    if key == "gt_expression_label_condition":
                        cond[-silent_frames_end:] = 0
                        cond[-silent_frames_end:, silent_emotion_end] = 1
                sample_copy[key] = cond
            sample_copy["output_name"] = create_name(int_idx, emo_idx, identity_idx, training_subjects)

            samples.append(sample_copy)
    return samples



def interpolate_predictions(first_expression, last_expression, first_jaw_pose, last_jaw_pose, static_frames_start, static_frames_end, num_mouth_closure_frames):
    num_interpolation_frames = num_mouth_closure_frames
    weights = torch.from_numpy( np.linspace(0, 1, num_interpolation_frames)[np.newaxis, ..., np.newaxis]).to(first_expression.device)
    ## add the static frames 
    weights = torch.cat([torch.zeros((1, static_frames_start, 1), dtype=weights.dtype, device=weights.device), weights], dim=1)
    weights = torch.cat([weights, torch.ones((1, static_frames_end, 1), dtype=weights.dtype, device=weights.device)], dim=1)
    interpolated_jaw_pose = last_jaw_pose * weights + first_jaw_pose * (1 - weights)
    interpolated_expression = last_expression * weights.repeat(1,1, 50)  + first_expression * (1 - weights.repeat(1,1, 50))
    return interpolated_expression.float(), interpolated_jaw_pose.float()


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def run_evalutation(talking_head, 
                    samples, 
                    audio_path, 
                    overwrite=False, 
                    save_meshes=False, 
                    pyrender_videos=True, 
                    save_flame=False,
                    out_folder = None, 
                    # silent_start=0, 
                    # silent_end=0, 
                    # manual_mouth_closure_start=0, 
                    mouth_opening_intervals=[(0, 0)],
                    # manual_mouth_closure_end=0, 
                    mouth_closure_intervals=[(0, 0)],
                    silent_intervals=None,
                    original_audios=None, 
                    neutral_mesh_path=None,
                    ):
    silent_intervals = silent_intervals or []
    batch_size = 1

    # pass the interpolated part through FLAME 
    flame = talking_head.talking_head_model.sequence_decoder.get_shape_model()
    mesh_suffix = ""
    if neutral_mesh_path is not None:
        neutral_mesh = Mesh(filename=str(neutral_mesh_path))
        neutral_v = torch.from_numpy( neutral_mesh.v).to(dtype=torch.float32, device=talking_head.talking_head_model.device)
        talking_head.set_neutral_mesh(neutral_v)
        mesh_suffix = "_" + Path(neutral_mesh_path).stem        
    try:
        template_mesh_path = Path(talking_head.cfg.model.sequence_decoder.flame.flame_lmk_embedding_path).parent / "FLAME_sample.ply" 
        # template_mesh_path = Path(talking_head.cfg.model.sequence_decoder.flame.flame_lmk_embedding_path).parent / "head_template.obj"   ## this one has UV 
    except AttributeError:
        template_mesh_path = get_path_to_assets() / "FLAME" / "geometry" / "FLAME_sample.ply"
        # template_mesh_path = Path("/ps/scratch/rdanecek/data/FLAME/geometry/head_template.obj") ## this one has UV 
    # obj_template_path = template_mesh_path.parent / "head_template_blender.obj"
    if not template_mesh_path.is_absolute():
        template_mesh_path = get_path_to_assets() / template_mesh_path
    obj_template_path = template_mesh_path.parent / "head_template.obj"
    # import pyvista 
    # template = pyvista.read(obj_template_path)
    template = trimesh.load_mesh(template_mesh_path, process=False)
    template_obj = trimesh.load_mesh(obj_template_path, process=False)
    template.visual = template_obj.visual

    template_obj_ps = Mesh(filename=str(obj_template_path))

    if pyrender_videos:
        renderer = PyRenderMeshSequenceRenderer(template_mesh_path)
    else:
        renderer = None
    D = len(samples)
    BD = int(np.ceil(D / batch_size))
    training_subjects = talking_head.get_subject_labels('training')
    device = talking_head.talking_head_model.device

    # samples = samples[batch_size:]
    dl = torch.utils.data.DataLoader(TestDataset(samples), batch_size=batch_size, shuffle=False, num_workers=0, 
                                     collate_fn=robust_collate)

    if out_folder is None:
        output_dir = Path(talking_head.cfg.inout.full_run_dir) / "test_videos" / (audio_path.parent.name + "_" + audio_path.stem)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = Path(out_folder)
        output_dir.mkdir(exist_ok=True, parents=True)


    for bi, batch in enumerate(tqdm(dl)):
        batch = dict_to_device(batch, device)
        with torch.no_grad():
            batch = talking_head(batch)

        for mouth_opening_interval_start, mouth_opening_interval_end in mouth_opening_intervals: 
            T = batch["predicted_jaw"].shape[1]
            if mouth_opening_interval_end < 0: 
                mouth_opening_interval_end = T + mouth_opening_interval_end
            if mouth_opening_interval_start < 0:
                mouth_opening_interval_start = T + mouth_opening_interval_start

            mouth_opening_interval_len = mouth_opening_interval_end - mouth_opening_interval_start
            assert mouth_opening_interval_len >= 0
            
            if mouth_opening_interval_len > 0: 
                # first jaw pose 
                last_jaw_pose = batch['predicted_jaw'][:,mouth_opening_interval_end]
                first_jaw_pose = torch.zeros_like(batch['predicted_jaw'][:,0])

                last_expression = batch['predicted_exp'][:,mouth_opening_interval_end] 
                first_expression = torch.zeros_like(batch['predicted_exp'][:,0])

                num_interpolation_frames = mouth_opening_interval_len
                interpolated_expression, interpolated_jaw_pose = interpolate_predictions(
                    first_expression, last_expression, first_jaw_pose, last_jaw_pose, 
                    # static_frames_start=manual_mouth_closure_start, 
                    static_frames_start=0, 
                    # static_frames_start=silent_start - manual_mouth_closure_start, 
                    # num_mouth_closure_frames=manual_mouth_closure_start, 
                    num_mouth_closure_frames=mouth_opening_interval_len, 
                    static_frames_end = 0)
                interpolated_expression = torch.zeros_like(interpolated_expression) + last_expression[:, None]

                ## add silence to the audio 
                # silence = torch.zeros((batch["raw_audio"].shape[0], num_interpolation_frames, batch["raw_audio"].shape[2]), dtype=batch["raw_audio"].dtype, device=batch["raw_audio"].device)
                # batch["raw_audio"] = torch.cat([silence, batch["raw_audio"]], dim=1)
                # batch["predicted_jaw"] = torch.cat([interpolated_jaw_pose, batch["predicted_jaw"]], dim=1)
                # batch["predicted_exp"] = torch.cat([interpolated_expression, batch["predicted_exp"]], dim=1)
                # batch["predicted_jaw"][:, :interpolated_jaw_pose.shape[1]] = interpolated_jaw_pose
                batch["predicted_jaw"][:, mouth_opening_interval_start:mouth_opening_interval_end] = interpolated_jaw_pose
                # batch["predicted_exp"][:, :interpolated_expression.shape[1]]  = interpolated_expression
                


                pose = torch.cat([torch.zeros_like(interpolated_jaw_pose),interpolated_jaw_pose], dim=-1) 
                # exp = batch["predicted_exp"][:, :interpolated_expression.shape[1]]
                exp = batch["predicted_exp"][:, mouth_opening_interval_start:mouth_opening_interval_end]
                B_, T_ = exp.shape[:2]
                exp = exp.view(B_ * T_, -1)
                pose = pose.view(B_ * T_, -1)
                shape = batch["gt_shape"]
                shape = shape[:,None, ...].repeat(1, T_, 1).contiguous().view(B_ * T_, -1)
                predicted_verts, _, _ = flame(shape, exp, pose)
                predicted_verts = predicted_verts.reshape(B_, T_, -1) 
                # batch["predicted_vertices"] = torch.cat([predicted_verts, batch["predicted_vertices"]], dim=1) 
                # batch["predicted_vertices"][:, :predicted_verts.shape[1]] = predicted_verts
                batch["predicted_vertices"][:, mouth_opening_interval_start:mouth_opening_interval_end] = predicted_verts

        
        for mouth_closure_interval_start, mouth_closure_interval_end in mouth_closure_intervals:
            T = batch["predicted_jaw"].shape[1]
            if mouth_closure_interval_end < 0: 
                mouth_closure_interval_end = T + mouth_closure_interval_end
            if mouth_closure_interval_start < 0:
                mouth_closure_interval_start = T + mouth_closure_interval_start

            mouth_closure_interval_len = mouth_closure_interval_end - mouth_closure_interval_start
            assert mouth_closure_interval_len >= 0

            if mouth_closure_interval_start > 0:
                # first_jaw_pose = batch['predicted_jaw'][:,-manual_mouth_closure_end]
                first_jaw_pose = batch['predicted_jaw'][:,mouth_closure_interval_start]
                last_jaw_pose = torch.zeros_like(batch['predicted_jaw'][:,-1])

                # first_expression = batch['predicted_exp'][:,-manual_mouth_closure_end]
                first_expression = batch['predicted_exp'][:,-mouth_closure_interval_start]
                last_expression = torch.zeros_like(batch['predicted_exp'][:,-1])

                # num_interpolation_frames = manual_mouth_closure_end
                num_interpolation_frames = mouth_closure_interval_len
                interpolated_expression, interpolated_jaw_pose = interpolate_predictions(first_expression, last_expression, first_jaw_pose, last_jaw_pose, 
                                                                                        static_frames_start=0, 
                                                                                        # num_mouth_closure_frames=manual_mouth_closure_end, 
                                                                                        num_mouth_closure_frames=mouth_closure_interval_len, 
                                                                                        # static_frames_end = silent_end - manual_mouth_closure_end
                                                                                        static_frames_end = 0
                                                                                        )
                interpolated_expression = torch.zeros_like(interpolated_expression) + first_expression[:, None]

                # batch["predicted_jaw"] = torch.cat([batch["predicted_jaw"], interpolated_jaw_pose], dim=1)
                # batch["predicted_exp"] = torch.cat([batch["predicted_exp"], interpolated_expression], dim=1)
                # batch["predicted_jaw"][:, -interpolated_jaw_pose.shape[1]:] = interpolated_jaw_pose
                batch["predicted_jaw"][:, mouth_closure_interval_start:mouth_closure_interval_end] = interpolated_jaw_pose
                # batch["predicted_exp"][:, -interpolated_expression.shape[1]:]  = interpolated_expression
                # batch["predicted_exp"][:, mouth_closure_interval_start:mouth_closure_interval_end]  = interpolated_expression

                # pass the interpolated part through FLAME
                # flame = talking_head.talking_head_model.sequence_decoder.get_shape_model()

                pose = torch.cat([torch.zeros_like(interpolated_jaw_pose), interpolated_jaw_pose], dim=-1)
                # exp = batch["predicted_exp"][:, -interpolated_expression.shape[1]:] 
                exp = batch["predicted_exp"][:, mouth_closure_interval_start:mouth_closure_interval_end]
                B_, T_ = exp.shape[:2]
                exp = exp.view(B_ * T_, -1)
                pose = pose.view(B_ * T_, -1)
                shape = batch["gt_shape"]
                shape = shape[:,None, ...].repeat(1, T_, 1).contiguous().view(B_ * T_, -1)
                predicted_verts, _, _ = flame(shape, exp, pose)
                predicted_verts = predicted_verts.reshape(B_, T_, -1)
                # batch["predicted_vertices"] = torch.cat([batch["predicted_vertices"], predicted_verts], dim=1)
                # batch["predicted_vertices"][:, -predicted_verts.shape[1]:] = predicted_verts
                batch["predicted_vertices"][:, mouth_closure_interval_start:mouth_closure_interval_end] = predicted_verts


        for silent_start, silent_end in silent_intervals:
            if silent_end - silent_start <= 0:
                continue
            batch["predicted_jaw"][:, silent_start:silent_end] = 0

            # pass the interval through FLAME
            # flame = talking_head.talking_head_model.sequence_decoder.get_shape_model()
            pose = torch.cat([torch.zeros_like(batch["predicted_jaw"][:, silent_start:silent_end] ), batch["predicted_jaw"][:, silent_start:silent_end] ], dim=-1)
            exp = batch["predicted_exp"][:, silent_start:silent_end]
            B_, T_ = exp.shape[:2]
            exp = exp.view(B_ * T_, -1)
            pose = pose.view(B_ * T_, -1)
            shape = batch["gt_shape"]
            shape = shape[:,None, ...].repeat(1, T_, 1).contiguous().view(B_ * T_, -1)
            predicted_verts, _, _ = flame(shape, exp, pose)
            predicted_verts = predicted_verts.reshape(B_, T_, -1)
            batch["predicted_vertices"][:, silent_start:silent_end] = predicted_verts

        B = batch["predicted_vertices"].shape[0]
        for b in range(B):

            if "output_name" in batch:
                suffix = batch["output_name"][b]
            else:
                try: 
                    intensity = batch["gt_expression_intensity_condition"][b].argmax().item()
                    emotion = batch["gt_expression_label_condition"][b].argmax().item()
                    identity = batch["gt_expression_identity_condition"][b].argmax().item()

                    emotion = AffectNetExpressions(emotion).name
                    identity = training_subjects[identity]
                    suffix = f"_{identity}_{emotion}_{intensity}"
                except Exception as e:
                    print(e)
                    suffix = f"_{bi * batch_size + b}"

            out_audio_path = output_dir / f"{suffix[1:]}" / f"audio.wav"
            out_audio_path.parent.mkdir(exist_ok=True, parents=True)
            if original_audios is not None:
                orig_audio, sr = original_audios[bi * batch_size + b]
            # else:
            #     orig_audio, sr = librosa.load(audio_path) 
            #     ## prepend the silent frames
            #     if silent_start > 0:
            #         orig_audio = np.concatenate([np.zeros(int(silent_start * sr / 25), dtype=orig_audio.dtype), orig_audio], axis=0)
            #     if silent_end > 0:
            #         orig_audio = np.concatenate([orig_audio, np.zeros(int(silent_end * sr / 25 , ), dtype=orig_audio.dtype)], axis=0)
                sf.write(out_audio_path, orig_audio, samplerate=sr)

            if talking_head.render_results:
                predicted_mouth_video = batch["predicted_video"]["front"][b]

                out_video_path = output_dir / f"{suffix[1:]}" / f"pytorch_video.mp4"
                save_video(out_video_path, predicted_mouth_video, fourcc="mp4v", fps=25)
                
                out_audio_path = output_dir / f"{suffix[1:]}" / f"audio.wav"
                
                if not out_audio_path.exists(): 
                    # link the audio 
                    os.symlink(audio_path, out_audio_path)


                out_video_with_audio_path = output_dir / f"{suffix[1:]}" / f"pytorch_video_with_audio_{suffix}.mp4"

                # attach audio to video with ffmpeg

                ffmpeg_cmd = f"ffmpeg -i {out_video_path} -i {out_audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {out_video_with_audio_path}"
                print(ffmpeg_cmd)
                os.system(ffmpeg_cmd)

                # delete video without audio
                if out_video_with_audio_path.exists():
                    os.remove(out_video_path)
                # os.remove(out_audio_path)

            predicted_vertices = batch["predicted_vertices"][b]
            T = predicted_vertices.shape[0]

            out_video_path = output_dir / f"{suffix[1:]}" / f"pyrender{mesh_suffix}.mp4"
            out_video_path.parent.mkdir(exist_ok=True, parents=True)
            out_video_with_audio_path = output_dir / f"{suffix[1:]}" / f"pyrender_with_audio{mesh_suffix}.mp4"

            if save_meshes: 
                mesh_folder = output_dir / f"{suffix[1:]}"  / f"meshes{mesh_suffix}"
                mesh_folder.mkdir(exist_ok=True, parents=True)
                for t in tqdm(range(T)):
                    mesh_path = mesh_folder / (f"{t:05d}" + ".obj")
                    # mesh_path = mesh_folder / (f"{t:05d}" + ".ply")
                    if not (mesh_path.exists() and not overwrite):
                        # continue
                        pred_vertices = predicted_vertices[t].detach().cpu().view(-1,3).numpy()
                        
                        
                        mesh = copy.deepcopy(template_obj_ps)
                        mesh.v = pred_vertices
                        mesh.write_obj(str(mesh_path))

                        # mesh = template.copy(deep=True)
                        # mesh.points = pred_vertices
                        # pl = pyvista.Plotter()
                        # _ = pl.add_mesh(mesh)
                        # pl.export_obj(str(mesh_path))  

                        # # mesh = trimesh.base.Trimesh(pred_vertices, template.faces)
                        # mesh.vertices = pred_vertices
                        # mesh.export(mesh_path)

            if save_flame:
                flame_folder = output_dir / f"{suffix[1:]}"  / f"flame{mesh_suffix}"
                flame_folder.mkdir(exist_ok=True, parents=True)

                flame_dict = {}
                flame_dict["shape"] = batch["gt_shape"][b].detach().cpu().numpy()
                flame_dict["expression"] = batch["predicted_exp"][b].detach().cpu().numpy()
                flame_dict["jaw_pose"] = batch["predicted_jaw"][b].detach().cpu().numpy()
                flame_dict["global_pose"] = np.zeros_like(batch["predicted_jaw"][b].detach().cpu().numpy())

                flame_path = flame_folder / (f"flame_{suffix[1:]}.pkl")
                if not( flame_path.exists() and not overwrite):
                    import pickle
                    with open(flame_path, "wb") as f:
                        pickle.dump(flame_dict, f)

                    # audio_link_path = output_dir / f"{suffix[1:]}" / "audio.wav"
                    # if not audio_link_path.exists():
                    #     sf.write(audio_link_path, orig_audio, samplerate=sr)
                        # os.symlink(audio_path, audio_link_path)

            if pyrender_videos:
                if not(out_video_with_audio_path.exists() and not overwrite):
                    # continue

                    pred_images = []
                    for t in tqdm(range(T)):
                        pred_vertices = predicted_vertices[t].detach().cpu().view(-1,3).numpy()
                        pred_image = renderer.render(pred_vertices)
                        pred_images.append(pred_image)

                    pred_images = np.stack(pred_images, axis=0)

                    save_video(out_video_path, pred_images, fourcc="mp4v", fps=25)

                    if not out_audio_path.exists(): 
                        # link the audio 
                        os.symlink(audio_path, out_audio_path)


                    ffmpeg_cmd = f"ffmpeg -y -i {out_video_path} -i {out_audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {out_video_with_audio_path}"
                    print(ffmpeg_cmd)
                    os.system(ffmpeg_cmd)

                    # delete video without audio
                    if out_video_with_audio_path.exists():
                        os.remove(out_video_path)
                    # os.remove(out_audio_path)

            # chmod_cmd = f"find {str(output_dir)} -print -type d -exec chmod 775 {{}} +"
            chmod_cmd = f"find {str(output_dir)} -type d -exec chmod 775 {{}} +"
            os.system(chmod_cmd)


def read_audio(audio_path):
    sampling_rate = 16000
    # try:
    wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    # except ValueError: 
    #     import soundfile as sf
    #     wavdata, sampling_rate = sf.read(audio_path, channels=1, samplerate=16000,dtype=np.float32, subtype='PCM_32',format="RAW",endian='LITTLE')
    # wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    if wavdata.ndim > 1:
        wavdata = librosa.to_mono(wavdata)
    wavdata = (wavdata.astype(np.float64) * 32768.0).astype(np.int16)
    # if longer than 30s cut it
    if wavdata.shape[0] > 22 * sampling_rate:
        wavdata = wavdata[:22 * sampling_rate]
        print("Audio longer than 30s, cutting it to 30s")
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
    sample = {}
    sample["raw_audio"] = wavdata_ 
    sample["samplerate"] = sampling_rate
    return sample
