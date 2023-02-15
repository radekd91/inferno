import os, sys 
from pathlib import Path
from gdl.models.video_emorec.VideoEmotionClassifier import VideoEmotionClassifier, TransformerSequenceClassifier
from gdl.layers.losses.VideoEmotionLoss import VideoEmotionRecognitionLoss, create_video_emotion_loss
# from gdl.layers.losses.EmoNetLoss import 
from gdl.models.temporal.external.LipReadingLoss import LipReadingLoss
from gdl.models.temporal.motion_prior.MotionPrior import MotionPrior
from gdl.models.temporal.motion_prior.L2lMotionPrior import L2lVqVae
from gdl.models.temporal.Preprocessors import FlamePreprocessor
from gdl.layers.losses.EmoNetLoss import create_emo_loss
from gdl.datasets.MEADPseudo3DDM import MEADPseudo3DDM
from gdl.models.DecaFLAME import FLAME, FLAMETex
from omegaconf import OmegaConf
from gdl_apps.TalkingHead.training.train_talking_head import get_condition_string_from_config
from gdl.models.temporal.BlockFactory import renderer_from_cfg
from gdl.models.temporal.TemporalFLAME import FlameShapeModel
from gdl.models.rotation_loss import compute_rotation_loss
from gdl.models.IO import get_checkpoint_with_kwargs
from skvideo.io import vwrite, vread
import torch
from munch import Munch, munchify
import datetime
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
import copy
import yaml
from gdl.datasets.AffectNetDataModule import AffectNetExpressions
from gdl.utils.other import class_from_str
from gdl_apps.MotionPrior.training.train_motion_prior import prepare_data
from gdl_apps.MotionPrior.training.training_pass import get_rendering_callback
from typing import Dict, List, Tuple, Union
from gdl.callbacks.TalkingHeadRenderingCallback import TalkingHeadTestRenderingCallback
from skimage.io import imsave
from gdl.utils.PyRenderMeshSequenceRenderer import PyRenderMeshSequenceRenderer
import subprocess


def np_matrix_to_plotly(data):
    # create a plotly figure from a numpy matrix
    import plotly.graph_objects as go
    import numpy as np
    fig = go.Figure(data=go.Heatmap(z=data))
    # # show the fig now 
    # fig.show()
    return fig


def concatenate_videos(video_list, output_path, horizontal=True, with_audio=True): 
    assert len(video_list) > 0, "No videos to concatenate"
    # video_list_str = " ".join([str(video_path) for video_path in video_list])
    # output_path = Path("/is/cluster/work/rdanecek/emoca/finetune_deca/video_output") / video_name / "video_geometry_coarse_with_sound.mp4"
    # save video list into a text file
    video_list_path = Path(output_path).with_suffix(".txt")
    with open(video_list_path, "w") as f:
        f.write("\n".join([str(video_path) for video_path in video_list]))
    # print("Done")
    # stack the videos and keep the audio from the first file 
    video_list_str = "-i " + " -i ".join([str(video_path) for video_path in video_list])
    filter_str = [f"[{n}:v]" for n in range(len(video_list))]
    filter_str = "".join(filter_str)
    if horizontal:
        keyword = "hstack"
    else:
        keyword = "vstack"
    if with_audio:
        audio = "-map 1:a"
    else: 
        audio = ""
    cmd = f'ffmpeg -hide_banner -loglevel error -n {video_list_str} -filter_complex "{filter_str}{keyword}=inputs={len(video_list)}[v]" -map "[v]" {audio} {output_path}'
    # print(cmd)
    # os.system(cmd)
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # os.system(f"ffmpeg {video_list_str} -filter_complex hstack=inputs={2} -map 1:a {output_path}")


def rename_inconsistencies(sample):
    sample['vertices'] = sample['verts'] 
    sample['jaw'] = sample['jawpose']
    sample['exp'] = sample['expcode']
    sample['shape'] = sample['shapecode']

    # del sample['verts']
    return sample


def temporal_trim_dict(sample, start_frame, end_frame, is_batched=False):
    for key in sample.keys():
        if isinstance(sample[key], np.ndarray):
            if len(sample[key].shape) >= 2 + int(is_batched):
                sample[key] = sample[key][start_frame:end_frame]
        elif isinstance(sample[key], torch.Tensor):
            if len(sample[key].shape) >= 2 + int(is_batched):
                sample[key] = sample[key][start_frame:end_frame]
        elif isinstance(sample[key], list):
            sample[key] = sample[key][start_frame:end_frame]
        elif isinstance(sample[key], dict):
            sample[key] = temporal_trim_dict(sample[key], start_frame, end_frame)
    return sample


class ExpressionRegLoss(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def prepare_target(self, target_sample):
        pass 
        
    def forward(self, sample):
        return (sample['exp'] ** 2).mean()


class TargetJawRegLoss(torch.nn.Module):

    def __init__(self, input_space, output_space) -> None:
        super().__init__()
        self.jaw_pose = None
        self.input_space = input_space 
        self.output_space = output_space

    def prepare_target(self, target_sample):
        self.jaw_pose = target_sample['jawpose']
        
    def forward(self, sample):
        return compute_rotation_loss(sample['jawpose'], self.jaw_pose, mask=None, 
        r1_input_rep=self.input_space, r2_input_rep=self.output_space)


class LipReadingTargetLoss(torch.nn.Module):

    def __init__(self, lip_reading_loss) -> None:
        super().__init__()
        self.lip_reading_loss = lip_reading_loss
        self.target_features = None
        self.camera = "front"

        self.lip_reading_loss.eval()
        for param in self.lip_reading_loss.parameters():
            param.requires_grad = False
        

    def prepare_target(self, target_sample):
        # target = rename_inconsistencies(target)
        self.target_features = self.lip_reading_loss._forward_input(target_sample["predicted_mouth_video"][self.camera])
        

    def forward(self, sample):
        assert self.target_features is not None, "You need to call prepare_target first"
        features = self.lip_reading_loss._forward_output(sample["predicted_mouth_video"][self.camera])
        return self.lip_reading_loss._compute_feature_loss(self.target_features, features)


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


class VideoEmotionTargetLoss(torch.nn.Module):

    def __init__(self, video_emotion_loss : VideoEmotionRecognitionLoss) -> None:
        super().__init__()
        self.video_emotion_loss = video_emotion_loss
        # freeze the video emotion loss
        for param in self.video_emotion_loss.parameters():
            param.requires_grad = False
        self.video_emotion_loss.eval()
        self.target_features = None
        self.target_logits = None
        self.camera = "front"

        if isinstance(video_emotion_loss.video_emotion_recognition.sequence_encoder, TransformerSequenceClassifier):
            transformer = video_emotion_loss.video_emotion_recognition.sequence_encoder.transformer_encoder.encoder
            self.save_output = SaveOutput()
            patch_attention(transformer.layers[0].self_attn)
            self.hook_handle = transformer.layers[0].self_attn.register_forward_hook(self.save_output)

    def _extract_attention(self, sample):
        if self.save_output is not None:
            predicted_attention = self.save_output.outputs[0][0]#.detach().cpu().numpy()
            self.save_output.clear()
            if 'predicted_attention' not in sample:
                sample['predicted_attention'] = {}
            sample['predicted_attention'][self.camera] = predicted_attention

    def prepare_target(self, target_per_frame_emotion_feature):
        self.target_features, self.target_logits = self.video_emotion_loss._forward_output(
            output_emotion_features=target_per_frame_emotion_feature, 
            return_logits=True
            )
        self.target_logits = torch.softmax(self.target_logits, dim=1)
        if self.save_output is not None: 
            self.save_output.clear()
        
    def forward(self, sample):
        assert self.target_features is not None, "You need to call prepare_target first"
        features, logits = self.video_emotion_loss._forward_output(sample["predicted_video"][self.camera], return_logits=True)
        if "predicted_video_emotion_logits" not in sample:
            sample["predicted_video_emotion_logits"] = {}
        sample["predicted_video_emotion_logits"][self.camera] = torch.softmax(logits, dim=1)

        self._extract_attention(sample)

        return self.video_emotion_loss._compute_feature_loss(self.target_features, features)


class GeometryTargetLoss(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        
    def prepare_target(self, target_sample):
        self.target_geometry = target_sample['gt_vertices']
        
    def forward(self, sample):
        assert self.target_geometry is not None, "You need to call prepare_target first"
        loss = torch.nn.functional.mse_loss(sample['reconstructed_vertices'], self.target_geometry)
        return loss


# class EmotionTargetLoss(torch.nn.Module):

#     def __init__(self, emotion_loss) -> None:
#         super().__init__()
#         self.emotion_loss = emotion_loss
#         for param in self.emotion_loss.parameters():
#             param.requires_grad = False
#         self.emotion_loss.eval()
#         self.target_features = None
#         self.camera = "front"

#     def prepare_target(self, target_sample):
#         target_emotion_images = target_sample["predicted_video"][self.camera]
#         B, T = target_emotion_images.shape[:2]
#         target_emotion_images = target_emotion_images.view(B*T, *target_emotion_images.shape[2:])    
#         self.target_features  = self.emotion_loss._forward_input(target_emotion_images)['emo_feat_2'].view(B, T, -1)
        
#     def forward(self, sample):
#         assert self.target_features is not None, "You need to call prepare_target first"
#         images = sample["predicted_video"][self.camera]
#         B, T = images.shape[:2]
#         images = images.view(B*T, *images.shape[2:])
#         features = self.emotion_loss._forward_output(images)['emo_feat_2'].view(B, T, -1)
#         return self.emotion_loss._compute_feature_loss(self.target_features, features)#.view(B, T, -1)


def detach_dict(sample):
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                sample[key] = value
            else:
                sample[key] = value.detach()
        elif isinstance(value, dict):
            sample[key] = detach_dict(value)
        elif isinstance(value, list):
            sample[key] = [detach_dict(v) for v in value]
        else:
            raise NotImplementedError(f"Cannot detach {type(value)}")
    return sample


def copy_dict(sample):
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample[key] = value.clone()
        elif isinstance(value, dict):
            sample[key] = copy_dict(value)
        elif isinstance(value, list):
            sample[key] = [copy_dict(v) for v in value]
        else:
            raise NotImplementedError(f"Cannot detach {type(value)}")
    return sample


def dict_to_device(sample, device):
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample[key] = value.to(device)
        elif isinstance(value, dict):
            sample[key] = dict_to_device(value, device)
        elif isinstance(value, list):
            sample[key] = [dict_to_device(v, device) for v in value]
        else:
            raise NotImplementedError(f"Cannot detach {type(value)}")
    return sample


def create_visualization_to_log(sample, prefix, output_folder, with_video=True): 
    dict_to_log = {}

    for cam in sample['predicted_video'].keys():
        if with_video:
            predicted_video = (sample['predicted_video'][cam].detach().cpu().numpy()[0] * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            video_fname = output_folder / f"{cam}_best.mp4"
            vwrite(video_fname, predicted_video)
            # video_wandb = wandb.Video(str(video_fname), fps=25, format="mp4")
            # dict_to_log[f"{prefix}_video/{cam}"] = video_wandb
        
        
        logits = sample['predicted_video_emotion_logits'][cam].detach().cpu().numpy()[0]
        for i in range(logits.shape[0]):
            dict_to_log[f"{prefix}_video_emotion/{cam}/{AffectNetExpressions(i).name}"] = logits[i]

        if "predicted_attention" in sample.keys(): 
            predicted_attention = sample['predicted_attention'][cam]
            for h in range(predicted_attention.shape[0]):
                plotly_fig = np_matrix_to_plotly(predicted_attention[h].detach().cpu().numpy())
                dict_to_log[f"{prefix}_attention/{cam}/head_{h}"] = wandb.Plotly(plotly_fig)
        #         dict_to_log[f"best_video_attention/head_{h}"] = predicted_attention[h].detach().cpu().numpy()
    return dict_to_log

class OptimizationProblem(object): 

    def __init__(self, renderer, flame, motion_prior : MotionPrior, losses, output_folder, visualization_renderer) -> None:
        self.renderer = renderer
        self.flame = flame
        self.motion_prior = motion_prior
        self.losses = losses
        self.optimizer = None
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=False)
        self.visualization_renderer = visualization_renderer
        # self.vis_rendering_frequency = 20 
        self.vis_rendering_iters = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    def forward(self, sample,):
        """
        The sample is a dictionary with the following keys
        - 'shapecode': the shape code of the flame model
        - 'expression_code': the squence of expression codes of the flame model
        - 'jaw_pose': the squence of jaw poses of the flame model
        - 'tex_code': the texture codes of the flame model
        """
        
        # 1. Quantize the latent if need be
        if self.motion_prior.motion_quantizer is not None:
            sample = self.motion_prior.quantize(sample, training_or_validation=False)
        
        # 2. Decode the latent
        sample = self.motion_prior.decode(sample)
        
        # 3. Postprocess the latent
        sample = self.motion_prior.decompose_sequential_output(sample)
        sample = self.motion_prior.postprocess(sample)
        
        # 4. Render the current values
        if self.renderer is not None:
            sample = self.flame(sample)
            # sample['vertices'] = sample['verts'] 
            # del sample['verts']
            sample = rename_inconsistencies(sample)
            sample = self.renderer(sample, input_key_prefix="", output_prefix="predicted_")
            # check_nan(sample)

        # # # render the sequence
        # # sample = self.render_sequence(sample)
        return sample

    def configure_optimizers(self, variables, optim, lr=1e-3):
        assert variables is not None, "You need to specify the variables to optimize"
        self.variables = variables
        vars_ = list(self.variables.values())
        if optim == "adam":
            self.optimizer = torch.optim.Adam(vars_, lr=lr)
        elif optim == "sgd":
            self.optimizer = torch.optim.SGD(vars_, lr=lr)
        elif optim == "lbfgs":
            self.optimizer = torch.optim.LBFGS(vars_, lr=lr)
        else: 
            raise ValueError("Unknown optimizer {}".format(optim))

    def optimize(self, init_sample, n_iter=None, patience=None): 
        sample = init_sample.copy()
        n_iter = n_iter or 100

        # vid_visualization_frequency = 10
        vid_visualization_frequency = 1
        
        best_loss = 1e10
        best_iter = 0
        best_sample = None
        iter_since_best = 0

        result_video_paths = []

        for i in range(n_iter):
            
            # # delete keys from old forward pass that correspond to the vertices and images 
            # del sample['verts']
            # del sample['predicted_video']
            # del sample['predicted_mouth_video'] 
            # del sample['predicted_landmarks2d_flame_space']
            # del sample['predicted_landmarks3d_flame_space']
            # del sample['predicted_landmarks_2d']

            sample, total_loss, losses, weighted_losses = self.optimization_step(sample)
            print(f"Iteration {i} - Total loss: {total_loss.item():.4f}")

            if total_loss < best_loss:
                best_loss = total_loss
                best_iter = i
                best_sample = copy_dict( detach_dict( sample.copy()))
                iter_since_best = 0
            else:
                iter_since_best += 1


            # if i % vid_visualization_frequency == 0:
            if i in self.vis_rendering_iters:
                if self.visualization_renderer is not None:
                    self.render_sequence(sample, self.output_folder / f"visualization_{i:06d}", label=f"iter {i:06d}")

                    # concatenate videos
                    concatenated_video_path = self.output_folder / f"visualization_{i:06d}/video_concat.mp4"
                    concatenate_videos(
                        [self.output_folder / f"visualization_{i:06d}/video.mp4", 
                        self.output_folder / f"visualization_target/video.mp4"],
                        output_path=concatenated_video_path, 
                        with_audio=False, 
                        horizontal=False
                    )
                    result_video_paths.append(concatenated_video_path)

            # dict_to_log = {}
                
            #     for cam in sample['predicted_video'].keys():
            #         predicted_video = (sample['predicted_video'][cam].detach().cpu().numpy()[0] * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            #         # predicted_mouth_video = sample['predicted_mouth_video'].detach().cpu().numpy()
            #         # write the video at 25 fps
            #         video_fname = self.output_folder / f"{cam}_iter_{i:05d}.mp4"
            #         vwrite(video_fname, predicted_video)

            #         # add the video to wandb
            #         # video_wandb = wandb.Video(str(video_fname), fps=25, format="mp4")
            #         # dict_to_log[f"video/{cam}"] = video_wandb

            # for cam in sample['predicted_video'].keys():
            #     logits = sample['predicted_video_emotion_logits'][cam].detach().cpu().numpy()[0]
            #     for j in range(logits.shape[0]):
            #         dict_to_log[f"predicted_video_emotion/{cam}/{AffectNetExpressions(j).name}"] = logits[j]

            #     if "predicted_attention" in sample.keys(): 
            #         predicted_attention = sample['predicted_attention'][cam]
            #         for h in range(predicted_attention.shape[0]):
            #             plotly_fig = np_matrix_to_plotly(predicted_attention[h].detach().cpu().numpy())
            #             dict_to_log[f"predicted_attention/{cam}/head_{h}"] = wandb.Plotly(plotly_fig)
            #     #         dict_to_log[f"predicted_attention/head_{h}"] = predicted_attention[h].detach().cpu().numpy()

            # dict_to_log = create_visualization_to_log(sample, "predicted", self.output_folder, i % vid_visualization_frequency == 0)
            dict_to_log = {}
            if i in self.vis_rendering_iters:
                video_wandb = wandb.Video(str(concatenated_video_path), fps=25, format="mp4")
                dict_to_log[f"video"] = video_wandb
            dict_to_log.update({f"loss/{k}": v.item() for k, v in losses.items()})
            dict_to_log.update({f"weighted_loss/{k}": v.item() for k, v in weighted_losses.items()})
            dict_to_log.update({"total_loss": total_loss.item()})
            wandb.log(dict_to_log, step=i)

            if patience is not None and iter_since_best > 100:
                print(f"Stopping optimization after {i} iterations because the loss did not improve for {patience} iterations.")
                break

        print(f"Best loss: {best_loss.item():.4f} at iteration {best_iter}")

        if self.visualization_renderer is not None:
            self.render_sequence(sample, self.output_folder / f"visualization_best", label=f"best")
            best_concatenated_video_path = self.output_folder / f"visualization_best/video_concat.mp4"
            concatenate_videos(
                    [self.output_folder / f"visualization_best/video.mp4", 
                    self.output_folder / f"visualization_target/video.mp4"],
                    output_path=best_concatenated_video_path, 
                    with_audio=False, 
                    horizontal=False
                )

        # dict_to_log = {}
        # for cam in sample['predicted_video'].keys():
        #     predicted_video = (best_sample['predicted_video'][cam].detach().cpu().numpy()[0] * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        #     video_fname = self.output_folder / f"{cam}_best.mp4"
        #     vwrite(video_fname, predicted_video)
        #     # video_wandb = wandb.Video(str(video_fname), fps=25, format="mp4")
        #     # dict_to_log[f"best_video/{cam}"] = video_wandb
        #     logits = sample['predicted_video_emotion_logits'][cam].detach().cpu().numpy()[0]
        #     for i in range(logits.shape[0]):
        #         dict_to_log[f"best_video_emotion/{cam}/{AffectNetExpressions(i).name}"] = logits[i]

        #     if "best_video_attention" in sample.keys(): 
        #         predicted_attention = sample['best_video_attention'][cam]
        #         for h in range(predicted_attention.shape[0]):
        #             plotly_fig = np_matrix_to_plotly(predicted_attention[h].detach().cpu().numpy())
        #             dict_to_log[f"predicted_attention/{cam}/head_{h}"] = wandb.Plotly(plotly_fig)
        #     #         dict_to_log[f"best_video_attention/head_{h}"] = predicted_attention[h].detach().cpu().numpy()

        # best_dict_to_log = create_visualization_to_log(best_sample, "best", self.output_folder)
            best_dict_to_log = {}
            video_wandb = wandb.Video(str(best_concatenated_video_path), fps=25, format="mp4")
            best_dict_to_log[f"video_best"] = video_wandb
            wandb.log(best_dict_to_log)

        # gather all the videos, concatenate them 
        concatenate_videos(
            [self.output_folder / "visualization_init/video_concat.mp4"] + \
            result_video_paths + \
            [self.output_folder / f"visualization_best/video_concat.mp4"], 
            output_path=self.output_folder / "complete_visualization.mp4",
            with_audio=False, 
            horizontal=True
        )

        return sample

    def optimization_step(self, sample):
        # 0. Compute the forward pass
        sample = self.forward(sample)
        # 1. Compute the losses
        total_loss, losses, weighted_losses = self.compute_losses(sample)
        # 2. Compute the gradients
        total_loss.backward()
        # 3. Update the parameters
        
        class Closure(object):
            
            def __init__(self, optimization_object, sample):
                self.optimization_object = optimization_object
                self.sample = sample

            def __call__(self):
                return self.forward()

            def forward(self):
                self.optimization_object.optimizer.zero_grad()
                sample = self.optimization_object.forward(self.sample)
                # 1. Compute the losses
                total_loss_, _, _ = self.optimization_object.compute_losses(sample)
                total_loss_.backward()
                return total_loss_

        # exp1 = sample['exp'].detach().clone()
        self.optimizer.step(closure=Closure(self, sample))
        # exp2 = sample['exp'].detach().clone()
        # exp3 = self.variables['expcode'].detach().clone()
        # print((exp1-exp2).sum())
        # print((exp1-exp3).sum())
        
        self.optimizer.zero_grad()    
        
        sample = detach_dict(sample)
        losses = detach_dict(losses)
        weighted_losses = detach_dict(weighted_losses)
        total_loss.detach_()


        return sample, total_loss, losses, weighted_losses

    
    def compute_losses(self, sample, allow_failure=False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        losses = {}
        for loss_name in self.losses.keys():
            if not allow_failure:
                losses[loss_name] = self.losses[loss_name].object(sample)
            else: #if allow_failure:
                try:
                    losses[loss_name] = self.losses[loss_name].object(sample)
                except:
                    print(f"Failed to compute loss {loss_name}")
                    # losses[loss_name] = torch.tensor(0.0, device=sample['image'].device)

        final_loss = 0 
        weighted_losses = {}
        for loss_name in losses.keys():
            weighted_losses[loss_name] = losses[loss_name] * self.losses[loss_name].weight
            final_loss += weighted_losses[loss_name]

        return final_loss, losses, weighted_losses


    def render_sequence(self, sample, output_folder, key="reconstructed_vertices", label=None):
        if self.visualization_renderer is None:
            return 
        output_folder.mkdir(parents=True, exist_ok=True)
        T = sample[key].shape[1]
        vertices = sample[key]
        rendering_fnames = []
        for t in range(T):
            verts = vertices[0, t].detach().cpu().view(-1,3).numpy()
            pred_image = self.visualization_renderer.render(verts)
            
            from PIL import Image, ImageFont, ImageDraw
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 25)
            # Pil image from numpy array
            img = Image.fromarray(pred_image)
            draw = ImageDraw.Draw(img)
            # draw.text((0,0), "This is a test", (255,255,0), font=font)
            
            # write frame number to the lower left corner
            draw.text((0, pred_image.shape[0]-30), f"t={t:4d}", (255,255,2550), font=font)
            # write label to the lower right corner
            if label is not None:
                draw.text((pred_image.shape[1]-200, pred_image.shape[0]-30), f"{label}", (255,255,255), font=font)

            filename = output_folder / f"{t:06d}.png"
            # imsave(filename, img) 
            img.save(filename)
            rendering_fnames += [filename]

        # compose all
        video_filename = output_folder / "video.mp4"
        framerate = 25
        start_number = 0
        image_format = "%06d"
        audio_path = None
        if audio_path is not None:
            ffmpeg_cmd = f"ffmpeg -hide_banner -loglevel error -y -framerate {framerate} -start_number {start_number} -i {str(output_folder)}/" + image_format + ".png"\
                    f" -i {str(audio_path)} -c:v libx264 -c:a aac -strict experimental -b:a 192k -pix_fmt yuv420p {str(video_filename)}"
        else:
            ffmpeg_cmd = f"ffmpeg -hide_banner -loglevel error -y -framerate {framerate} -start_number {start_number} -i {str(output_folder)}/" + image_format + ".png"\
                    f" -c:v libx264 -pix_fmt yuv420p {str(video_filename)}"
        # os.system(ffmpeg_cmd)
        res = subprocess.run(ffmpeg_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # check if the video was created
        if not video_filename.exists():
            print(f"Video {video_filename} was not created. The renderings are in {video_filename}")
            return
        
        # print(f"Video {video_filename} was created. Deleting renderings.")
        for rendering in rendering_fnames:
            os.remove(rendering)
        
                

def create_experiment_name(cfg): 
    name = "Opt_"

    # if cfg.settings.optimize_exp: 
    #     name += "E"
    # if cfg.settings.optimize_jaw_pose: 
    #     name += "J"
    name += "_"

    name += cfg.data.data_class[:4]
    name += f"-{cfg.init.geometry_type}"

    name += f"_s{cfg.init.source_sample_idx}"
    name += f"-t{cfg.init.target_sample_idx}"

    name += "_L"
    # if cfg.losses.emotion_loss.active:
    #     name += "-E" + "s" if cfg.losses.emotion_loss.from_source else "t"
    # if cfg.losses.video_emotion_loss.active:
    #     # name += "-EV"  + "s" if cfg.losses.video_emotion_loss.from_source else "t"
    #     name += "-EV"  + "s" if cfg.losses.emotion_loss.from_source else "t" # it's set on the feature loss
    # if cfg.losses.lip_reading_loss.active:
    #     name += "-LR"  + "s" if cfg.losses.lip_reading_loss.from_source else "t"
    # if cfg.losses.expression_reg.active:
    #     name += "-ExR"
    if cfg.losses.geometry_loss.active:
        name += "-G"
    return name


def optimize(cfg): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    motion_prior_net_class = class_from_str(cfg.model.pl_module_class, sys.modules[__name__])
    # instantiate the model

    model_config_path =  cfg.model.path_to_config
    with open(model_config_path, 'r') as f:
        model_config = OmegaConf.load(f)
    checkpoint_mode = 'best' # resuming in the same stage, we want to pick up where we left of
    
    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(
        model_config, "", 
        checkpoint_mode=checkpoint_mode,
        pattern="val"
        )

    motion_prior_net = motion_prior_net_class.instantiate(model_config, "", "", checkpoint, checkpoint_kwargs)
    motion_prior_net.eval()
    motion_prior_net.to(device)

    # emotion_feature_loss_cfg = cfg.losses.emotion_loss
    # emotion_loss = create_emo_loss(device, 
    #                 emoloss=emotion_feature_loss_cfg.network_path,
    #                 trainable=emotion_feature_loss_cfg.trainable, 
    #                 emo_feat_loss=emotion_feature_loss_cfg.emo_feat_loss,
    #                 normalize_features=emotion_feature_loss_cfg.normalize_features, 
    #                 dual=False).to(device)

    # ## Video emotion loss
    # # load the video emotion classifier loss
    # video_emotion_loss_cfg = cfg.losses.video_emotion_loss 
    # video_network_folder = "/is/cluster/work/rdanecek/video_emotion_recognition/trainings/"
    
    # video_emotion_loss_cfg.network_path = str(Path(video_network_folder) / video_emotion_loss_cfg.video_network_name)
    # video_emotion_loss = create_video_emotion_loss( video_emotion_loss_cfg).to(device)
    # video_emotion_loss.feature_extractor = emotion_loss.backbone

    ## Video emotion loss
    # # load the video emotion classifier loss
    # video_emotion_loss_cfg = cfg.losses.video_emotion_loss 
    # # video_network_folder = "/is/cluster/work/rdanecek/video_emotion_recognition/trainings/"
    
    # # TODO: experiment with different nets
    # # video_network_name = "2023_01_09_12-42-15_7763968562013076567_VideoEmotionClassifier_MEADP_TSC_PE_Lnce"
    # # video_emotion_loss_cfg.network_path = str(Path(video_network_folder) / video_network_name)
    # video_emotion_loss = create_video_emotion_loss( video_emotion_loss_cfg).to(device)
    # video_emotion_loss.feature_extractor = emotion_loss.backbone


    # ## Lip reading loss
    # # load the lip reading loss
    # lip_reading_loss_cfg = cfg.losses.lip_reading_loss
    # lip_reading_loss = LipReadingLoss(device, lip_reading_loss_cfg.get('metric', 'cosine_similarity')).to(device)
    

    # renderer = renderer_from_cfg(cfg.settings.renderer).to(device)


    # condition_source, condition_settings = get_condition_string_from_config(cfg)
    # instantiate a data module 
    dm, _ = prepare_data(model_config)
    if hasattr(dm, 'debug_mode'):
        dm.debug_mode = True
    # override sequence length
    dm.sequence_length_train = cfg.data.batching.sequence_length_train 
    dm.sequence_length_val = cfg.data.batching.sequence_length_val
    dm.sequence_length_test = cfg.data.batching.sequence_length_test 

    dm.prepare_data()
    dm.setup()

    # flame_cfg = cfg.settings.flame_cfg 

    # # # instantiate FLAME model
    # flame = FlameShapeModel(flame_cfg).to(device)
    # # # flame = FLAME(flame_cfg.flame)
    # # # flame_tex = FLAMETex(flame_cfg.flame_tex)

    # renderer.set_shape_model(flame.flame)

    dl = dm.val_dataloader()

    dataset = dl.dataset

    # source_sample_idx = 60 # 'M003/video/front/neutral/level_1/008.mp4'
    # target_sample_idx = 58 # 'M003/video/front/happy/level_3/024.mp4'
    source_sample_idx = cfg.init.source_sample_idx
    target_sample_idx = cfg.init.target_sample_idx

    source_sample = dataset[source_sample_idx]
    target_sample = dataset[target_sample_idx]

    # optimize_jaw_pose = cfg.settings.optimize_jaw_pose
    # optimize_exp = cfg.settings.optimize_exp
    geometry_type = cfg.init.geometry_type
    
    # get the size of the latent space 
    
    if 'reconstruction' in target_sample.keys():
        T_out = target_sample['reconstruction'][geometry_type]['gt_exp'].shape[0]
        target_rec_dict = target_sample['reconstruction'][geometry_type]
    else: 
        T_out = target_sample['gt_vertices'].shape[0]
        target_rec_dict = target_sample

    if 'reconstruction' in source_sample.keys():
        source_rec_dict = source_sample['reconstruction'][geometry_type]
        T_outs = source_sample['gt_vertices'].shape[0]
    else: 
        source_rec_dict = source_sample
        T_outs = source_rec_dict['gt_vertices'].shape[0]

    T_out = min(T_out, T_outs)
    T_latent = T_out // (2** motion_prior_net.cfg.model.sizes.quant_factor)
    T_out = T_latent * (2** motion_prior_net.cfg.model.sizes.quant_factor)
    latent_dim = motion_prior_net.get_bottleneck_dim()
    codebook_size = motion_prior_net.get_codebook_size()
    is_quanitized = codebook_size is not None

    # trim the source and target to the same length
    target_rec_dict = temporal_trim_dict(target_rec_dict, start_frame=0, end_frame=T_out)
    target_sample = temporal_trim_dict(target_sample, start_frame=0, end_frame=T_out)
    source_rec_dict = temporal_trim_dict(source_rec_dict, start_frame=0, end_frame=T_out)
    source_sample = temporal_trim_dict(source_sample, start_frame=0, end_frame=T_out)

    if is_quanitized:
        latent_seq_shape = (T_latent, latent_dim, codebook_size)
    else: 
        latent_seq_shape = (T_latent, latent_dim)

    variable_list = {}
    if cfg.init.latent_seq_init == 'source':
        # encode the source sequence
        raise NotImplementedError()
    elif cfg.init.latent_seq_init == 'zeros':
        latent_sequence = torch.zeros(latent_seq_shape, requires_grad=True, device=device)
    elif cfg.init.latent_seq_init == 'random': 
        latent_sequence = torch.randn(latent_seq_shape, requires_grad=True, device=device)
    variable_list['latentcode'] = latent_sequence

    # if optimize_exp:
    #     exp_sequence = torch.tensor(source_sample['reconstruction'][geometry_type]['gt_exp'], requires_grad=True, device=device)
    #     variable_list['expcode'] = exp_sequence
    # else:
    #     exp_sequence = source_sample['reconstruction'][geometry_type]['gt_exp'].to(device)

    # if optimize_jaw_pose:
    #     jaw_sequence = torch.tensor(source_sample['reconstruction'][geometry_type]['gt_jaw'], requires_grad=True, device=device)
    #     variable_list['jawpose'] = jaw_sequence
    # else: 
    #     jaw_sequence = source_sample['reconstruction'][geometry_type]['gt_jaw'].to(device)

    

    Tt =  target_rec_dict['gt_exp'].shape[0]
    # target_sample_ = {
    #     'expcode': target_rec_dict['gt_exp'],
    #     'jawpose': target_rec_dict['gt_jaw'],
    #     # 'pose': source_sample['reconstruction'][target_geometry]['gt_pose'],
    #     'globalpose': torch.zeros_like( target_rec_dict['gt_jaw']),
    #     # 'texcode': target_rec_dict['gt_tex'].repeat(Tt, 1).contiguous(),
    # }
    # if 'gt_shape' in target_rec_dict.keys(): 
    #     target_sample_['shapecode'] = target_rec_dict['gt_shape'][None, ...].repeat(Tt, 1).contiguous()

    target_sample_ = {
        'gt_exp': target_rec_dict['gt_exp'],
        'gt_jaw': target_rec_dict['gt_jaw'],
        # 'pose': source_sample['reconstruction'][target_geometry]['gt_pose'],
        'gt_pose': torch.zeros_like( target_rec_dict['gt_jaw']),
        # 'texcode': target_rec_dict['gt_tex'].repeat(Tt, 1).contiguous(),
    }
    if 'template' in target_rec_dict.keys():
        target_sample_['template'] = target_rec_dict['template']
    target_sample_ = dict_to_device(target_sample_, device)


    if 'gt_shape' in target_rec_dict.keys(): 
        target_sample_['gt_shape'] = target_rec_dict['gt_shape'][None, ...].repeat(Tt, 1).contiguous()

    # unsqeueze the batch dimension
    for k in target_sample_:
        target_sample_[k] = target_sample_[k].unsqueeze(0).to(device)

    if 'gt_shape' not in target_rec_dict.keys(): 
            # feed it through the flame preprocessor 
        assert isinstance(motion_prior_net.preprocessor, FlamePreprocessor)
        target_sample_ = motion_prior_net.preprocessor(target_sample_, input_key="", output_prefix="gt_")

    # # do the same for source sample
    Ts =  source_rec_dict['gt_exp'].shape[0]
    source_sample_ = {
        'encoded_features': latent_sequence,
        'gt_exp': source_rec_dict['gt_exp'],
        'gt_jaw': source_rec_dict['gt_jaw'],
        'gt_pose': torch.zeros_like( source_rec_dict['gt_jaw']),
        # 'texcode': source_sample['reconstruction'][geometry_type]['gt_tex'].repeat(Ts, 1).contiguous(),
    }
    if 'template' in source_rec_dict.keys():
        if cfg.init.shape_from_source:
            source_sample_['template'] = source_rec_dict['template']
        else:
            source_sample_['template'] = target_rec_dict['template']
    
    if cfg.init.shape_from_source: 
        if 'gt_shape' in source_rec_dict.keys(): 
            source_sample_['gt_shape'] = source_rec_dict['gt_shape'][None, ...].repeat(Ts, 1).contiguous()
    else:
        if 'gt_shape' in source_rec_dict.keys(): 
            source_sample_['gt_shape'] = target_sample_['gt_shape'].detach().clone()
    target_sample_ = dict_to_device(target_sample_, device)

    # unsqeueze the batch dimension
    for k in source_sample_:
        source_sample_[k] = source_sample_[k].unsqueeze(0).to(device)

    if cfg.init.shape_from_source:
        if 'gt_shape' not in source_rec_dict.keys(): 
            # feed it through the flame preprocessor 
            assert isinstance(motion_prior_net.preprocessor, FlamePreprocessor)
            source_sample_ = motion_prior_net.preprocessor(source_sample_, input_key="", output_prefix="gt_")
    else: 
        source_sample_['gt_shape'] = target_sample_['gt_shape'].detach().clone()
        if 'gt_vertices' not in source_sample_.keys(): 
            source_sample_ = motion_prior_net.preprocessor(source_sample_, input_key="", output_prefix="gt_")

    losses = munchify({
        # "expression_reg": {
        #     "weight": cfg.losses.expression_reg.weight,
        #     "object": ExpressionRegLoss()
        # }, 
        # "video_emotion_loss": {
        #     "weight": cfg.losses.video_emotion_loss.weight,
        #     "object": VideoEmotionTargetLoss(video_emotion_loss=video_emotion_loss)
        # }, 
        # "lip_reading_loss": {
        #     "weight": cfg.losses.lip_reading_loss.weight,
        #     "object": LipReadingTargetLoss(lip_reading_loss=lip_reading_loss)
        # },
        # "emotion_loss": {
        #     "weight": cfg.losses.emotion_loss.weight,
        #     "object": EmotionTargetLoss(emotion_loss=emotion_loss)
        # }

        "geometry_loss": {
            "weight": cfg.losses.geometry_loss.weight,
            "object": GeometryTargetLoss()
        }, 

    })

    

    # if optimize_jaw_pose:
    #     losses.jaw_pose_reg = munchify({
    #         "weight": cfg.losses.jaw_pose_reg.weight,
    #         "object": TargetJawRegLoss(cfg.losses.jaw_pose_reg.input_space, cfg.losses.jaw_pose_reg.output_space)
    #     })

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    random_id = str(hash(time))
    name = create_experiment_name(cfg)
    experiment_name = time + "_" + random_id + "_" + name
    renderer = None 
    flame = None

    flame_template_path = Path("/ps/scratch/rdanecek/data/FLAME/geometry/FLAME_sample.ply")
    # rendering_callback = get_rendering_callback(model_config, flame_template_path)

    vis_renderer = PyRenderMeshSequenceRenderer(
        flame_template_path, 
        height=600., 
        width=600.,
    )

    model = OptimizationProblem(renderer, flame, motion_prior_net, losses, 
        Path(cfg.inout.result_root) / experiment_name, vis_renderer)
    model.configure_optimizers(variable_list, cfg.optimizer.type, cfg.optimizer.lr)
    cfg.inout.experiment_name = experiment_name
    cfg.inout.result_dir = str(Path(cfg.inout.result_root) / experiment_name)
    Path(cfg.inout.result_dir).mkdir(parents=True, exist_ok=True)

    # save config into the result directory
    with open(Path(cfg.inout.result_dir) / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    wandb_logger = WandbLogger(
        project="FittingMotionPrior", 
        name=name,
        config=cfg,
        tags=cfg.inout.get('tags', None) 
    )
    wandb_logger.experiment # make sure it's initialized
    
    # with torch.no_grad():
    #     source_sample_ = model.forward(source_sample_)
    #     target_sample_ = model.forward(target_sample_)

    # if cfg.losses.lip_reading_loss.from_source:
    #     losses.lip_reading_loss.object.prepare_target(source_sample_)
    # else:
    #     losses.lip_reading_loss.object.prepare_target(target_sample_)

    # if optimize_jaw_pose:
    #     if cfg.losses.jaw_pose_reg.from_source:
    #         losses.jaw_pose_reg.object.prepare_target(source_sample_)
    #     else: 
    #         losses.jaw_pose_reg.object.prepare_target(target_sample_)
    
    # # image-based emotion loss
    # if cfg.losses.emotion_loss.from_source:
    #     losses.emotion_loss.object.prepare_target(source_sample_)
    # else:
    #     losses.emotion_loss.object.prepare_target(target_sample_)

    ## geometry loss 
    if cfg.losses.geometry_loss.from_source:
        losses.geometry_loss.object.prepare_target(source_sample_)
    else:
        losses.geometry_loss.object.prepare_target(target_sample_)


    # # emotion from target
    # if cfg.losses.video_emotion_loss.from_source:
    # losses.video_emotion_loss.object.prepare_target(target_per_frame_emotion_feature=losses.emotion_loss.object.target_features)
    # target_features = losses.emotion_loss.object.target_features.clone().detach()
    # # target_features.requires_grad = True
    # losses.video_emotion_loss.object.prepare_target(target_per_frame_emotion_feature=target_features)

    # if not cfg.losses.emotion_loss.active:
    #     losses.pop('emotion_loss')
    # if not cfg.losses.video_emotion_loss.active:
    #     losses.pop('video_emotion_loss')
    # if not cfg.losses.lip_reading_loss.active:
    #     losses.pop('lip_reading_loss')
    # if not cfg.losses.expression_reg.active:
    #     losses.pop('expression_reg')
    # if optimize_jaw_pose and not cfg.losses.jaw_pose_reg.active:
    #     losses.pop('jaw_pose_reg')
    
    # # with torch.no_grad():
    # for cam in source_sample_['predicted_video'].keys():
    #     source_sample_['predicted_video'][cam].requires_grad = True
    #     target_sample_['predicted_video'][cam].requires_grad = True
    # _, _, _ = model.compute_losses(source_sample_, allow_failure=True)
    # _, _, _ = model.compute_losses(target_sample_, allow_failure=True)
    
    source_sample_ = detach_dict(model.forward(source_sample_))
    # target_sample_ = detach_dict(model.forward(target_sample_))

    # source_sample_visdict = create_visualization_to_log(source_sample_, "source", model.output_folder)
    # target_sample_visdict = create_visualization_to_log(target_sample_, "target", model.output_folder)
    # dict_to_log = {**source_sample_visdict, **target_sample_visdict}
    # wandb.log(dict_to_log)

    model.render_sequence(source_sample_,  model.output_folder / f"visualization_init", label="init")
    model.render_sequence(target_sample_,  model.output_folder / f"visualization_target", key="gt_vertices", label="target")
    concatenated_init_video_path = model.output_folder / f"visualization_init/video_concat.mp4"
    concatenate_videos(
            [model.output_folder / f"visualization_init/video.mp4", 
            model.output_folder / f"visualization_target/video.mp4"],
            output_path=concatenated_init_video_path, 
            with_audio=False, 
            horizontal=False,
        )
    dict_to_log = {}
    dict_to_log['video_init'] = wandb.Video(str(concatenated_init_video_path), fps=25, format="mp4")
    wandb.log(dict_to_log)
    output_sample = model.optimize(source_sample_, n_iter=cfg.optimizer.n_iter, patience=cfg.optimizer.patience)


    # # gather all the videos 
    # for cam in output_sample['predicted_video'].keys():
    #     source_vid_path = model.output_folder / f"{cam}_source.mp4" 
    #     target_vid_path = model.output_folder / f"{cam}_target.mp4"
    #     vid_list = sorted(list(model.output_folder.glob(f"{cam}_iter_*.mp4")))
    #     best_vid_path =  model.output_folder / f"{cam}_best.mp4" 
    #     vid_list = [source_vid_path] + [target_vid_path] + [best_vid_path] + vid_list

    #     # concatenate the videos using ffmpeg
    #     concatenate_videos(vid_list, str(model.output_folder / f"{cam}_all.mp4"), with_audio=False)
    #     concatenate_videos([source_vid_path] + [target_vid_path] + [best_vid_path], str(model.output_folder / f"{cam}_src_target_best.mp4"), with_audio=False)
    #     # ffmpeg_cmd = f"ffmpeg -y -i 'concat:{'|'.join([str(p) for p in vid_list])}' -c copy {str(model.output_folder / f'{cam}_all.mp4')}"
    #     # os.system(ffmpeg_cmd)

    print("Optimization done")


def main(): 

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        cfg = OmegaConf.load(config_file)
        cfg = munchify(OmegaConf.to_container(cfg))
    else:
        
        cfg = Munch()
        # path_to_config = Path("/is/cluster/work/rdanecek/motion_prior/trainings/2023_02_08_14-51-54_6121455154279531419_L2lVqVae_Facef_AE/cfg.yaml")
        # path_to_config = Path("/is/cluster/work/rdanecek/motion_prior/trainings/2023_02_14_12-49-15_-845824944828324001_L2lVqVae_Facef_VAE/cfg.yaml")
        path_to_config = Path("/is/cluster/work/rdanecek/motion_prior/trainings/2023_02_14_12-49-31_-5593838506145801374_L2lVqVae_Facef_VAE/cfg.yaml")
        helper_config = OmegaConf.load(path_to_config)
        cfg.data = munchify(OmegaConf.to_container( helper_config.data))
        cfg.data.batching = Munch() 
        cfg.data.batching.batch_size_train = 1
        cfg.data.batching.batch_size_val = 1
        cfg.data.batching.batch_size_test = 1 
        # cfg.data.batching.sequence_length_train = 1
        # cfg.data.batching.sequence_length_val = 1
        # cfg.data.batching.sequence_length_test = 1

        # cfg.data.batching.sequence_length_train = 25
        # cfg.data.batching.sequence_length_val = 25
        # cfg.data.batching.sequence_length_test = 25
        cfg.data.batching.sequence_length_train = "all"
        cfg.data.batching.sequence_length_val = "all"
        cfg.data.batching.sequence_length_test = "all"

        # cfg.data.batching.sequence_length_train = 150
        # cfg.data.batching.sequence_length_val = 150
        # cfg.data.batching.sequence_length_test = 150

        cfg.model = munchify(OmegaConf.to_container( helper_config.model))
        cfg.model.path_to_config = str(path_to_config)

        cfg.losses = Munch() 

        # ## Emotion feature loss
        # emotion_feature_loss_cfg = {
        #     "weight": 1.0,
        #     "network_path": "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_05-15-38_-8198495972451127810_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early",
        #     # emo_feat_loss: mse_loss
        #     "emo_feat_loss": "masked_mse_loss",
        #     "trainable": False, 
        #     "normalize_features": False, 
        #     "target_method_image": "emoca",
        #     "mask_invalid": "mediapipe_landmarks", # frames with invalid mediapipe landmarks will be masked for loss computation
        # }
        # cfg.losses.emotion_loss = Munch(emotion_feature_loss_cfg)
        # cfg.losses.emotion_loss.from_source = False
        # cfg.losses.emotion_loss.active = False
        # # cfg.losses.emotion_loss.active = True
        # # cfg.losses.emotion_loss.weight = 1.0
        # # cfg.losses.emotion_loss.weight = 15.0
        # # cfg.losses.emotion_loss.weight = 50.0
        # cfg.losses.emotion_loss.weight = 150.0

        ## geometry loss 
        cfg.losses.geometry_loss = Munch()
        cfg.losses.geometry_loss.from_source = False
        cfg.losses.geometry_loss.active = True
        cfg.losses.geometry_loss.weight = 1000000.

        # cfg.losses.video_emotion_loss = Munch()
        # # TODO: experiment with different nets
        # cfg.losses.video_emotion_loss.video_network_folder = "/is/cluster/work/rdanecek/video_emotion_recognition/trainings/"
        # ## best transformer, 4 layers, 512 hidden size
        # cfg.losses.video_emotion_loss.video_network_name = "2023_01_09_12-42-15_7763968562013076567_VideoEmotionClassifier_MEADP_TSC_PE_Lnce"
        # ## gru, 4 layers, 512 hidden size
        # # cfg.losses.video_emotion_loss.video_network_name = "2023_01_09_12-44-24_-8682625798410410834_VideoEmotionClassifier_MEADP_GRUbi_nl-4_Lnce"
        # cfg.losses.video_emotion_loss.network_path = str(Path(cfg.losses.video_emotion_loss.video_network_folder) / cfg.losses.video_emotion_loss.video_network_name)
        # # cfg.losses.video_emotion_loss.from_source = False # set this on emotion_loss instead
        # cfg.losses.video_emotion_loss.active = True
        # # cfg.losses.video_emotion_loss.active = False
        # cfg.losses.video_emotion_loss.feature_extractor = "no"
        # cfg.losses.video_emotion_loss.metric = "mse"
        # # cfg.losses.video_emotion_loss.weight = 1000.0
        # cfg.losses.video_emotion_loss.weight = 100.0
        # # cfg.losses.video_emotion_loss.weight = 10.0
        # # cfg.losses.video_emotion_loss.weight = 5.0
        # # cfg.losses.video_emotion_loss.weight = 2.5
        # # cfg.losses.video_emotion_loss.weight = 1.0
        
        # video_emotion_loss_cfg.feat_extractor_cfg = "no"

        # cfg.losses.lip_reading_loss = munchify(OmegaConf.to_container(helper_config.learning.losses.lip_reading_loss))
        # cfg.losses.lip_reading_loss.from_source = True
        # # cfg.losses.lip_reading_loss.from_source = False
        # cfg.losses.lip_reading_loss.active = True
        # # cfg.losses.lip_reading_loss.active = False
        # cfg.losses.lip_reading_loss.weight = 100.00
        # # cfg.losses.lip_reading_loss.weight = 0
        # cfg.losses.expression_reg = Munch()
        # # cfg.losses.expression_reg.weight = 1.0
        # cfg.losses.expression_reg.weight = 1e-3
        # cfg.losses.expression_reg.active = True

        cfg.settings = Munch()
        # cfg.settings.optimize_exp = True
        # cfg.settings.optimize_jaw_pose = True

        # if cfg.settings.optimize_jaw_pose:
        #     cfg.losses.jaw_pose_reg = Munch()
        #     cfg.losses.jaw_pose_reg.weight = 1.0
        #     # cfg.losses.jaw_pose_reg.active = True
        #     cfg.losses.jaw_pose_reg.active = False
        #     cfg.losses.jaw_pose_reg.from_source = True
        #     # cfg.losses.jaw_pose_reg.from_source = False
        #     cfg.losses.jaw_pose_reg.input_space = 'aa'
        #     cfg.losses.jaw_pose_reg.output_space = '6d'

        cfg.settings.flame_cfg = munchify({
            "type": "flame",
            "flame_model_path": "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl",
            "n_shape": 100 ,
            # n_exp: 100,
            "n_exp": 50,
            "flame_lmk_embedding_path": "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy" ,
            "tex_type": "BFM",
            "tex_path": "/ps/scratch/rdanecek/data/FLAME/texture/FLAME_albedo_from_BFM.npz",
            "n_tex": 50,
        })

        # cfg.settings.renderer = munchify(OmegaConf.to_container(helper_config.model.renderer))

        cfg.optimizer = Munch()
        # cfg.optimizer.type = "adam"
        # cfg.optimizer.type = "sgd"
        cfg.optimizer.type = "lbfgs"
        # cfg.optimizer.lr = 1e-4
        # cfg.optimizer.lr = 1e-3
        # cfg.optimizer.lr = 1e-6
        # cfg.optimizer.lr = 1e-2
        # cfg.optimizer.lr = 1e-1
        cfg.optimizer.lr = 1.
        if cfg.optimizer.type == "lbfgs":
            cfg.optimizer.n_iter = 1000
        else:
            cfg.optimizer.n_iter = 10000
        # cfg.optimizer.n_iter = 100
        cfg.optimizer.patience = 50
        
        cfg.init = Munch()
        # cfg.init.source_sample_idx = 61 #
        # cfg.init.target_sample_idx = 58 #
        cfg.init.source_sample_idx = 0 # 
        cfg.init.target_sample_idx = 1 #
        cfg.init.geometry_type = 'emoca'
        # cfg.init.geometry_type = 'spectre'
        # cfg.init.init = 'random'
        # cfg.init.init = 'source'
        cfg.init.latent_seq_init = 'zeros'
        # cfg.init.shape_from_source = True
        cfg.init.shape_from_source = False

        cfg.inout = Munch()
        cfg.inout.result_root = "/is/cluster/work/rdanecek/talkinghead/motion_prior_fitting"

    optimize(cfg)


if __name__ == "__main__":    
    main()
