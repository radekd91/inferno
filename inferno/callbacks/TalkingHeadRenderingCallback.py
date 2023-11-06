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
import pytorch_lightning as pl 
from inferno.utils.PyRenderMeshSequenceRenderer import PyRenderMeshSequenceRenderer
from pathlib import Path
from skimage.io import imsave
import os, sys
from wandb import Video
import pickle as pkl
import numpy as np
# import librosa
import soundfile as sf


class TalkingHeadTestRenderingCallback(pl.Callback):

    def __init__(self, template_mesh_path, path_chunks_to_cat=None, predicted_vertex_key=None, save_meshes=False):
        # self.talking_head = talking_head
        # self.sample_interval = sample_interval
        self.renderer = PyRenderMeshSequenceRenderer(template_mesh_path)
        self.image_format = "%06d"
        self.video_names_to_process = {}
        self.audio_samplerates_to_process = {}
        self.video_framerates_to_process = {}
        self.video_conditions = {}
        self.dl_names = {}
        self.framerate = 25
        self.overwrite = False
        self.predicted_vertex_key = predicted_vertex_key or "predicted_vertices"
        self.path_chunks_to_cat = path_chunks_to_cat or 0
        self.save_meshes = save_meshes
        self.save_frames_to_disk = False
        self._image_cache = []
        self._audio_cache = []

    def _path_chunk(self, video_name):
        video_name = Path(video_name)
        if self.path_chunks_to_cat == 0:
            return Path(video_name.stem)
        return Path(*video_name.parts[-self.path_chunks_to_cat-1:-1], video_name.stem)


    def on_test_batch_end(self, trainer, pl_module, 
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        
        # for each mesh in batch, render the talking head 
        # and save the image to the output directory
        # print("Rendering talking head")
        logger = pl_module.logger
        predicted_vertices = batch[self.predicted_vertex_key]
        if "gt_vertices" in batch.keys():
            gt_vertices = batch["gt_vertices"]  
        else: 
            rec_type = list(batch["reconstruction"].keys())[0]
            if len(batch["reconstruction"]) > 1:
                print("[WARNING]: More than one reconstruction type in batch. Using first one.")
            gt_vertices = batch["reconstruction"][rec_type]["gt_vertices"]
        B, T = predicted_vertices.shape[:2]

        for b in range(B):
            video_name = batch["filename"][b]
            condition_name = None
            if "condition_name" in batch.keys():
                condition_name = batch["condition_name"][b][0]
            
            if hasattr(trainer.datamodule, "test_set_names"):
                dl_name = trainer.datamodule.test_set_names[dataloader_idx]
            else: 
                dl_name = f"{dataloader_idx:02d}"

            if condition_name is None:
                path = Path(pl_module.cfg.inout.full_run_dir) / "videos" / dl_name / self._path_chunk(video_name)
            else: 
                path = Path(pl_module.cfg.inout.full_run_dir) / "videos" / dl_name / self._path_chunk(video_name) / condition_name
            self.video_conditions[path] = condition_name
            self.dl_names[path] = dl_name

            path.mkdir(parents=True, exist_ok=True)


            # self.video_names_to_process[path] = audio_path
            if "framerate" in batch:
                self.video_framerates_to_process[path] = batch["framerate"][b]
            else:
                self.video_framerates_to_process[path] = self.framerate

            if 'raw_audio' in batch.keys():
                self.audio_samplerates_to_process[path] = batch["samplerate"][b]
            
            if (path / "output.mp4").is_file() and not self.overwrite:
                print(f"Skipping {path}. Video already exists.")
                continue

            # audio_path = Path(video_name)
            for t in range(T):
                frame_index = batch["frame_indices"][b, t].item()
                image_path = path / (self.image_format % frame_index + ".png")
                if image_path.is_file() and not self.overwrite:
                    continue
                if t == 0:
                    print("Writing:", image_path)

                valid = True
                if "landmarks_validity" in batch.keys() and "mediapipe" in batch["landmarks_validity"]:
                    valid = bool(batch["landmarks_validity"]["mediapipe"][b, t].item())

                pred_vertices = predicted_vertices[b, t].detach().cpu().view(-1,3).numpy()
                pred_image = self.renderer.render(pred_vertices)

                ref_vertices = gt_vertices[b, t].detach().cpu().view(-1,3).numpy()
                gt_image = self.renderer.render(ref_vertices, valid=valid)

                # concatenate the images
                image = np.concatenate([gt_image, pred_image], axis=1)

                # if trainer.datamodule.sequence_length_test != "all" or not self.save_images_to_disk:
                # if trainer.datamodule.sequence_length_test != "all" or not self.save_images_to_disk:
                if self.save_frames_to_disk:
                    imsave(image_path, image)
                else: 
                    if not hasattr(self, "_image_cache"):
                        self._image_cache = []
                    self._image_cache += [image]

                if self.save_meshes: 
                    import trimesh
                    mesh = trimesh.base.Trimesh(pred_vertices, self.renderer.template.faces)
                    mesh_path = path / (self.image_format % frame_index + ".obj")
                    mesh.export(mesh_path)

                raw_audio_chunk = batch["raw_audio"][b,t].detach().cpu().numpy()
                if self.save_frames_to_disk:
                    if 'raw_audio' in batch.keys():
                        audio_chunk_path = path / (self.image_format % frame_index + ".pkl")
                        with open(audio_chunk_path, "wb") as f:
                            pkl.dump(raw_audio_chunk, f)
                else: 
                    if not hasattr(self, "_audio_cache"):
                        self._audio_cache = []
                    self._audio_cache += [raw_audio_chunk]

            if not self.save_frames_to_disk and len(self._image_cache) > 0:
                # write the video
                # self._compose_chunked_audio(path, samplerate=self.audio_samplerates_to_process[path])
                self._compose_chunked_audio(path, samplerate=self.audio_samplerates_to_process[path], audio_chunks=self._audio_cache)
                self._save_video_from_tensor(self._image_cache, self.video_framerates_to_process[path], path)

                self._image_cache = []
                self._audio_cache = []

            # self._create_video(path, logger, trainer.global_step)

    # def on_test_epoch_begin(self, trainer, pl_module):
    #     super().on_test_epoch_begin(trainer, pl_module)
    #     self._create_videos(trainer, pl_module))

    def _save_video_from_tensor(self, image_list, framerate, path): 
        import skvideo
        image_tensor = np.stack(image_list)
        audio_path = path / "audio.wav"
        video_path = path / "output_no_audio.mp4"
        # write the video including audio 

        skvideo.io.vwrite(video_path, image_tensor, 
                          inputdict={'-r': str(framerate)}, 
                          outputdict={'-r': str(framerate)})
        
        # combine the audio and video using ffmpeg
        if audio_path.is_file():
            ffmpeg_cmd = f"ffmpeg -y -i {str(video_path)} -i {str(audio_path)} -c:v copy -c:a aac -strict experimental {str(path / 'output.mp4')}"
            os.system(ffmpeg_cmd)

            # remove the video without audio
            video_path.unlink()
            # remove the audio file
        else: 
            video_path.rename(path / "output.mp4")

        # remove the audio file
        audio_path.unlink()
        

    def on_test_epoch_end(self, trainer, pl_module):
        super().on_test_epoch_end(trainer, pl_module)
        self._create_videos(trainer, pl_module)

    def _create_videos(self, trainer, pl_module):
        logger = pl_module.logger
        
        # for subfolder in self.video_names_to_process.keys():
        for subfolder in self.video_framerates_to_process.keys():
            self._create_video(subfolder, logger, trainer.global_step)

        self.video_names_to_process = {}
        self.audio_samplerates_to_process = {}
        self.video_conditions = {}
        self.dl_names = {}

    def _compose_chunked_audio(self, subfolder, samplerate, audio_chunks=None):
        # create audio 
        audio_chunks_files = None
        if audio_chunks is None:
            audio_chunks_files = sorted([f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".pkl"])
            audio_chunks = [pkl.load(open(f, "rb")) for f in audio_chunks_files]

        if len(audio_chunks) > 0:
            audio_chunks = np.concatenate(audio_chunks, axis=0)
            audio_path = subfolder / "audio.wav"
            sf.write(audio_path, audio_chunks, samplerate)
        else:
            audio_path = None

        if audio_path.is_file() and audio_chunks_files is not None:
            # delete the audio chungs 
            for audio_chunk in audio_chunks_files:
                os.remove(audio_chunk)
        return audio_path

    def _create_video(self, subfolder, logger, epoch):
        # find audio
        # audio = self.video_names_to_process[subfolder]
        framerate = self.video_framerates_to_process[subfolder]
        samplerate = self.audio_samplerates_to_process[subfolder] if subfolder in self.audio_samplerates_to_process else 16000
        video_path = subfolder / ("output.mp4")

        if not video_path.is_file():

            # find all renderings in the subfolder
            renderings = sorted([f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".png"])    

            # create audio 
            audio_path = self._compose_chunked_audio(subfolder, samplerate)
            audio_chunks_files = sorted([f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".pkl"])
            # if len(audio_chunks_files) > 0:
            #     audio_chunks = [pkl.load(open(f, "rb")) for f in audio_chunks_files]
            #     audio_chunks = np.concatenate(audio_chunks, axis=0)
            #     audio_path = subfolder / "audio.wav"
            #     sf.write(audio_path, audio_chunks, samplerate)
            # else:
            #     audio_path = None

            # create video with audio
            if audio_path is not None:
                ffmpeg_cmd = f"ffmpeg -y -framerate {framerate} -start_number {renderings[0].stem} -i {str(subfolder)}/" + self.image_format + ".png"\
                    f" -i {str(audio_path)} -c:v libx264 -c:a aac -strict experimental -b:a 192k -pix_fmt yuv420p {str(video_path)}"
            else:
                ffmpeg_cmd = f"ffmpeg -y -framerate {framerate} -start_number {renderings[0].stem} -i {str(subfolder)}/" + self.image_format + ".png"\
                    f" -c:v libx264 -pix_fmt yuv420p {str(video_path)}"
            # ffmpeg_cmd = f"ffmpeg -y -framerate {framerate}  -start_number {renderings[0].stem} -i {str(subfolder)}/*.png"\
            #     f" -i {str(audio_path)} -c:v libx264 -c:a aac -strict experimental -b:a 192k -pix_fmt yuv420p {str(video_path)}"
            # ffmpeg_cmd = f"ffmpeg -framerate 30 -i {str(subfolder)}/" + self.image_format + f".png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {str(video_path)}"

            # create video from renderings
            os.system(ffmpeg_cmd)

            # check if the video was created
            if not video_path.exists():
                print(f"Video {video_path} was not created. The renderings are in {subfolder}")
                return
            
            print(f"Video {video_path} was created. Deleting renderings.")
            for rendering in renderings:
                os.remove(rendering)
            

            if audio_path is not None:
                os.remove(audio_path)

        # log the video
        self._log_video(video_path, logger, epoch)
                    
    def _log_video(self, video_path, logger, epoch):
        if logger is not None: 
            if isinstance(logger, pl.loggers.WandbLogger):
                name = "test_video" 
                dl_name = self.dl_names[video_path.parent]
                if dl_name is not None:
                    name += f"/{dl_name}"
                condition = self.video_conditions[video_path.parent]
                if condition is not None:
                    name += "/" + condition 
                name += "/" + str(self._path_chunk(video_path.parent))

                logger.experiment.log({name: Video(str(video_path), 
                    fps=self.framerate, format="mp4", caption=str(video_path))}) #, step=epoch)