import pytorch_lightning as pl 
from gdl.utils.PyRenderMeshSequenceRenderer import PyRenderMeshSequenceRenderer
from pathlib import Path
from skimage.io import imsave
import os, sys
from wandb import Video
import pickle as pkl
import numpy as np
# import librosa
import soundfile as sf


class TalkingHeadTestRenderingCallback(pl.Callback):

    def __init__(self, template_mesh_path, path_chunks_to_cat=None):
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

        self.path_chunks_to_cat = path_chunks_to_cat or 0

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

        predicted_vertices = batch["predicted_vertices"]
        gt_vertices = batch["gt_vertices"]
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

            self.audio_samplerates_to_process[path] = batch["samplerate"][b]

            # continue

            # audio_path = Path(video_name)
            for t in range(T):
                frame_index = batch["frame_indices"][b, t].item()
                image_path = path / (self.image_format % frame_index + ".png")
                if image_path.is_file() and not self.overwrite:
                    continue

                valid = True
                if "landmarks_validity" in batch.keys() and "mediapipe" in batch["landmarks_validity"]:
                    valid = bool(batch["landmarks_validity"]["mediapipe"][b, t].item())

                pred_vertices = predicted_vertices[b, t].detach().cpu().view(-1,3).numpy()
                pred_image = self.renderer.render(pred_vertices)

                ref_vertices = gt_vertices[b, t].detach().cpu().view(-1,3).numpy()
                gt_image = self.renderer.render(ref_vertices, valid=valid)

                # concatenate the images
                image = np.concatenate([gt_image, pred_image], axis=1)

                imsave(image_path, image)

                raw_audio_chunk = batch["raw_audio"][b,t].detach().cpu().numpy()
                audio_chunk_path = path / (self.image_format % frame_index + ".pkl")
                with open(audio_chunk_path, "wb") as f:
                    pkl.dump(raw_audio_chunk, f)


    def on_test_epoch_end(self, trainer, pl_module):
        super().on_test_epoch_end(trainer, pl_module)
        logger = pl_module.logger
        
        # for subfolder in self.video_names_to_process.keys():
        for subfolder in self.video_framerates_to_process.keys():
            self._create_video(subfolder, logger, trainer.global_step)

        self.video_names_to_process = {}
        self.audio_samplerates_to_process = {}
        self.video_conditions = {}
        self.dl_names = {}


    def _create_video(self, subfolder, logger, epoch):
        # find audio
        # audio = self.video_names_to_process[subfolder]
        framerate = self.video_framerates_to_process[subfolder]
        samplerate = self.audio_samplerates_to_process[subfolder]

        # find all renderings in the subfolder
        renderings = sorted([f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".png"])    

        # create audio 
        audio_chunks_files = sorted([f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".pkl"])
        audio_chunks = [pkl.load(open(f, "rb")) for f in audio_chunks_files]
        audio_chunks = np.concatenate(audio_chunks, axis=0)
        audio_path = subfolder / "audio.wav"
        sf.write(audio_path, audio_chunks, samplerate)


        # create video with audio
        video_path = subfolder / ("output.mp4")
        ffmpeg_cmd = f"ffmpeg -y -framerate {framerate} -start_number {renderings[0].stem} -i {str(subfolder)}/" + self.image_format + ".png"\
            f" -i {str(audio_path)} -c:v libx264 -c:a aac -strict experimental -b:a 192k -pix_fmt yuv420p {str(video_path)}"
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
        
        # delete the audio chungs 
        for audio_chunk in audio_chunks_files:
            os.remove(audio_chunk)
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