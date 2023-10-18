from pathlib import Path
from gdl.datasets.MEADDataModule import MEADDataModule 
from gdl.datasets.IO import load_reconstruction_list
from gdl.models.DecaFLAME import FLAME_mediapipe
from gdl.utils.PyRenderMeshSequenceRenderer import PyRenderMeshSequenceRenderer
import numpy as np
import os, sys
from munch import munchify, Munch
import torch
from skimage.io import imsave
from gdl_apps.EMOCA.paper_scripts.model_comparison_video import concatenate_videos


def main(): 
    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/celebvhq/auto_processed")
    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/mead/MEAD")
    # root_dir = Path("/is/cluster/work/rdanecek/data/mead_25fps/resampled_videos")
    root_dir = Path("/is/cluster/fast/rdanecek/data/mead_25fps/resampled_videos")
    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/celebvhq/auto_processed_online")
    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/celebvhq/auto_processed_online_25fps")
    # output_dir = Path("/is/cluster/work/rdanecek/data/mead/")
    # output_dir = Path("/is/cluster/work/rdanecek/data/mead_25fps/")
    output_dir = Path("/is/cluster/fast/rdanecek/data/mead_25fps/")
    # output_dir = Path("/ps/scratch/rdanecek/data/celebvhq/")
    # output_dir = Path("/home/rdanecek/Workspace/Data/celebvhq/")

    # root_dir = Path("/ps/project/EmotionalFacialAnimation/data/lrs2/mvlrs_v1")
    # output_dir = Path("/ps/scratch/rdanecek/data/lrs2")

    processed_subfolder = "processed"

    add_original_video = True
    
    # overwrite = False
    overwrite = True
    if len(sys.argv) > 1:
        rec_methods = sys.argv[1].split(',')
    else:
        # rec_methods = ["EMICA_v0_mp", "EMICA_v0_mp_lr_cos_1"]
        # rec_methods = ['EMICA_v0_mp', 'EMICA_v0_mp_lr_cos_1.5',  \
        #     'EMICA_v0_mp_lr_mse_15', 'EMICA_v0_mp_lr_mse_20',]
        rec_methods = ['spectre', 'emoca', 'EMOCA_v2_lr_mse_15_with_bfmtex', 'EMICA_v0_mp', 'EMICA_v0_mp_lr_cos_1.5', 'EMICA_v0_mp_lr_cos_1.5',  \
            'EMICA_v0_mp_lr_mse_15', 'EMICA_v0_mp_lr_mse_20',]
    audios = output_dir / processed_subfolder / "audio" 

    out_video_path_folder = output_dir / processed_subfolder / "reconstructions_comparisons" 

    rec_method = rec_methods[0]
    reconstructions = output_dir / processed_subfolder / "reconstructions" / rec_method
    # audios = output_dir / processed_subfolder / "audio" 

    # reconstructions = output_dir / processed_subfolder / "reconstructions" / rec_method
    # audios = output_dir / processed_subfolder / "audio" 

    identities = sorted([x for x in os.listdir(reconstructions) if os.path.isdir(reconstructions / x)])
    if len(sys.argv) > 2:
        identity_index = int(sys.argv[2])
    else:        
        identity_index = 0
    
    expressions = sorted([x for x in os.listdir(reconstructions / identities[identity_index] / "front") if os.path.isdir(reconstructions / identities[identity_index] / "front" / x)])
    
    image_format = "%06d"

    for expression in expressions:
        # levels = sorted([x for x in os.listdir(reconstructions / identities[0] / "front"/ expressions[0]) if os.path.isdir(reconstructions / identities[0] / "front" / expressions[0] / x)])
        levels = sorted([x for x in os.listdir(reconstructions / identities[identity_index] / "front"/ expression) 
                         if os.path.isdir(reconstructions / identities[identity_index] / "front" / expression / x)])
        
        for level in levels:
            # sentences = sorted([x for x in os.listdir(reconstructions / identities[0] / "front" / expressions[0] / levels[0]) if os.path.isdir(reconstructions / identities[0] / "front" / expressions[0] / levels[0] / x)])
            sentences = sorted([x for x in os.listdir(reconstructions / identities[identity_index] / "front" / expression/ level) 
                                if os.path.isdir(reconstructions / identities[identity_index] / "front" / expression / level / x)])

            rec_file_name = "shape_pose_cam.pkl"
            # reconstruction_file_name = reconstructions / identities[0] / "front" / expressions[0] / levels[0] / sentences[0] / rec_file_name
            sentence_indices = [0,1,-1,-2]

            for sentence_index in sentence_indices:
                

                videos_to_concat = []

                broken = False
                for rec_method in rec_methods:
                    reconstruction_file_name = output_dir / processed_subfolder / "reconstructions" / rec_method / identities[identity_index] / "front" / expression / level / sentences[sentence_index] / rec_file_name
                    video_path = reconstruction_file_name.parent / "unposed.mp4" 
                    if not video_path.exists():
                        print(f"Video {video_path} does not exist. Skipping...")
                        broken = True
                        break
                    videos_to_concat.append(video_path)

                if broken:
                    continue

                out_video_path = out_video_path_folder / identities[identity_index] / expression / level / f"{sentences[sentence_index]}_{'-'.join(rec_methods)}" / "unposed.mp4"
                out_video_path.parent.mkdir(parents=True, exist_ok=True)

                if add_original_video:
                    original_video = (output_dir / processed_subfolder / "videos_aligned" / identities[identity_index] / "front" / expression / level / sentences[sentence_index] ).with_suffix(".mp4")

                    resized_video = (output_dir / processed_subfolder / "videos_aligned_resized" / identities[identity_index] / "front" / expression / level / sentences[sentence_index] ).with_suffix(".mp4")

                    if not resized_video.exists():
                        resized_video.parent.mkdir(parents=True, exist_ok=True)
                        width = 800 
                        height = 800
                        # use ffmpeg to resize the video to a particular resolution (keeping all other things the same)
                        cmd = f"ffmpeg -i {str(original_video)} -vf scale={width}:{height} {str(resized_video)}"
                        os.system(cmd)
                        print("Resized video saved to: ", resized_video)

                    videos_to_concat = [resized_video] + videos_to_concat
                
                audio_file_name = (audios / identities[identity_index] / "front" / expression / level / sentences[sentence_index] ).with_suffix(".wav")
                if not out_video_path.exists() or overwrite:
                    # concatenate_videos(videos_to_concat, out_video_path, horizontal=True, with_audio=True)
                    assert audio_file_name.exists(), f"Audio file {audio_file_name} does not exist"
                    concatenate_videos(videos_to_concat, out_video_path, horizontal=True, with_audio=False, audio_file=str(audio_file_name), overwrite=overwrite)


                print("Video saved to: ", out_video_path)

            
            

if __name__ == "__main__": 
    main()
