from pathlib import Path
from inferno.datasets.MEADDataModule import MEADDataModule 
from inferno.datasets.IO import load_reconstruction_list
from inferno.models.DecaFLAME import FLAME_mediapipe
from inferno.utils.PyRenderMeshSequenceRenderer import PyRenderMeshSequenceRenderer
import numpy as np
import os, sys
from munch import munchify, Munch
import torch
from skimage.io import imsave


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

    if len(sys.argv) > 1:
        rec_method = sys.argv[1]
    else:
        # rec_method = "EMICA_v0_mp"
        rec_method = "emoca"

    n_shape = 100
    if  "mica" in rec_method.lower():
        n_shape = 300 


    # load FLAME model
    flame_model_path = Path("/ps/project/EmotionalFacialAnimation/data/flame_model/generic_model.pkl")
    flame_cfg = munchify({
            "type": "flame",
            "flame_model_path": "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl",
            "n_shape": n_shape ,
            # n_exp: 100,
            "n_exp": 50,
            "flame_lmk_embedding_path": "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy" ,
            "flame_mediapipe_lmk_embedding_path": "/ps/scratch/rdanecek/data/FLAME/geometry/mediapipe_landmark_embedding.npz",
            "tex_type": "BFM",
            "tex_path": "/ps/scratch/rdanecek/data/FLAME/texture/FLAME_albedo_from_BFM.npz",
            "n_tex": 50,
        })
    
    flame = FLAME_mediapipe(flame_cfg)
    template_file = flame_template_path = Path("/ps/scratch/rdanecek/data/FLAME/geometry/FLAME_sample.ply")
    renderer = PyRenderMeshSequenceRenderer(
            template_file,
            # height=600., 
            # width=600.,
            # bg_color=None, 
            # t_center=None, 
            # rot=np.zeros(3), 
            # tex_img=None, 
            # z_offset=0,
    )
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    flame = flame.to(device)

    # # Create the dataset
    # dm = MEADDataModule(
    #         root_dir, output_dir, processed_subfolder,
    #         scale=1.35, # zooms out the face a little bit s.t. forehead is very likely to be visible and lower part of the chin and a little bit of the neck as well
    #         bb_center_shift_x=0., # in relative numbers
    #         bb_center_shift_y=-0.1, # in relative numbers (i.e. -0.1 for 10% shift upwards, ...)
    #         # processed_video_size=256,
    #         processed_video_size=384,
    # )

    # print("Create the dataloader")
    # dm.prepare_data() 
    # # sys.exit(0)
    # # TODO: take care of these #
    # # [WARNING] Video file has no audio streams! 'M041/video/front/sad/level_2/020.mp4'
    # # [WARNING] Video file has no audio streams! 'M041/video/front/sad/level_2/021.mp4'
    # # [WARNING] Video file has no audio streams! 'M041/video/front/sad/level_2/022.mp4'
    # # [WARNING] Video file has no audio streams! 'M041/video/front/sad/level_2/023.mp4'

    # videos_per_shard = 200 
    # shard_idx = 0
    # if len(sys.argv) > 1:
    #     videos_per_shard = int(sys.argv[1])

    # if len(sys.argv) > 2:
    #     shard_idx = int(sys.argv[2])

    # print(videos_per_shard, shard_idx)
    # print(dm._get_num_shards(videos_per_shard))
    # sys.exit(0)

    # if len(sys.argv) > 3:
    #     extract_audio = bool(int(sys.argv[3]))
    # else: 
    #     extract_audio = False
    # if len(sys.argv) > 4:
    #     restore_videos = bool(int(sys.argv[4]))
    # else: 
    #     restore_videos = False
    # if len(sys.argv) > 5:
    #     detect_landmarks = bool(int(sys.argv[5]))
    # else: 
    #     detect_landmarks = False
    # if len(sys.argv) > 6:
    #     segment_videos = bool(int(sys.argv[6]))
    # else: 
    #     segment_videos = False
    # if len(sys.argv) > 7:
    #     detect_aligned_landmarks = bool(int(sys.argv[7]))
    # else: 
    #     detect_aligned_landmarks = False
    # if len(sys.argv) > 8:
    #     reconstruct_faces = bool(int(sys.argv[8])) 
    # else: 
    #     reconstruct_faces = True
    # if len(sys.argv) > 9:
    #     recognize_emotions = bool(int(sys.argv[9])) 
    # else: 
    #     recognize_emotions = False


    # dm._process_shard(
    #     videos_per_shard, 
    #     shard_idx, 
    #     extract_audio=extract_audio,
    #     restore_videos=restore_videos, 
    #     detect_landmarks=detect_landmarks, 
    #     segment_videos=segment_videos, 
    #     detect_aligned_landmarks=detect_aligned_landmarks,
    #     reconstruct_faces=reconstruct_faces,
    #     recognize_emotions=recognize_emotions,
    # )
    

    reconstructions = output_dir / processed_subfolder / "reconstructions" / rec_method
    audios = output_dir / processed_subfolder / "audio" 

    identities = sorted([x for x in os.listdir(reconstructions) if os.path.isdir(reconstructions / x)])
    if len(sys.argv) > 2:
        identity_index = int(sys.argv[2])
    else:        
        identity_index = 0
    
    expressions = sorted([x for x in os.listdir(reconstructions / identities[identity_index] / "front") if os.path.isdir(reconstructions / identities[identity_index] / "front" / x)])
    
    image_format = "%06d"

    for expression in expressions:
        # levels = sorted([x for x in os.listdir(reconstructions / identities[0] / "front"/ expressions[0]) if os.path.isdir(reconstructions / identities[0] / "front" / expressions[0] / x)])
        levels = sorted([x for x in os.listdir(reconstructions / identities[identity_index] / "front"/ expression) if os.path.isdir(reconstructions / identities[identity_index] / "front" / expression / x)])
        
        for level in levels:
            # sentences = sorted([x for x in os.listdir(reconstructions / identities[0] / "front" / expressions[0] / levels[0]) if os.path.isdir(reconstructions / identities[0] / "front" / expressions[0] / levels[0] / x)])
            sentences = sorted([x for x in os.listdir(reconstructions / identities[identity_index] / "front" / expression/ level) if os.path.isdir(reconstructions / identities[identity_index] / "front" / expression / level / x)])

            rec_file_name = "shape_pose_cam.pkl"
            # reconstruction_file_name = reconstructions / identities[0] / "front" / expressions[0] / levels[0] / sentences[0] / rec_file_name
            sentence_indices = [0,1,-1,-2]
            for sentence_index in sentence_indices:
                reconstruction_file_name = reconstructions / identities[identity_index] / "front" / expression / level / sentences[sentence_index] / rec_file_name
                video_path = reconstruction_file_name.parent / "unposed.mp4" 
                if video_path.exists():
                    print(f"Video {video_path} already exists. Skipping...")
                    continue

                # audio_file_name = reconstructions / identities[identity_index] / "front" / expressions[0] / levels[0] / sentences[0] / rec_file_name
                audio_file_name = (audios / identities[identity_index] / "front" / expression / level / sentences[sentence_index] ).with_suffix(".wav")
                rec = load_reconstruction_list(reconstruction_file_name)

                shape_params = torch.tensor(rec["shape"]).to(device)[0]
                expression_params = torch.tensor(rec["exp"]).to(device)[0]
                jaw_params = torch.tensor(rec["jaw"]).to(device)[0]
                global_pose_params = torch.tensor(rec["global_pose"]).to(device)[0]
                global_pose_params = torch.zeros_like(global_pose_params).to(device)

                pose_params = torch.cat([global_pose_params, jaw_params], dim=-1)

                predicted_vertices, _, _, _ = flame(shape_params, expression_params, pose_params)
                # expression_params = expression_params * 0
                # pose_params = pose_params * 0
                # predicted_vertices_shape_only, _, _, _ = flame(shape_params, expression_params*0, pose_params*0)

                # image_list = []
                image_paths = []
                for b in range(0, shape_params.shape[0]):
                    
                    pred_vertices = predicted_vertices[b].detach().cpu().view(-1,3).numpy()
                    pred_image = renderer.render(pred_vertices)


                    out_image_path = (reconstruction_file_name.parent / (image_format % b)).with_suffix(".png")
                    image_paths.append(out_image_path)
                    imsave(out_image_path, pred_image)


                framerate = 25

                ffmpeg_cmd = f"ffmpeg -y -framerate {framerate} -start_number {image_paths[0].stem} -i {str(video_path.parent)}/" + image_format + ".png"\
                                f" -i {str(audio_file_name)} -c:v libx264 -c:a aac -strict experimental -b:a 192k -pix_fmt yuv420p {str(video_path)}"

                print(ffmpeg_cmd)
                os.system(ffmpeg_cmd)
                # delete images
                for image_path in image_paths:
                    os.remove(image_path)
                print("Video saved to: ", video_path)

            # image_paths = []
            # for b in range(0, shape_params.shape[0]):
                
            #     pred_vertices_shape_only = predicted_vertices_shape_only[b].detach().cpu().view(-1,3).numpy()
            #     pred_image_shape_only = renderer.render(pred_vertices_shape_only)
            #     out_image_path = (reconstruction_file_name.parent / (image_format % b)).with_suffix(".png")
            #     image_paths.append(out_image_path)
            #     imsave(out_image_path, pred_image_shape_only)

            # video_path = reconstruction_file_name.parent / "shape_only.mp4" 
            # ffmpeg_cmd = f"ffmpeg -y -framerate {framerate} -start_number {image_paths[0].stem} -i {str(video_path.parent)}/" + image_format + ".png"\
            #                 f" -i {str(audio_file_name)} -c:v libx264 -c:a aac -strict experimental -b:a 192k -pix_fmt yuv420p {str(video_path)}"


            # #
            # print(ffmpeg_cmd)
            # os.system(ffmpeg_cmd)

            
            

if __name__ == "__main__": 
    main()
