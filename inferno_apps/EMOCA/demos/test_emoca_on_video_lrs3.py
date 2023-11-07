# import tensorflow
from inferno.models.external.Deep3DFace import Deep3DFaceModule
from inferno_apps.EMOCA.utils.load import load_model
from inferno.datasets.FaceVideoDataModule import TestFaceVideoDM
import inferno
from pathlib import Path
from tqdm import auto
import argparse
from inferno_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
import numpy as np
import os
from omegaconf import DictConfig


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_video', type=str, default="/ps/project/EmotionalFacialAnimation/data/aff-wild2/Aff-Wild2_ready/AU_Set/videos/Test_Set/82-25-854x480.mp4")
    # parser.add_argument('--input_video', type=str, default="/ps/project/EmotionalFacialAnimation/data/aff-wild2/Aff-Wild2_ready/AU_Set/videos/Test_Set/30-30-1920x1080.mp4", 
        # help="Filename of the video for reconstruction.")
    parser.add_argument('--input_video', type=str, default="/ps/scratch/rdanecek/EMOCA/Videos/ThisIsUs_s01_trailer.mp4", 
        help="Filename of the video for reconstruction.")
    # parser.add_argument('--output_folder', type=str, default="/ps/scratch/rdanecek/Deep3DFace/lrs3", help="Output folder to save the results to.")
    parser.add_argument('--output_folder', type=str, default="/ps/scratch/rdanecek/EMOCA/lrs3", help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA', help='Name of the model to use.')
    # parser.add_argument('--model_name', type=str, default='Deep3DFace', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str, default=Path(inferno.__file__).parents[1] / "assets/EMOCA/models")
    parser.add_argument('--save_images', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=False, help="If true, output meshes will be saved")
    # add a string argument with several options for image type
    parser.add_argument('--image_type', type=str, default='geometry_detail', 
        choices=["geometry_detail", "geometry_coarse", "output_images_detail", "output_images_coarse"], 
        help="Which image to use for the reconstruction video.")
    parser.add_argument('--processed_subfolder', type=str, default=None, 
        help="If you want to resume previously interrupted computation over a video, make sure you specify" \
            "the subfolder where the got unpacked. It will be in format 'processed_%Y_%b_%d_%H-%M-%S'")
    args = parser.parse_args()

    input_folder = Path("/ps/project/EmotionalFacialAnimation/data/lrs3/extracted/test")
    detect = True

    if not detect: 
        args.output_folder += "_nodetect"

    detector= '3fabrec'
    if detector is not None:
        args.output_folder += f"_{detector}" 

    videos = sorted(list(input_folder.glob("**/*.mp4")))
    print(f"Found {len(videos)} videos in {str(input_folder)}")

    video_idxs = np.arange(len(videos), dtype=int)
    np.random.seed(0)
    np.random.shuffle(videos)
    # start_video = 16
    start_video = 0
    end_video = 200
    # end_video = 16

    path_to_models = args.path_to_models
    output_folder = args.output_folder
    model_name = args.model_name
    image_type = args.image_type

    mode = 'detail'
    # mode = 'coarse'
    ## 2) Load the model
    if model_name == "Deep3DFace":
        model = instantiate_deep3d_face()
    else:
        model, conf = load_model(path_to_models, model_name, mode)
        model.cuda()
        model.eval()

    # processed_subfolder = args.processed_subfolder
    # processed_subfolder = None
    processed_subfolder = "processed"

    for vi in range(start_video, end_video):
        input_video = videos[video_idxs[vi]]
        
        output_folder =  Path(args.output_folder) / input_video.parent.stem / input_video.stem
        ## 1) Process the video - extract the frames from video and detected faces
        # processed_subfolder="processed_2022_Jan_15_02-43-06"
        # processed_subfolder=None
        dm = TestFaceVideoDM(input_video, output_folder, processed_subfolder=processed_subfolder, detect=detect,
            batch_size=4, num_workers=4, face_detector=detector)
        dm.prepare_data()
        dm.setup()
        processed_subfolder = Path(dm.output_dir).name

        # outfolder = str(Path(output_folder) / processed_subfolder / Path(input_video).stem / "results" / model_name)
        outfolder =  Path(dm.output_dir) / input_video.stem / "results" / model_name


        ## 3) Get the data loadeer with the detected faces
        dl = dm.test_dataloader()

        ## 4) Run the model on the data
        for j, batch in enumerate (auto.tqdm( dl)):

            current_bs = batch["image"].shape[0]
            img = batch
            vals, visdict = test(model, img)
            for i in range(current_bs):
                # name = f"{(j*batch_size + i):05d}"
                name =  batch["image_name"][i]

                sample_output_folder = Path(outfolder) /name
                sample_output_folder.mkdir(parents=True, exist_ok=True)

                if args.save_mesh:
                    save_obj(model, str(sample_output_folder / "mesh_coarse.obj"), vals, i)
                if args.save_images:
                    save_images(outfolder, name, visdict, i)
                if args.save_codes:
                    save_codes(Path(outfolder), name, vals, i)

        ## 5) Create the reconstruction video (reconstructions overlayed on the original video)
        dm.create_reconstruction_video(0,  rec_method=model_name, image_type=image_type, overwrite=True)
        #create symlink to the reconstruction video 
        symlink_path = Path(output_folder) / processed_subfolder / Path(input_video).stem / "results" / model_name / "video_geometry_detail_with_sound.mp4"
        # if symlink_path.exists():
            # os.remove(symlink_path)
        target = str(output_folder) + ".mp4"
        target2 = Path(args.output_folder) / (input_video.parent.stem + "_" + input_video.stem + ".mp4")
        try:
            os.symlink(str(symlink_path), target)
        except FileExistsError:
            pass

        try:
            os.symlink(str(symlink_path), target2)
        except FileExistsError:
            pass
        print("Done")



def instantiate_deep3d_face(): 
    model_cfg = {
                # "value": {
                    "mode": "detail",
                    # "n_cam": 3,
                    # "n_exp": 50,
                    # "n_tex": 50,
                    # "n_pose": 6,
                    # "n_light": 27,
                    # "n_shape": 100,
                    # "uv_size": 256,
                    # "n_detail": 128,
                    # "tex_path": "/ps/scratch/rdanecek/data/FLAME/texture/FLAME_albedo_from_BFM.npz",
                    # "tex_type": "BFM",
                    "n_dlatent": 512,
                    "deca_class": "Deep3DFaceModule",
                    "deep3dface": {
                        "name": "face_recon_feat0.2_augment",
                        "epoch": 20,
                        "focal": 1015,
                        "model": "facerecon",
                        "phase": "test",
                        "z_far": 15,
                        "center": 112,
                        "suffix": "null",
                        "z_near": 5,
                        "gpu_ids": 0,
                        "isTrain": False,
                        "use_ddp": False,
                        "verbose": False,
                        "camera_d": 10,
                        "ddp_port": 12355,
                        "add_image": True,
                        "bfm_model": "BFM_model_front.mat",
                        "init_path": "checkpoints/init_model/resnet50-0676ba61.pth",
                        "net_recon": "resnet50",
                        "bfm_folder": "BFM",
                        "img_folder": "./datasets/examples",
                        "world_size": 1,
                        "use_last_fc": False,
                        "dataset_mode": "None",
                        "vis_batch_nums": 1,
                        "checkpoints_dir": "./checkpoints",
                        "eval_batch_nums": "inf",
                        "display_per_batch": True
                    },
                    "image_size": 224,
                    "max_epochs": 4,
                    # "n_identity": 512,
                    # "topology_path": "/ps/scratch/rdanecek/data/FLAME/geometry/head_template.obj",
                    # "face_mask_path": "/ps/scratch/rdanecek/data/FLAME/mask/uv_face_mask.png",
                    # "neural_renderer": false,
                    # "flame_model_path": "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl",
                    # "val_vis_frequency": 200,
                    # "face_eye_mask_path": "/ps/scratch/rdanecek/data/FLAME/mask/uv_face_eye_mask.png",
                    # "test_vis_frequency": 1,
                    # "val_check_interval": 0.2,
                    # "train_vis_frequency": 1000,
                    # "pretrained_modelpath": "/home/rdanecek/Workspace/Repos/DECA/data/deca_model.tar",
                    # "background_from_input": true,
                    # "fixed_displacement_path": "/ps/scratch/rdanecek/data/FLAME/geometry/fixed_uv_displacements/fixed_displacement_256.npy",
                    # "flame_lmk_embedding_path": "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy",
                    # "pretrained_vgg_face_path": "/ps/scratch/rdanecek/pretrained_vggfaceresnet/resnet50_ft_weight.pkl"
                }
            # }

    learning_cfg =  {
                "path": "/ps/scratch/face2d3d/",
                "n_train": 10000000,
                "scale_max": 1.6,
                "scale_min": 1.2,
                "data_class": "DecaDataModule",
                "num_workers": 4,
                "split_ratio": 0.9,
                "split_style": "random",
                "trans_scale": 0.1,
                "testing_datasets": [
                    "now-test",
                    "now-val",
                    "celeb-val"
                ],
                "training_datasets": [
                    "vggface2hq",
                    "vox2"
                ],
                "validation_datasets": [
                    "now-val",
                    "celeb-val"
                ]
            }

    inout_cfg = {
                "name": "ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early",
                "time": "2021_11_13_03-43-40",
                "random_id": "3038711584732653067",
                "output_dir": "/is/cluster/work/rdanecek/emoca/finetune_deca",
                "full_run_dir": "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail",
                "checkpoint_dir": "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/checkpoints"
            }


    face_model = Deep3DFaceModule(DictConfig(model_cfg), DictConfig(learning_cfg),
                                    DictConfig(inout_cfg), "")
    # face_model.to(device)
    return face_model



if __name__ == '__main__':
    main()
