from inferno_apps.EMOCA.utils.load import load_model
from inferno.datasets.ImageTestDataset import TestData
import inferno
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from inferno_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test, torch_img_to_np
from inferno.utils.lightning_logging import _fix_image


def save_images(outfolder, name, vis_dict, i = 0, with_detection=False):
    prefix = None
    final_out_folder = Path(outfolder) #/ name
    final_out_folder.mkdir(parents=True, exist_ok=True)

    imname = f"0000_{int(name):04d}_00.png"

    (final_out_folder / f"inputs").mkdir(parents=True, exist_ok=True)
    (final_out_folder / f"geometry_coarse").mkdir(parents=True, exist_ok=True)
    (final_out_folder / f"geometry_detail").mkdir(parents=True, exist_ok=True)
    (final_out_folder / f"output_images_coarse").mkdir(parents=True, exist_ok=True)
    (final_out_folder / f"output_images_detail").mkdir(parents=True, exist_ok=True)

    if with_detection:
        imsave(final_out_folder / f"inputs" / imname ,  _fix_image(torch_img_to_np(vis_dict['inputs'][i])))
    imsave(final_out_folder / f"geometry_coarse" / imname,  _fix_image(torch_img_to_np(vis_dict['geometry_coarse'][i])))
    imsave(final_out_folder / f"geometry_detail" / imname, _fix_image(torch_img_to_np(vis_dict['geometry_detail'][i])))
    imsave(final_out_folder / f"output_images_coarse" / imname, _fix_image(torch_img_to_np(vis_dict['output_images_coarse'][i])))
    imsave(final_out_folder / f"output_images_detail" / imname, _fix_image(torch_img_to_np(vis_dict['output_images_detail'][i])))



def main():
    parser = argparse.ArgumentParser()
    # add the input folder arg 
    # parser.add_argument('--input_folder', type=str, default="/ps/data/SignLanguage/SignLanguage_210805_03586_GH/IOI/2021-08-05_ASL_PNG_MH/SignLanguage_210805_03586_GH_LiebBitte_2/Cam_0_35mm_90CW")
    parser.add_argument('--input_folder', type=str, default="/ps/scratch/rdanecek/EMOCA/ContemptImages/original")
    # parser.add_argument('--output_folder', type=str, default="/ps/scratch/rdanecek/EMOCA/OutContempt", help="Output folder to save the results to.")
    # parser.add_argument('--model_name', type=str, default='EMOCA', help='Name of the model to use.')
    # parser.add_argument('--model_name', type=str, default='EMOCA', help='Name of the model to use.')
    # parser.add_argument('--model_name', type=str, default='EMOCA', help='Name of the model to use.')
    # parser.add_argument('--model_name', type=str, default='EMOCA', help='Name of the model to use.')
    # parser.add_argument('--model_name', type=str, default='EMOCA', help='Name of the model to use.')
    # parser.add_argument('--model_name', type=str, default='EMOCA', help='Name of the model to use.')
    # parser.add_argument('--path_to_models', type=str, default=Path(inferno.__file__).parents[1] / "assets/EMOCA/models")
    parser.add_argument('--path_to_models', type=str, default="/is/cluster/work/rdanecek/emoca/finetune_deca/")
    parser.add_argument('--save_images', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=False, help="If true, output meshes will be saved")
    
    args = parser.parse_args()


    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    path_to_models = args.path_to_models
    input_folder = args.input_folder
    
    methods = {}
    # methods[
        # "MGCNet"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_MGCNet/detail/inputs"
    # methods[
        # "3DDFA_v2"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-40-45_-5868754668879675020_Face3DDFAModule/detail/affect_net_mturk_detail_test/geometry_coarse"
    methods[
        "Deep3DFace"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-09-34_6754141025581837735_Deep3DFaceModule/detail/affect_net_mturk_detail_test/geometry_coarse"
    # methods[
        # "DecaCoarse"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2/detail/affect_net_mturk_detail_test/geometry_coarse"
    # methods[
        # "DecaDetail"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2/detail/affect_net_mturk_detail_test/geometry_detail"

    # methods[
        # "EmocaCoarse"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_coarse"
    # methods[
        # "EmocaDetail"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_detail"

    for method_name, method_path in methods.items():

        # output_folder = args.output_folder
        # model_name = args.model_name
        method_path = Path(method_path)
        model_name = method_path.parts[7]
        subfolder = method_path.parts[-2]

        output_folder = method_path.parents[1] / "contempt_images" 


        mode = 'detail'
        # mode = 'coarse'

        # 1) Load the model
        emoca, conf = load_model(path_to_models, model_name, mode)
        emoca.cuda()
        emoca.eval()

        # 2) Create a dataset
        dataset = TestData(input_folder, face_detector="fan", max_detection=20)

        ## 4) Run the model on the data
        for i in auto.tqdm( range(len(dataset))):
            batch = dataset[i]
            vals, visdict = test(emoca, batch)
            # name = f"{i:02d}"
            current_bs = batch["image"].shape[0]

            for j in range(current_bs):
                name =  batch["image_name"][j]

                sample_output_folder = Path(output_folder) # / name
                sample_output_folder.mkdir(parents=True, exist_ok=True)

                # if args.save_mesh:
                    # save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, j)
                    # save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, i)
                if args.save_images:
                    save_images(output_folder, name, visdict, with_detection=True, i=j)
                    # save_images(output_folder, name, visdict, with_detection=True, i=i)
                # if args.save_codes:
                    # save_codes(Path(output_folder), name, vals, i=j)
                    # save_codes(Path(output_folder), name, vals, i=i)

        print("Done")


if __name__ == '__main__':
    main()
