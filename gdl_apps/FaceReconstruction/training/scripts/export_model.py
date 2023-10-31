import os, sys 
from pathlib import Path
from omegaconf import OmegaConf, DictConfig 
import shutil
from gdl.utils.other import get_path_to_assets
from gdl.models.IO import get_checkpoint_with_kwargs
from gdl.models.FaceReconstruction.FaceRecBase import FaceReconstructionBase 
from gdl.models.IO import locate_checkpoint


def export_model(input_model_folder, output_model_folder, overwrite=False, path_to_emotion_feature=None, with_texture=True):
    if not overwrite and output_model_folder.exists(): 
        print(f"There is already a model with the same name: {output_model_folder}")
        return

    conf = OmegaConf.load(input_model_folder / "cfg.yaml")

    checkpoint_mode = 'best' 
    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(
        conf, 
        "", 
        checkpoint_mode=checkpoint_mode,
        pattern="val"
        )

    path_to_assets = get_path_to_assets() 
    path_to_flame_assets = get_path_to_assets() / "FLAME" 
    path_to_flame = Path(conf.model.shape_model.flame.flame_model_path).parents[2]
    path_data = Path("/is/cluster/fast/rdanecek/data/")
    path_project_data = Path("/ps/project/EmotionalFacialAnimation/data/")

    orig_ckpt_folder = conf.inout.checkpoint_dir 
    rel_ckpt = Path(checkpoint).relative_to(Path(orig_ckpt_folder))

    ## edit the paths in the config file 
    
    # flame related stuff
    conf.model.shape_model.flame.flame_model_path = str(Path(conf.model.shape_model.flame.flame_model_path).relative_to(path_to_flame))
    conf.model.shape_model.flame.flame_lmk_embedding_path = str(Path(conf.model.shape_model.flame.flame_lmk_embedding_path).relative_to(path_to_flame))
    
    if with_texture:
        if conf.model.shape_model.flame.tex_path is not None:
            conf.model.shape_model.flame.tex_path = str(Path(conf.model.shape_model.flame.tex_path).relative_to(path_to_flame))
    else:
        del conf.model.shape_model.flame['tex_path'] 
        del conf.model.shape_model.flame['tex_type'] 

    if conf.model.init_from is not None:
        conf.model.init_from = str(Path(conf.model.init_from).parent.name)

    input_name = Path(conf.inout.full_run_dir).name 
    output_name = Path(output_model_folder).name

    ## inout related stuff
    conf.inout.checkpoint_dir = str(Path(output_name) / 'checkpoints')
    conf.inout.full_run_dir = output_name
    if conf.inout.previous_run_dir is not None:
        # conf.inout.previous_run_dir  = str( Path(conf.inout.checkpoint_dir).relative_to(Path(conf.inout.previous_run_dir)) )
        conf.inout.previous_run_dir = False
    if conf.inout.submission_dir is not None:
        # conf.inout.submission_dir = str( Path(conf.inout.checkpoint_dir).relative_to(Path(conf.inout.previous_run_dir)) )
        conf.inout.submission_dir = False

    if conf.learning.losses.emotion_loss is not None: 
        conf.learning.losses.emotion_loss.network_path = str(path_to_emotion_feature)

    conf.inout.output_dir = False

    ## data related stuff
    try:
        conf.data.input_dir = str( Path(conf.data.input_dir).relative_to(path_data))
    except ValueError:
        conf.data.input_dir = str( Path(conf.data.input_dir).relative_to(path_project_data))
    conf.data.output_dir = str( Path(conf.data.output_dir).relative_to(path_data))

    ## save the config file in the output folder
    output_model_folder.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=conf, f=output_model_folder / "cfg.yaml")
    (output_model_folder / "checkpoints").mkdir(parents=True, exist_ok=True)

    ## copy the model checkpoint to the output folder
    output_ckpt_folder = (output_model_folder / "checkpoints" / rel_ckpt)
    output_ckpt_folder.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkpoint, output_ckpt_folder) 

    ## check the model loading 
    # try:
    face_rec_cfg = OmegaConf.load(output_model_folder  / "cfg.yaml")
    checkpoint = locate_checkpoint(face_rec_cfg, mode = face_rec_cfg.get("checkpoint_mode", "best"))
    # face_rec_cfg.learning.losses = {}
    # face_rec_cfg.learning.metrics = {}
    model = FaceReconstructionBase.instantiate(face_rec_cfg, checkpoint=checkpoint)
    # except Exception as e:
    #     # print(e)
    #     print("Model loading failed.")
    #     return
    print("Model loading successful.")
    print("Model export is finished")



def main():
    # input_model_folder = sys.argv[1]
    # output_model_folder = sys.argv[2]
    
    # pass 
    input_models = Path("/is/cluster/work/rdanecek/face_reconstruction/trainings/")
    input_model_folder =  "2023_10_26_15-15-19_-2340495024515473390_FaceReconstructionBase_Celeb_ResNet50_Pe_Aug"
    output_models = get_path_to_assets() / "FaceReconstruction" / "models" 
    output_model = "EMICA_flame2020"

    path_to_emotion_feature = Path("EmotionRecognition") / "image_based_networks" / "ResNet50" 

    input_model_folder = input_models / input_model_folder
    output_model_folder = output_models / output_model
    # export_model(input_model_folder, output_model_folder, 
    #              overwrite=True, 
    #              path_to_emotion_feature=path_to_emotion_feature, 
    #              with_texture=True)
    
    output_model = output_model + "_notexture"
    output_model_folder = output_models / output_model
    export_model(input_model_folder, output_model_folder, 
                overwrite=True, 
                path_to_emotion_feature=path_to_emotion_feature, 
                with_texture=False)


if __name__ == "__main__":
    main()
