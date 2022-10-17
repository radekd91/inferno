import os, sys 
import omegaconf
from pathlib import Path
from gdl_apps.TalkingHead.training.train_talking_head import train_model


def main(): 
    path_to_models = "/is/cluster/work/rdanecek/talkinghead/trainings/"
    trained_model_folder = "2022_10_14_12-29-41_-798122913510811591_FaceFormer_Celeb_Awav2vec2T_Elinear_DFlameBertDecoder_SnoPPE_Tff_predEJ_LV"
    # trained_model_folder = "2022_10_14_12-29-33_8541647761690030000_FaceFormer_LRS3P_Awav2vec2T_Elinear_DFlameBertDecoder_SnoPPE_Tff_predEJ_LV"

    # load the config file
    config_path = Path(path_to_models) / trained_model_folder / "cfg.yaml"
    config = omegaconf.OmegaConf.load(config_path)
    
    start_from = 1 # this is the testing phase
    resume_from_previous = True 
    force_new_location = False
    train_model(config, start_from, resume_from_previous, force_new_location)


if __name__ == "__main__":
    main()
