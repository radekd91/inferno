"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
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
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import gdl_apps.TalkingHead.evaluation.read_eval_lrs3_results as script
import datetime
from omegaconf import OmegaConf
import time as t
import random
from omegaconf import DictConfig, OmegaConf, open_dict 
import sys
import shutil

# submit_ = False
submit_ = True

# if submit_:
#     config_path = Path(__file__).parent / "submission_settings.yaml"
#     if not config_path.exists():
#         cfg = DictConfig({})
#         cfg.cluster_repo_path = "todo"
#         cfg.submission_dir_local_mount = "todo"
#         cfg.submission_dir_cluster_side = "todo"
#         cfg.python_bin = "todo"
#         cfg.username = "todo"
#         OmegaConf.save(config=cfg, f=config_path)
        
#     user_config = OmegaConf.load(config_path)
#     for key, value in user_config.items():
#         if value == 'todo': 
#             print("Please fill in the settings.yaml file")
#             sys.exit(0)


def submit(resume_folder, subset, max_videos,
           emotion_index_list=None,
           bid=10, 
           max_price=None,
           ):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    submission_dir_local_mount = "/is/cluster/work/rdanecek/talking_head_eval/submission_lrs3_lipread"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/talking_head_eval/submission_lrs3_lipread"

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(time)) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(script.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[2].name / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    python_bin = 'python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = 30 * 1024
    gpu_mem_requirement_mb_max = 40000
    # gpu_mem_requirement_mb = None
    cpus = 8 #cfg.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = 1
    num_jobs = 1
    max_time_h = 36
    job_name = "train_talking_head"
    cuda_capability_requirement = 7
    mem_gb = 20

    args = f" {resume_folder} {str(subset)} {max_videos}"
    if emotion_index_list is not None:
        args += f" {','.join([str(x) for x in emotion_index_list])}"

    #    env="work38",
    #    env="work38_clone",
    # env = "/is/cluster/fast/rdanecek/envs/work38_fast" 
    env = "/is/cluster/fast/rdanecek/envs/work38_fast_clone"
    # if env is an absolute path to the conda environment
    if Path(env).exists() and Path(env).resolve() == Path(env):
        python_bin = str(Path(env) / "bin/python")
        assert Path(python_bin).exists(), f"Python binary {python_bin} does not exist"

    execute_on_cluster(str(cluster_script_path),
                       args,
                       str(submission_folder_local),
                       str(submission_folder_cluster),
                       str(cluster_repo_path),
                       python_bin=python_bin,
                       username=username,
                       gpu_mem_requirement_mb=gpu_mem_requirement_mb,
                       gpu_mem_requirement_mb_max=gpu_mem_requirement_mb_max,
                       cpus=cpus,
                       mem_gb=mem_gb,
                       gpus=gpus,
                       num_jobs=num_jobs,
                       bid=bid,
                       max_time_h=max_time_h,
                       max_price=max_price,
                       job_name=job_name,
                       cuda_capability_requirement=cuda_capability_requirement,
                       env=env, 
                       )
    t.sleep(1)



def run_talking_head_eval():

    root = "/is/cluster/work/rdanecek/talkinghead/trainings/"
    
    resume_folders = []

    # #### final PAPER models ####
    
    # # ### final ENSPARC models (WITH prior, lip reading, video emotion, disentanglement), trainable w2v (initially) 
    # # check the results - wl = wld = 0.000025, we, wed = 0.0000025
    resume_folders += ["2023_05_18_01-26-32_-6224330163499889169_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]
    
    # # check the results - wl = wld = 0.00005, we, wed = 0.000005
    resume_folders += ["2023_05_16_23-13-38_-2116095221923261916_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]

    ### ENSPARC WITHOUT disentanglement (WITH prior, lip reading, video emotion, WITHOUT disentanglement), trainable w2v (initially)  
    ## check the results - wl = 0.00005, we = 0.000005
    resume_folders += ["2023_05_16_23-13-12_-2523817769843276359_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]
    ## check the results - wl = 0.00005, we = 0.0000025
    resume_folders += ["2023_05_16_23-12-26_6547601109468810874_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]
    ## check the results - wl = 0.0001, we = 0.000005
    # resume_folders += ["2023_05_16_23-12-24_-8306171087132288898_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]

    ### ENSPARC WITHOUT video emotion (WITH prior, lip reading, WITHOUT video emotion, disentanglement), trainable w2v (initially)
    ## without disentanglement wl = 0.00005
    resume_folders += ["2023_05_18_01-27-11_7629119778539369902_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]
    ## with disentanglement wl = wld = 0.000025
    resume_folders += ["2023_05_18_01-27-28_8727579300900226200_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]

    ### ENSPARC WITHOUT lip reading (WITH prior, video emotion, WITHOUT lip reading, disentanglement), trainable w2v (initially)
    ## without disentanglement wl = 0.00005
    resume_folders += ["2023_05_18_01-28-06_-6355446600867862848_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]
    ## with disentanglement wl = wld = 0.000025
    resume_folders += ["2023_05_18_01-27-47_-4003478001355428123_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]

    # ENSPARC - prior (FLAMEBERT) (WITHOUT prior, but other stuff as final ENSPARC model) - to be revised
    # with disentanglement
    resume_folders += ["2023_05_16_20-26-30_5452619485602726463_FaceFormer_MEADP_Awav2vec2_Elinear_DFlameBertDecoder_Seml_PPE_Tff_predEJ_LVmmmLmm"] 
    # without disenganlement
    resume_folders += ["2023_05_16_20-34-11_8508648578323040497_FaceFormer_MEADP_Awav2vec2_Elinear_DFlameBertDecoder_Seml_PPE_Tff_predEJ_LVmmmLmm"]

    # ENSPARC - prior (FLAMEBERT) - no lip reading loss no disentanglement we = 0.00005
    resume_folders += ["2023_05_16_20-40-41_8088631020349769941_FaceFormer_MEADP_Awav2vec2_Elinear_DFlameBertDecoder_Seml_PPE_Tff_predEJ_LVmmmLmm"]

    # ENSPARC - prior (FLAMEBERT) - no lip reading loss with disentanglement we = wed= 0.000025
    resume_folders += ["2023_05_16_20-45-34_2799073586488120156_FaceFormer_MEADP_Awav2vec2_Elinear_DFlameBertDecoder_Seml_PPE_Tff_predEJ_LVmmmLmm"]

    # ENSPARC - prior (FLAMEBERT) - no video emotion loss with disentanglement wl = wl= 0.000025
    resume_folders += ["2023_05_16_20-44-52_4388314819577499314_FaceFormer_MEADP_Awav2vec2_Elinear_DFlameBertDecoder_Seml_PPE_Tff_predEJ_LVmmmLmm"]

    # ENSPARC - prior (FLAMEBERT) - no video emotion loss no disentanglement wl = 0.00005
    resume_folders += ["2023_05_16_20-42-05_6372071075743624200_FaceFormer_MEADP_Awav2vec2_Elinear_DFlameBertDecoder_Seml_PPE_Tff_predEJ_LVmmmLmm"]

    ## ENSPARC WITHOUT video emotion and with STATIC emotion loss 
    # with disentanglement, wl = wld = 0.00005,  we = wed = 0.00005
    resume_folders += ["2023_05_18_01-58-31_6242149878645900496_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmEmmLmm"]
    # without disentanglement, wl = 0.00005, we =  0.00005
    resume_folders += ["2023_05_18_01-58-51_2653017516632768605_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmEmmLmm"]
    # with disentanglement, no lip reading, we = wed = 0.00005
    resume_folders += ["2023_05_18_01-58-03_-2039300938773449621_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmEmmLmm"]
    # without disentanglement, no lip reading, we =  0.00005
    resume_folders += ["2023_05_18_01-58-34_-6870608487115958370_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmEmmLmm"]

    ## ENSPARC WITHOUT prior (FlameBERT), video emotion and with STATIC emotion loss 
    # with disentanglement, wl = wld = 0.00005,  we = wed = 0.00005
    resume_folders += ["2023_05_18_02-14-58_-5741489981604096939_FaceFormer_MEADP_Awav2vec2_Elinear_DFlameBertDecoder_Seml_PPE_Tff_predEJ_LVmEmmLmm"]
    # without disentanglement, wl = 0.00005, we = 0.00005
    resume_folders += ["2023_05_18_02-16-41_-8640432717482685377_FaceFormer_MEADP_Awav2vec2_Elinear_DFlameBertDecoder_Seml_PPE_Tff_predEJ_LVmEmmLmm"]
    # with disentanglement, no lip reading, we = wed = 0.00005
    resume_folders += ["2023_05_18_02-16-41_1632039793563347593_FaceFormer_MEADP_Awav2vec2_Elinear_DFlameBertDecoder_Seml_PPE_Tff_predEJ_LVmEmmLmm"]
    # without disentanglement, no lip reading, we = 0.00005
    resume_folders += ["2023_05_18_02-15-52_-7116303129003010747_FaceFormer_MEADP_Awav2vec2_Elinear_DFlameBertDecoder_Seml_PPE_Tff_predEJ_LVmEmmLmm"]


    ### ENSPARC WITHOUT ANY PERCEPTUAL LOSSES (lip reading, video emotion, disentanglement), trainable w2v (initially)
    resume_folders += ["2023_05_13_21-00-49_-6819445356403438364_FaceFormer_MEADP_Awav2vec2T_Elinear_DBertPriorDecoder_Seml_NPE_predEJ_LVm"]
    # FlameBert WITHOUT prior
    resume_folders += ["2023_05_10_14-26-58_7312238994463268480_FaceFormer_MEADP_Awav2vec2T_Elinear_DFlameBertDecoder_Seml_PPE_predEJ_LVm"]
    # FlameFormer with emotions - not well converged
    resume_folders += ["2023_05_10_13-21-50_1717396956261008837_FaceFormer_MEADP_Awav2vec2T_Elinear_DFlameFormerDecoder_Seml_PPE_predEJ_LV"]
    # FaceFormer with emotions 
    resume_folders += ["2023_05_10_13-10-08_8067654090108546902_FaceFormer_MEADP_Awav2vec2T_Elinear_DFaceFormerDecoder_Seml_PPE_predV_LV"]

    # # # emotion_index_list = list(range(8))


    # # # #### MODELS NOT CONDITIONED ON EMOTIONS AT ALL 
    # # # ## 1) not conditioned on emotions, 128d, 8 heads,  trainable w2v
    # # # ## FlameBERTPrior,
    # resume_folders += ["2023_05_12_11-34-40_8409157253283996274_FaceFormer_MEADP_Awav2vec2T_Elinear_DBertPriorDecoder_Seml_NPE_predEJ_LVm"]
    # # ## FlameBERTPrior, frozen w2v 
    # # resume_folders += ["2023_05_12_11-33-17_189824166655322547_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_predEJ_LVm"]
    # ## FlameBERT
    # resume_folders += ["2023_05_10_14-27-51_-7474011499178916721_FaceFormer_MEADP_Awav2vec2T_Elinear_DFlameBertDecoder_Seml_PPE_predEJ_LVm"]
    # # ## FlameFormer
    # resume_folders += ["2023_05_10_13-24-04_5562322546915629563_FaceFormer_MEADP_Awav2vec2T_Elinear_DFlameFormerDecoder_Seml_PPE_predEJ_LV"]
    # # ## FaceFormer
    resume_folders += ["2023_05_10_13-16-00_-3885098104460673227_FaceFormer_MEADP_Awav2vec2T_Elinear_DFaceFormerDecoder_Seml_PPE_predV_LV"]

    # # emotion_index_list = [0]

    # bid = 2000
    # bid = 150
    # bid = 28
    bid = 100
    # max_price = 250
    # max_price = 200
    max_price = 500

    
    # subset = "trainval"
    # max_videos = 1000
    
    subset = "test"
    max_videos = "all"
    for resume_folder in resume_folders:
        if submit_:
            submit(resume_folder, subset, max_videos, bid=bid, max_price=max_price)
        else: 
            script.read_results(Path(root) / resume_folder, subset, max_videos)



if __name__ == "__main__":
    run_talking_head_eval()

