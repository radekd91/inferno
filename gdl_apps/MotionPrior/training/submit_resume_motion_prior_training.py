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
import gdl_apps.MotionPrior.training.resume_motion_prior_training  as script
import datetime
from omegaconf import OmegaConf
import time as t
import random
from omegaconf import DictConfig, OmegaConf, open_dict 
import sys

# submit_ = False
submit_ = True

if submit_:
    config_path = Path(__file__).parent / "submission_settings.yaml"
    if not config_path.exists():
        cfg = DictConfig({})
        cfg.cluster_repo_path = "todo"
        cfg.submission_dir_local_mount = "todo"
        cfg.submission_dir_cluster_side = "todo"
        cfg.python_bin = "todo"
        cfg.username = "todo"
        OmegaConf.save(config=cfg, f=config_path)
        
    user_config = OmegaConf.load(config_path)
    for key, value in user_config.items():
        if value == 'todo': 
            print("Please fill in the settings.yaml file")
            sys.exit(0)


def submit(resume_folder,
           stage = 0,
           resume_from_previous = False,
           force_new_location = False,
           bid=10):
    cluster_repo_path = user_config.cluster_repo_path
    submission_dir_local_mount = user_config.submission_dir_local_mount
    submission_dir_cluster_side = user_config.submission_dir_cluster_side

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(time)) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(script.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[2].name / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    # config_file = Path(submission_folder_local) / resume_folder / "cfg.yaml"
    config_file = Path(resume_folder) / "cfg.yaml"

    with open(config_file, 'r') as f:
        cfg = OmegaConf.load(f)

    python_bin = user_config.python_bin
    username = user_config.username
    gpu_mem_requirement_mb = cfg.learning.batching.gpu_memory_min_gb * 1024
    gpu_mem_requirement_mb_max = cfg.learning.batching.get('gpu_memory_max_gb' , None)
    # gpu_mem_requirement_mb = None
    cpus = cfg.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.learning.batching.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_motion_prior"
    cuda_capability_requirement = 7
    # mem_gb = 16
    mem_gb = 45
    args = f"{str(config_file.parent)}"

    args += f" {stage} {int(resume_from_previous)} {int(force_new_location)}"

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
                    #    env="work38",
                       env="work38_clone",
                       )
    t.sleep(1)


def resume_motion_prior_on_cluster():
    root = "/is/cluster/work/rdanecek/motion_prior/trainings/"
    
    model_names = []
    # bugs
    # # model_names += ["2023_02_05_19-50-07_-5643053022395030803_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-47-04_-6204082330817104839_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-46-40_-8068289807722994107_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-45-15_1020392668976619472_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-45-15_-8493098869856430542_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-45-07_3708617454039099621_L2lVqVae_Facef"]
    # # model_names += ["2023_02_05_19-44-51_4819075661105074885_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-44-51_-4469641301695294897_L2lVqVae_Facef"]
    # model_names += ["2023_02_05_19-43-35_3322208019009558883_L2lVqVae_Facef"]

    # # post bug fix, need finetuning
    # model_names += ["2023_02_07_20-28-53_-2021013419590325583_L2lVqVae_Facef"]
    # model_names += ["2023_02_07_20-29-01_-2576280032826500507_L2lVqVae_Facef"]
    # model_names += ["2023_02_07_20-27-54_-8292189443712743736_L2lVqVae_Facef"]
    # model_names += ["2023_02_07_20-25-33_-8702650742274879322_L2lVqVae_Facef"]
    # model_names += ["2023_02_07_19-47-20_-8793559135758698718_L2lVqVae_Facef_VQVAE"]
    # model_names += ["2023_02_07_19-31-25_3806813112112742638_L2lVqVae_Facef_VQVAE"]
    # model_names += ["2023_02_07_19-30-36_-221751824748459707_L2lVqVae_Facef_VQVAE"]
    # model_names += ["2023_02_07_19-30-07_-1046633450114303061_L2lVqVae_Facef_VQVAE"]
    # model_names += ["2023_02_07_19-28-21_-2023277178393284087_L2lVqVae_Facef_VQVAE"]
    # model_names += ["2023_02_07_19-27-54_2042140273458874000_L2lVqVae_Facef_VQVAE"]
    # model_names += ["2023_02_07_19-27-39_-5992224788341585803_L2lVqVae_Facef_VQVAE"]
    # model_names += ["2023_02_07_19-26-18_7959811539499577467_L2lVqVae_Facef_VQVAE"]

    # # jobs that crashed mid training for no reason.
    # model_names += ['2023_02_13_09-50-47_588315614196140873_L2lVqVae_Facef_dVAE']

    # # needs a bit more finetuning, interesting results on vocaset for VQ VAE 
    # model_names += ['2023_02_08_23-53-45_-7897278066327257130_L2lVqVae_Facef_VQVAE']
    # model_names += ['2023_02_11_17-29-21_7975012759683643004_L2lVqVae_Facef_VQVAE']
    # model_names += ['2023_02_11_17-27-50_-7642823735391707989_L2lVqVae_Facef_VQVAE']
    # first AEs on MEAD: 
    # model_names += ["2023_02_12_20-01-17_-4462360882556841344_L2lVqVae_MEADP_AE"]

    # # models on mead that have converged and need testing
    # model_names += ["2023_02_14_21-37-22_-4437007122232758841_L2lVqVae_MEADP_VAE"]
    # # model_names += ["2023_02_14_21-36-30_-6645345463044072351_L2lVqVae_MEADP_VAE"]
    # # model_names += ["2023_02_14_21-36-04_-6167662386280290661_L2lVqVae_MEADP_VAE"]

    # # model_names += ["2023_02_14_21-36-06_3736766602660741736_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_02_14_21-37-42_-5906849939056023298_L2lVqVae_MEADP_VAE"]
    # # model_names += ["2023_02_14_21-35-48_-8091441205341210773_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_02_14_21-16-46_7403691266508414587_L2lVqVae_MEADP_AE"]
    
    # # model_names += ["2023_02_14_21-36-04_-2594523615553095319_L2lVqVae_MEADP_VAE"]
    # # model_names += ["2023_02_14_21-37-57_3688715351484736532_L2lVqVae_MEADP_VAE"]    

    # # # # models on mead that need more finetuning 
    # # model_names += ["2023_02_14_21-16-46_-2546611371074170025_L2lVqVae_MEADP_AE"]
    # # model_names += ["2023_02_14_21-38-17_-3365277498953081436_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_02_14_21-38-16_3819249676676666576_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_02_14_21-37-53_-8805504103833644898_L2lVqVae_MEADP_VAE"]
    # # model_names += ["2023_02_14_21-37-21_4936645424129419426_L2lVqVae_MEADP_VAE"]

    # model_names += [""]
    # model_names += [""]
    # model_names += [""]
    
    # Runs on EMICA-MEAD
    # ## random_by_identityV2_sorted_70_15_15
    # model_names += ["2023_04_19_00-28-36_2620263824860983578_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-08_3336320914244940859_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-09_-8239109196262280143_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-20_-5803196173148535762_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-29_1224983122974393223_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-48_-3920248191400610296_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-28-36_7641989056675034401_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-20-32_140076761040104708_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-33-45_3752709138041149366_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_03-12-13_-2352703592579167064_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_08-06-50_-2912228835425113432_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_07-20-52_7560882024093976248_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-05_-4997744900722365412_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-01_7323983481315963013_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-58-52_-431614051460033103_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-02_-967541212892117555_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-20_263244182211912227_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-31_985595843842576922_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-31_-5410683084982286477_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-33_6018236609208253182_L2lVqVae_MEADP_VAE"]

    # ## random_by_sequence_sorted_70_15_15
    # model_names += ["2023_04_19_00-00-05_-7677114385763699840_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-32_-3838203027570469007_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-31_-662511880207928567_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-31_107108105949616955_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-17_6691846097152544168_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-14_-453801061652362047_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-35_-8426467404534453753_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-30_-5904904494942409085_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-18_-6635429708944695063_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-43_-4368952989961137110_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-39_8943690337172483192_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-13_-2490096116192764107_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-39_8354726890857327493_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-16_6322900138811770517_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-16_1279880418853352871_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-18_5111038154583139267_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-33_-271736565210170671_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-20_-6785385252999123719_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-27_-2451312006709792497_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-27_516267371774217953_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-26_6809389698517385692_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-16_2684490964169142148_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-18_-6736424312364364268_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-33_4894095615562508970_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-17_-904605746287992741_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-24_-1201797576850643782_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-27_1086544893838595087_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-20_3822530478313716560_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-32_397696636039369469_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-17_-8715285656450941711_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-46_82870774431388047_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_18-21-20_1781989610217374714_L2lVqVae_MEADP_VAE"]

    # # specific_identity_sorted_80_20_M003
    # model_names += ["2023_04_19_00-01-33_8741517872782649963_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-49_3802241023296877179_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-37_-1230825983744144977_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-38_-3619334619956727409_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-36_8408178486888822267_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-37_-8322536348751144017_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-12_5970413263166749159_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-16_4566069169766000880_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-51_-3877074160401674034_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-14_-5622630547008254665_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-13_-3599943234089060607_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-13_7573360572940968708_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-13_7816736763074877860_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-13_3517003665362732815_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-10_-3866199628095394837_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-10_-2921706562687628710_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-00-16_2632277629187561565_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_19_00-08-19_1645645583971586379_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-51_8575967001023234628_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_18_23-59-49_4450890243668563649_L2lVqVae_MEADP_VAE"]


    ## larger KL models
    # model_names += ["2023_04_21_20-00-04_3803186418270666936_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-50_7771662601250590079_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-23_9151842974139292717_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-50_6507599727503458432_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-46_1180692456060681856_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-53_-8116924601889576700_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-47_4460674736343908381_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-49_5818615017774824991_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-52_671074197014815123_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-44_-7595534771615157500_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-48_2796810154160784651_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-00-02_-2154899297476125683_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-48_5864406916679810456_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-43_-9076677965478033308_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-48_6180701162000832912_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-00-17_-6133720105563802942_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-47_6979335684684731789_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-27_8187692813538748696_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-44_8405051855712043694_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-02-07_-8366803912264596368_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-28_-7815412565501690499_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-43_-287176949594610082_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-29_1453394315026872380_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-48_2389157612997268821_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-26_1448988728415557745_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-30_-4183045163457513202_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-29_3923089970841786309_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-48_-2316223941260356650_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-52_3629957734310315404_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-31_3516849649677689413_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-56_-6195352049081442109_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-25_-3148276364739998748_L2lVqVae_MEADP_VAE"]

    # model_names += ["2023_04_21_19-59-43_3505016096948649279_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-48_-1713444156454985802_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-48_5773970447279375988_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-26_-2919164218495577830_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-48_5457352997936553704_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_19-59-48_7337147168253027699_L2lVqVae_MEADP_VAE"]
    # model_names += ["2023_04_21_20-01-27_7092076731248702256_L2lVqVae_MEADP_VAE"]



    rename_video_result_folder = False
    # rename_video_result_folder = True

    bid = 1000

    # # # continue training
    # stage = 0 
    # resume_from_previous = False
    # force_new_location = False

    # ## test 
    stage = 1
    resume_from_previous = True
    force_new_location = False

    for model_folder in model_names:
        
        if rename_video_result_folder:
            old_folder = Path(root + model_folder) / "videos" 
            if old_folder.exists():
                # find a new name
                new_folder = Path(root + model_folder) / "videos"
                i = 0
                while new_folder.exists():
                    new_folder = Path(root + model_folder) / f"videos_{i:02d}"
                    i += 1
                old_folder.rename(new_folder) 
                print("Renaming", old_folder, "to", new_folder)

        if submit_:
            submit(root + model_folder, stage, resume_from_previous, force_new_location, bid=bid)
        else: 
            script.resume_training(root + model_folder, stage, resume_from_previous, force_new_location)


if __name__ == "__main__":
    resume_motion_prior_on_cluster()

