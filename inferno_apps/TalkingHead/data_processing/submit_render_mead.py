from inferno.utils.condor import execute_on_cluster
from pathlib import Path
import inferno_apps.Speech4D.data_processing.render_mead as script
import datetime
from omegaconf import OmegaConf
import time as t
import random

def submit(rec_method, subject_idx,
        bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/inferno"

    submission_dir_local_mount = "/is/cluster/work/rdanecek/mead_processing/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/mead_processing/submission"

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(random.randint(0,1000000))) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(script.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[2].name / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name


    submission_folder_local.mkdir(parents=True)
    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = 12000
    # gpu_mem_requirement_mb = None
    # cpus = cfg_coarse.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = 1
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_deca"
    cuda_capability_requirement = 7.5
    mem_gb = 16
    args = f"{rec_method} {subject_idx}"

    execute_on_cluster(str(cluster_script_path),
                       args,
                       str(submission_folder_local),
                       str(submission_folder_cluster),
                       str(cluster_repo_path),
                       python_bin=python_bin,
                       username=username,
                       gpu_mem_requirement_mb=gpu_mem_requirement_mb,
                       cpus=cpus,
                       mem_gb=mem_gb,
                       gpus=gpus,
                       num_jobs=num_jobs,
                       bid=bid,
                       max_time_h=max_time_h,
                       max_price=max_price,
                       job_name=job_name,
                       cuda_capability_requirement=cuda_capability_requirement, 
                        env="work38_clone", 
                        max_concurrent_jobs=90,
                       concurrency_tag = 'render_mead',
                       )   
    # t.sleep(1)


def main():

    # rec_methods = ['EMICA_v0_mp', 'EMICA_v0_mp_lr_cos_1.5', 'EMICA_v0_mp_lr_cos_1.5', \
    #         'EMICA_v0_mp_lr_mse_15', 'EMICA_v0_mp_lr_mse_20', 'EMOCA_v2_mp_lr_mse_15', 'EMICA_v0_mp_lr_cos_1.5', 'EMICA_v0_mp_lr_mse_15']

    # rec_methods = ['EMICA_v0_mp', ]
    rec_methods = ['EMICA_mead_mp_lr_mse_15', ]
    # rec_methods = ['emoca', 'spectre', 'EMOCA_v2_lr_mse_15_with_bfmtex', 'EMOCA_v2_lr_cos_1.5_with_bfmtex']
    # rec_methods = [ 'EMICA_v0_mp_lr_cos_1', 'EMOCA_v2_mp_with_bfmtex', \
    #                'EMOCA_v2_lr_mse_15_with_bfmtex', 'EMOCA_v2_lr_mse_20_with_bfmtex', \
    #                 'EMOCA_v2_lr_cos_1.5_with_bfmtex']
    # rec_methods = ['EMICA_v0_mp_lr_cos_1.5', 'EMICA_v0_mp_lr_cos_1', \
    #         'EMICA_v0_mp_lr_mse_15', 'EMICA_v0_mp_lr_mse_20', 'EMOCA_v2_mp_lr_mse_15', 'EMICA_v0_mp_lr_cos_1.5', 'EMICA_v0_mp_lr_mse_15']


    subjects = list(range(0, 47))

    for rec_method in rec_methods:
        for subject_idx in subjects:
            submit(rec_method, subject_idx, bid=2000)

  



if __name__ == "__main__":
    main()
