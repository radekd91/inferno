import os
import subprocess
import stat

condor_template = """
executable = <<SCRIPTNAME>>
arguments = <<ARGS>>
error = <<PATH>>/<<JOBNAME>><<PROCESS_ID>>.err
output = <<PATH>>/<<JOBNAME>><<PROCESS_ID>>.out
log = <<PATH>>/<<JOBNAME>><<PROCESS_ID>>.log
request_memory = <<MEMORYMBS>>
request_cpus = <<CPU_COUNT>>
request_gpus = <<GPU_COUNT>>
<<REQUIREMENTS>>
<<CONCURRENCY>>
+MaxRunningPrice = <<MAX_PRICE>>
+RunningPriceExceededAction = "kill"
queue <<NJOBS>>
"""
# requirements=TARGET.CUDAGlobalMemoryMb>2000
# requirements=TARGET.CUDACapability>=6.0

script_template = """
source /home/<<USERNAME>>/.bashrc
source /home/<<USERNAME>>/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate <<ENV>>
export SUBMISSION_DIR=$PWD
export PYTHONPATH=$PYTHONPATH:<<REPO_ROOT>>
<<MODULES>>
<<PYTHON_BIN>> <<SCRIPT_NAME>> $@
# OUTFOLDER=$(cat out_folder.txt)
# ln -s $PWD $OUTFOLDER/submission 
# ln -s $OUTFOLDER results
# source deactivate
"""


apptainer_script_template = """
#!/bin/bash
echo "Running the script inside the container"
nvidia-smi
source ~/.bashrc
#eval "$(mamba shell.bash hook)"
mamba activate <<ENV>>
echo "Activated the environment"
which python
export SUBMISSION_DIR=$PWD
export PYTHONPATH=$PYTHONPATH:<<REPO_ROOT>>
<<MODULES>>
<<PYTHON_BIN>> <<SCRIPT_NAME>> $@
# OUTFOLDER=$(cat out_folder.txt)
# ln -s $PWD $OUTFOLDER/submission 
# ln -s $OUTFOLDER results
# source deactivate
echo "The script inside the container has finished"
"""


apptainer_command_template = """
echo "Running apptainer"
apptainer exec \
    --no-home  \
    --contain \
    --writable-tmpfs \
    <<MOUNT_OPTIONS>> \
    <<CONTAINER_PATH>> \
    /bin/bash <<SCRIPT_NAME>> $@
echo "Apptainer exited"
"""

# ######################################################
# #Cluster paramters example
# ######################################################
# CPU_COUNT = 4
# GPU_COUNT = 1
# MAX_MEM_GB = 20
# NUM_JOBS = 1
# MAX_TIME_H = 36
# MAX_PRICE = 5000
# BID = 10
# USERNAME = 'rdanecek'
# ######################################################


# def execute_on_cluster(config_fname):
def execute_on_cluster(cluster_script_path, args, submission_dir_local_mount,
                       submission_dir_cluster_side=None,
                       cluster_repo_dir='/home/rdanecek/workspace/repos/inferno',
                       cpus=1, gpus=0, mem_gb=4, num_jobs=1, bid=10, max_time_h=None,
                       max_price=None,
                       job_name="skynet",
                       python_bin='python',
                       #env='work36',
                    #    env='work36_cu11',
                       env='work38',
                       username='rdanecek',
                       gpu_mem_requirement_mb=None,
                       gpu_mem_requirement_mb_max=None,
                       cuda_capability_requirement=None,
                       max_concurrent_jobs=None,
                       concurrency_tag = None,
                       modules_to_load = None,
                       chmod=True):
    max_price = max_price or 300
    assert bid <= max_price, "Bid must be lower than max price"
    modules_to_load = modules_to_load or []
    submission_dir_cluster_side = submission_dir_cluster_side or submission_dir_local_mount
    logdir = 'logs'

    st = script_template
    # st = st.replace('<<CONFIG_FNAME>>', config_fname)
    st = st.replace('<<REPO_ROOT>>', cluster_repo_dir)
    st = st.replace('<<PYTHON_BIN>>', python_bin)
    st = st.replace('<<SCRIPT_NAME>>', cluster_script_path)
    st = st.replace('<<ENV>>', env)
    st = st.replace('<<USERNAME>>', username)
    modules = ""
    if len(modules_to_load) > 0:
        modules = f"module load {' '.join(modules_to_load)}"
    st = st.replace('<<MODULES>>', modules)
    script_fname = os.path.join(submission_dir_local_mount, 'run.sh')


    cs = condor_template
    cs = cs.replace('<<PATH>>', logdir)
    cs = cs.replace('<<ARGS>>', args)
    cs = cs.replace('<<SCRIPTNAME>>', os.path.basename(script_fname))
    cs = cs.replace('<<JOBNAME>>', job_name)
    cs = cs.replace('<<CPU_COUNT>>', str(int(cpus)))
    cs = cs.replace('<<GPU_COUNT>>', str(int(gpus)))
    cs = cs.replace('<<MEMORYMBS>>', str(int(mem_gb * 1024)))
    cs = cs.replace('<<MAX_TIME>>', str(int(max_time_h * 3600))) #TODO: fix this, i'ts missing in the script
    cs = cs.replace('<<MAX_PRICE>>', str(int(max_price)))
    cs = cs.replace('<<NJOBS>>', str(num_jobs))


    if num_jobs>1:
        cs = cs.replace('<<PROCESS_ID>>', ".$(Process)")
    else:
        cs = cs.replace('<<PROCESS_ID>>', "")

    requirements = []

    if cuda_capability_requirement:
        if isinstance(cuda_capability_requirement, int):
            cuda_capability_requirement = str(cuda_capability_requirement) + ".0"
        requirements += [f"( TARGET.CUDACapability>={cuda_capability_requirement} )"]
    if gpu_mem_requirement_mb:
        requirements += [f"( TARGET.CUDAGlobalMemoryMb>={gpu_mem_requirement_mb} )"]
    if gpu_mem_requirement_mb_max:
        requirements += [f"( TARGET.CUDAGlobalMemoryMb<={gpu_mem_requirement_mb_max} )"]
    if len(requirements) > 0:
        requirements = " && ".join(requirements)
        requirements = "requirements=" + requirements

    cs = cs.replace('<<REQUIREMENTS>>', requirements)
    condor_fname = os.path.join(submission_dir_local_mount, 'run.condor')

    concurrency_string = ""
    if concurrency_tag is not None and max_concurrent_jobs is not None:
        concurrency_limits = 10000 // max_concurrent_jobs
        concurrency_string += f"concurrency_limits = user.{concurrency_tag}:{concurrency_limits}"
    cs = cs.replace('<<CONCURRENCY>>', concurrency_string)

    # write files
    with open(script_fname, 'w') as fp:
        fp.write(st)
    os.chmod(script_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC | stat.S_IXUSR | stat.S_IRUSR)  # make executable
    with open(condor_fname, 'w') as fp:
        fp.write(cs)
    os.chmod(condor_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC | stat.S_IXUSR | stat.S_IRUSR)  # make executable

    if chmod:
        cmd = f'cd {submission_dir_cluster_side} && ' \
              f'echo $PWD > pwd.txt && ' \
              f'mkdir {logdir} && ' \
              f'chmod +x {os.path.basename(script_fname)} && ' \
              f'chmod +x {os.path.basename(condor_fname)} && ' \
              f'condor_submit_bid {bid} {os.path.basename(condor_fname)}'
    else:
        cmd = f'cd {submission_dir_cluster_side} && ' \
              f'echo $PWD > pwd.txt && ' \
              f'mkdir {logdir} && ' \
              f'condor_submit_bid {bid} {os.path.basename(condor_fname)}'

    print("Called the following on the cluster: ")
    print(cmd)
    subprocess.call(["ssh", "%s@login.cluster.is.localnet" % (username,)] + [cmd])
    # subprocess.call(["ssh", "%s@login1.cluster.is.localnet" % (username,)] + [cmd])
    # subprocess.call(["ssh", "%s@login2.cluster.is.localnet" % (username,)] + [cmd])
    # subprocess.call(["ssh", "%s@login4.cluster.is.localnet" % (username,)] + [cmd])
    print("Done")


def execute_on_cluster_apptainer(
    cluster_script_path, args, submission_dir_local_mount,
    submission_dir_cluster_side=None,
    cluster_repo_dir='/home/rdanecek/workspace/repos/inferno',
    container_path='/path/to/container.sif',
    mount_locations=None,
    cpus=1, gpus=0, mem_gb=4, num_jobs=1, bid=10, max_time_h=None,
    max_price=None,
    job_name="skynet",
    python_bin='python',
    env='work38',
    username='rdanecek',
    gpu_mem_requirement_mb=None,
    gpu_mem_requirement_mb_max=None,
    cuda_capability_requirement=None,
    max_concurrent_jobs=None,
    concurrency_tag = None,
    modules_to_load = None,
    chmod=True, 
    private_key=None
    ):
    max_price = max_price or 300
    assert bid <= max_price, "Bid must be lower than max price"
    modules_to_load = modules_to_load or []
    submission_dir_cluster_side = submission_dir_cluster_side or submission_dir_local_mount
    logdir = 'logs'
    mount_locations = mount_locations or {}

    # Prepare Apptainer command
    
    # mount_options = " ".join([f"--bind {paths[0]}:{paths[1]}" for name, paths in mount_locations.items()])
    mount_options = " --bind " + ",".join([f"{local}:{cluster}" for name, (local, cluster) in mount_locations.items()])

    st = apptainer_script_template
    st = st.replace('<<REPO_ROOT>>', cluster_repo_dir)
    st = st.replace('<<PYTHON_BIN>>', python_bin)
    st = st.replace('<<SCRIPT_NAME>>', cluster_script_path)
    st = st.replace('<<ENV>>', env)
    st = st.replace('<<USERNAME>>', username)
    modules = ""
    if len(modules_to_load) > 0:
        modules = f"module load {' '.join(modules_to_load)}"
    st = st.replace('<<MODULES>>', modules)
    script_fname = os.path.join(submission_dir_local_mount, 'run.sh')

    apptainer_command = apptainer_command_template
    apptainer_command = apptainer_command.replace('<<MOUNT_OPTIONS>>', mount_options)
    apptainer_command = apptainer_command.replace('<<CONTAINER_PATH>>', container_path)
    apptainer_command = apptainer_command.replace('<<SCRIPT_NAME>>', script_fname)

    apptainer_command_fname = os.path.join(submission_dir_local_mount, 'run_apptainer.sh')


    # Prepare Condor submission file
    cs = condor_template
    cs = cs.replace('<<PATH>>', logdir)
    cs = cs.replace('<<ARGS>>', args)
    cs = cs.replace('<<SCRIPTNAME>>', os.path.basename(apptainer_command_fname))
    cs = cs.replace('<<JOBNAME>>', job_name)
    cs = cs.replace('<<CPU_COUNT>>', str(int(cpus)))
    cs = cs.replace('<<GPU_COUNT>>', str(int(gpus)))
    cs = cs.replace('<<MEMORYMBS>>', str(int(mem_gb * 1024)))
    cs = cs.replace('<<MAX_PRICE>>', str(int(max_price)))
    cs = cs.replace('<<NJOBS>>', str(num_jobs))

    if num_jobs > 1:
        cs = cs.replace('<<PROCESS_ID>>', ".$(Process)")
    else:
        cs = cs.replace('<<PROCESS_ID>>', "")

    requirements = []
    if cuda_capability_requirement:
        if isinstance(cuda_capability_requirement, int):
            cuda_capability_requirement = str(cuda_capability_requirement) + ".0"
        requirements += [f"( TARGET.CUDACapability>={cuda_capability_requirement} )"]
    if gpu_mem_requirement_mb:
        requirements += [f"( TARGET.CUDAGlobalMemoryMb>={gpu_mem_requirement_mb} )"]
    if gpu_mem_requirement_mb_max:
        requirements += [f"( TARGET.CUDAGlobalMemoryMb<={gpu_mem_requirement_mb_max} )"]
    if len(requirements) > 0:
        requirements = " && ".join(requirements)
        requirements = "requirements=" + requirements

    cs = cs.replace('<<REQUIREMENTS>>', requirements)
    condor_fname = os.path.join(submission_dir_local_mount, 'run.condor')

    concurrency_string = ""
    if concurrency_tag is not None and max_concurrent_jobs is not None:
        concurrency_limits = 10000 // max_concurrent_jobs
        concurrency_string += f"concurrency_limits = user.{concurrency_tag}:{concurrency_limits}"
    cs = cs.replace('<<CONCURRENCY>>', concurrency_string)

    # Write files
    with open(script_fname, 'w') as fp:
        fp.write(st)
    os.chmod(script_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC | stat.S_IXUSR | stat.S_IRUSR)  # make executable
    with open(condor_fname, 'w') as fp:
        fp.write(cs)
    os.chmod(condor_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC | stat.S_IXUSR | stat.S_IRUSR)  # make executable
    with open(apptainer_command_fname, 'w') as fp:
        fp.write(apptainer_command)
    os.chmod(apptainer_command_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC | stat.S_IXUSR | stat.S_IRUSR)  # make executable

    if chmod:
        cmd = f'cd {submission_dir_cluster_side} && ' \
              f'echo $PWD > pwd.txt && ' \
              f'mkdir {logdir} && ' \
              f'chmod +x {os.path.basename(script_fname)} && ' \
              f'chmod +x {os.path.basename(condor_fname)} && ' \
              f'condor_submit_bid {bid} {os.path.basename(condor_fname)}'
    else:
        cmd = f'cd {submission_dir_cluster_side} && ' \
              f'echo $PWD > pwd.txt && ' \
              f'mkdir {logdir} && ' \
              f'condor_submit_bid {bid} {os.path.basename(condor_fname)}'

    print("Called the following on the cluster: ")
    print(cmd)
    ssh_cmd = ["ssh",]
    if private_key is not None: 
        ssh_cmd += ["-i", private_key] 
    ssh_cmd += ["%s@login.cluster.is.localnet" % (username,),]
    # subprocess.call(["ssh", "%s@login.cluster.is.localnet" % (username,)] + [cmd])
    subprocess.call(ssh_cmd + [cmd])
    print("Done")

