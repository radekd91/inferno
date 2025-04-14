#!/bin/bash 

# if there is no argument, runc the docker with /bin/bash
if [ $# -eq 0 ]
then
    ## run docker with the codebase mounted
    docker run -it --gpus all -v $PWD/..:/workspace/repos/inferno inferno_docker /bin/bash
    # docker run -it --gpus all -v $PWD/..:/workspace/repos/inferno inferno_docker_egl /bin/bash
    # docker run -it --gpus all -v $PWD/..:/workspace/repos/inferno inferno_docker_cudagl /bin/bash
else
    ## run docker with the codebase mounted and execute the command
    docker run -it --gpus all -v $PWD/..:/workspace/repos/inferno inferno_docker $@
    # docker run -it --gpus all -v $PWD/..:/workspace/repos/inferno inferno_docker_egl $@
    # docker run -it --gpus all -v $PWD/..:/workspace/repos/inferno inferno_docker_cudagl $@
fi



