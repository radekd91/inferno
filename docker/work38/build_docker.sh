#!/bin/bash
echo "Building docker image"
cp ../../conda-environment_py38_cu11.yaml environment.yaml 
cp ../../requirements38.txt requirements.txt
docker build --build-arg USER=$(whoami) --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)  -t inferno_docker .
# docker build -t inferno_docker_egl . > build_log_egl.out 2> build_log_egl.err
# docker build --no-cache -t inferno_docker_egl . > build_log_egl.out 2> build_log_egl.err
# docker build -t inferno_docker_cudagl . > build_log_cudagl.out 2> build_log_cudagl.err

