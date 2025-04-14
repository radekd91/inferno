# Start from an NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
LABEL maintainer="RadekDanecek"

# Set a non-interactive shell to avoid prompts during the build process
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Berlin 

# Pre-configure tzdata to avoid prompts during the build process
RUN echo "tzdata tzdaata/Areas select Europe" | debconf-set-selections && \ 
    echo "tzdata tzdata/Zones/Europe select Berlin" | debconf-set-selections && \
    echo 'keyboard-configuration keyboard-configuration/variant select pc105' | debconf-set-selections && \
    echo 'keyboard-configuration keyboard-configuration/layout select us' | debconf-set-selections 

ARG USER 
ARG USER_ID 
ARG GROUP_ID

# Create a user with the same user id as the host user
RUN groupadd -g ${GROUP_ID} ${USER} && \
    useradd -l -u ${USER_ID} -g ${USER} ${USER} 

# Create the necessary directorires and set ownership 
RUN mkdir -p /home/${USER}/workspace/repos && \
    install -d -m 0755 -o ${USER} -g ${USER} /home/${USER} && \
    install -d -m 0755 -o ${USER} -g ${USER} /home/${USER}/workspace && \
    install -d -m 0755 -o ${USER} -g ${USER} /home/${USER}/workspace/repos 

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends tzdata sudo && \ 
    echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/$USER && \ 
    chmod 0440 /etc/sudoers.d/$USER

# Switch to the newly created user
USER ${USER}


# Update and install necessary packages for X11 and OpenGL
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libboost-dev \ 
    xorg \
    xvfb \
    libglvnd-dev \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    cmake \
    curl \
    wget \
    zip \ 
    unzip \
    nano \
    git 
    # && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /home/$USER/workspace/repos

# # Set the DISPLAY environment variable for X11 forwarding
# ENV DISPLAY=:0
# Set up the virtual display environment variable
ENV DISPLAY=:99

# Add a startup script to the image

# Setup
ENV DEBIAN_FRONTEND=noninteractive
ENV MINIFORGE_VERSION=Miniforge3-Linux-x86_64.sh
ENV PATH=/home/$USER/miniforge3/bin:$PATH


RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/${MINIFORGE_VERSION} && \
    bash ${MINIFORGE_VERSION} -b -p /home/$USER/miniforge3 && \
    rm ${MINIFORGE_VERSION} && \
    /home/$USER/miniforge3/bin/mamba init

RUN . "/home/$USER/miniforge3/etc/profile.d/conda.sh" && \
    . "/home/$USER/miniforge3/etc/profile.d/mamba.sh" && \
    conda init && mamba init && \
    conda activate
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN conda update -n base -c conda-forge conda

# RUN mkdir -p /workspace && chmod 777 /workspace
COPY environment.yaml /home/$USER/workspace/environment.yaml
COPY requirements.txt /home/$USER/workspace/requirements.txt

RUN awk '/^ *- pip:/ {exit} {print}' /home/$USER/workspace/environment.yaml > /home/$USER/workspace/env_mamba_only.yaml 

RUN FORCE_CUDA=1 mamba env create -f /home/$USER/workspace/env_mamba_only.yaml && \
    echo "source activate $(head -1 /home/$USER/workspace/environment.yaml | cut -d' ' -f2)" >> /home/$USER/.bashrc
# TODO: dynamic environment name
SHELL ["conda", "run", "-n", "work38", "/bin/bash", "-c"]


# RUN mamba activate $(head -1 /home/$USER/workspace/environment.yaml | cut -d' ' -f2) 
# # && \
RUN mamba install mkl=2024.0.0 -y && \
    mamba install pytorch=2.0.0 torchvision=0.15.0 torchaudio=2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
    mamba install pytorch3d -c pytorch3d -y
RUN pip install Cython==3.0.12
RUN pip install -r /home/$USER/workspace/requirements.txt

RUN mkdir -p  /home/$USER/.insightface/models/ && \
    wget -O /home/$USER/.insightface/models/antelopev2.zip "https://keeper.mpdl.mpg.de/f/2d58b7fed5a74cb5be83/?dl=1" && \
    unzip /home/$USER/.insightface/models/antelopev2.zip -d /home/$USER/.insightface/models/antelopev2
RUN   wget -O /home/$USER/.insightface/models/buffalo_l.zip "https://keeper.mpdl.mpg.de/f/8faabd353cfc457fa5c5/?dl=1" && \
    unzip /home/$USER/.insightface/models/buffalo_l.zip -d /home/$USER/.insightface/models/buffalo_l


RUN /home/$USER/miniforge3/bin/conda init bash
RUN /home/$USER/miniforge3/bin/mamba init bash

# RUN git clone https://github.com/radekd91/inferno.git
# WORKDIR /home/$USER/workspace/repos/inferno
# RUN bash pull_submodules.sh
# RUN pip install -e .
# RUN 

# Set the script to run on container start
# CMD ["/usr/local/bin/startup_script.sh"] 

