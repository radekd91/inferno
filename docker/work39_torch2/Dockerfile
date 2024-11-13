# Start from an NVIDIA CUDA base image
# FROM nvidia/cuda:11.0-base-ubuntu20.04
# FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04
LABEL maintainer="RadekDanecek"

# Set a non-interactive shell to avoid prompts during the build process
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ Europe/Berlin 

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

# Update and install necessary packages for X11 and OpenGL
USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
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
    git \
    && rm -rf /var/lib/apt/lists/*
USER ${USER}

# Set the working directory in the container
WORKDIR /home/$USER/workspace/repos

# # Set the DISPLAY environment variable for X11 forwarding
# ENV DISPLAY=:0
# Set up the virtual display environment variable
ENV DISPLAY=:99

# Copy files
USER root
COPY install_39_docker.sh /usr/local/bin/install_script.sh
COPY startup_script.sh /usr/local/bin/startup_script.sh
COPY environment.yaml /workspace/environment.yaml
COPY requirements.txt /workspace/requirements.txt

# Set permissions
RUN chmod +x /usr/local/bin/install_script.sh

# Switch back to user
USER ${USER}

# Run installation
RUN bash /usr/local/bin/install_script.sh
RUN /home/$USER/miniforge3/bin/conda init bash
RUN /home/$USER/miniforge3/bin/mamba init bash

# # Set the script to run on container start
# CMD ["/usr/local/bin/startup_script.sh"] 

