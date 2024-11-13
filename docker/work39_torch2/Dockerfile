FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV DISPLAY=:99

# Create user arguments
ARG USER
ARG USER_ID
ARG GROUP_ID

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    cmake \
    curl \
    git \
    nano \
    tzdata \
    unzip \
    wget \
    xorg \
    xvfb \
    zip \
    libgl1 \
    libgles2 \
    libglvnd-dev \
    libglx0 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Create user with sudo privileges
RUN groupadd -g ${GROUP_ID} ${USER} \
    && useradd -l -u ${USER_ID} -g ${USER} ${USER} \
    && mkdir -p /home/${USER}/workspace/repos \
    && chown -R ${USER}:${USER} /home/${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

# Copy installation files
COPY install_39_docker.sh /usr/local/bin/install_script.sh
COPY environment.yaml /workspace/environment.yaml
COPY requirements.txt /workspace/requirements.txt
RUN chmod +x /usr/local/bin/install_script.sh

# Switch to user and set working directory
USER ${USER}
WORKDIR /home/${USER}/workspace/repos

# Run installation and initialize conda
RUN bash /usr/local/bin/install_script.sh \
    && /home/${USER}/miniforge3/bin/conda init bash
