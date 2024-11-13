FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV DISPLAY=:99
# Set CUDA architecture flags
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

# Create user arguments
ARG USER
ARG USER_ID
ARG GROUP_ID

# Install essential packages including gcc-11
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    cmake \
    curl \
    git \
    gcc-11 \
    g++-11 \
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
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxrender1 \
    libfontconfig1 \
    libgoogle-glog-dev \
    libboost-all-dev \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    libceres-dev \
    pkg-config \
    python3-dev \
    python3-pip \
    ninja-build \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set gcc-11 as default compiler
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-11 \
    --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 \
    --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11

# Setup cpp-11 separately
RUN update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-11 110

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

# Run installation and setup environment
SHELL ["/bin/bash", "-c"]

# First install script and conda setup
RUN bash /usr/local/bin/install_script.sh && \
    /home/${USER}/miniforge3/bin/conda init bash && \
    echo '. /home/${USER}/miniforge3/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate' >> ~/.bashrc

# Install PyTorch and dependencies first
RUN /bin/bash -c "source /home/${USER}/miniforge3/etc/profile.d/conda.sh && \
    source /home/${USER}/miniforge3/bin/activate && \
    mamba install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia && \
    pip install fvcore iopath"

# Now install PyTorch3D with CUDA architecture flags
RUN /bin/bash -c "source /home/${USER}/miniforge3/etc/profile.d/conda.sh && \
    source /home/${USER}/miniforge3/bin/activate && \
    TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9;9.0' \
    MAX_JOBS=$(nproc) \
    FORCE_CUDA=1 \
    CC=gcc-11 \
    CXX=g++-11 \
    pip install --no-cache-dir 'pytorch3d@git+https://github.com/facebookresearch/pytorch3d.git'"

# Set environment variables for gcc-11
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11

# Set default shell to bash
SHELL ["/bin/bash", "-l", "-c"]

# Set default command
CMD ["/bin/bash", "-l"]
