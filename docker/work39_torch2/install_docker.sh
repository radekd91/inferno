#!/bin/bash

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install system dependencies
install_system_dependencies() {
    if ! command_exists unzip || ! command_exists wget; then
        echo "Installing required system dependencies..."
        sudo apt-get update
        sudo apt-get install -y unzip wget
    fi
}

# Function to setup mamba initialization
setup_mamba_init() {
    mamba init bash
    
    # Add mamba initialization to bashrc
    echo '
# >>> mamba initialize >>>
# !! Contents within this block are managed by '"'mamba init'"' !!
export MAMBA_EXE="/home/$USER/miniforge3/bin/mamba";
export MAMBA_ROOT_PREFIX="/home/$USER/miniforge3";
__mamba_setup="$('\''/home/$USER/miniforge3/bin/mamba'\'' shell hook --shell bash --root-prefix '\''/home/$USER/miniforge3'\'' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    if [ -f "/home/$USER/miniforge3/etc/profile.d/mamba.sh" ]; then
        . "/home/$USER/miniforge3/etc/profile.d/mamba.sh"
    fi
fi
unset __mamba_setup
# <<< mamba initialize <<<
' >> ~/.bashrc
    source ~/.bashrc
}
# Install Miniforge if not already installed
if [ ! -d "$HOME/miniforge3" ]; then
    echo "Installing Miniforge..."
    MINIFORGE_VERSION=Miniforge3-Linux-x86_64.sh
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE_VERSION
    bash $MINIFORGE_VERSION -b
    rm $MINIFORGE_VERSION
    export PATH="$HOME/miniforge3/bin:$PATH"
    setup_mamba_init
fi

# Source necessary profile files
source ~/.bashrc
source ~/miniforge3/etc/profile.d/conda.sh
source ~/miniforge3/etc/profile.d/mamba.sh

# Get environment name and create if it doesn't exist
export ENV_NAME=$(head -1 /workspace/environment.yaml | cut -d' ' -f2)
echo "Setting up Python environment: ${ENV_NAME}"

if ! conda env list | grep -q "^${ENV_NAME}"; then
    FORCE_CUDA=1 mamba env create -f /workspace/environment.yaml
fi

# Activate the environment
source "$HOME/miniforge3/bin/activate" "$ENV_NAME"

# Ensure mamba is properly initialized in the active environment
if ! command_exists mamba; then
    setup_mamba_init
fi

# Install Cython if not present
if ! python -c "import Cython" &> /dev/null; then
    echo "Installing Cython..."
    mamba install -y cython=0.29
fi

# Install other requirements
echo "Installing other requirements..."
FORCE_CUDA=1 pip install -r /workspace/requirements.txt

# Download Insightface models
echo "Downloading insightface models..."
mkdir -p ~/.insightface/models/

# Download antelopev2 if not present
if [ ! -d ~/.insightface/models/antelopev2 ]; then
    wget -O ~/.insightface/models/antelopev2.zip "https://keeper.mpdl.mpg.de/f/2d58b7fed5a74cb5be83/?dl=1"
    unzip ~/.insightface/models/antelopev2.zip -d ~/.insightface/models/antelopev2
fi

# Download buffalo_l if not present
if [ ! -d ~/.insightface/models/buffalo_l ]; then
    wget -O ~/.insightface/models/buffalo_l.zip "https://keeper.mpdl.mpg.de/f/8faabd353cfc457fa5c5/?dl=1"
    unzip ~/.insightface/models/buffalo_l.zip -d ~/.insightface/models/buffalo_l
fi

# Install INFERNO if present
cd ~/workspace/repos/inferno 2>/dev/null
if [ $? -eq 0 ] && [ -f "setup.py" ]; then
    if ! python -c "import inferno" &> /dev/null; then
        echo "Installing INFERNO..."
        pip install -e .
    else
        echo "INFERNO is already installed"
    fi
fi

# Print environment information
echo -e "\nSetup Complete!"
echo "Using Python environment: ${ENV_NAME}"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Execute the main command or fall back to bash
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec /bin/bash
fi
