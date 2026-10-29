#!/bin/bash
CONDA_DIR="$HOME/miniforge3"
ENV_FILE="$HOME/workspace/environment.yaml"
REQ_FILE="$HOME/workspace/requirements.txt"

# Install Miniforge if not already installed
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniforge..."
    MINIFORGE_VERSION=Miniforge3-Linux-x86_64.sh
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE_VERSION"
    bash "$MINIFORGE_VERSION" -b -p "$CONDA_DIR"
    rm "$MINIFORGE_VERSION"
fi

export PATH="$CONDA_DIR/bin:$PATH"
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Activate the environment and install the codebase if the environment.yml file exists
# if [ -f "~/workspace/environment.yaml" ]; then
echo "Creating and activating the Python environment..."
# mamba env update -n $ENV_NAME --file ~/workspace/environment.yaml
FORCE_CUDA=1 mamba env create -f "$ENV_FILE"
export ENV_NAME=$(head -1 "$ENV_FILE" | cut -d' ' -f2)
conda activate "$ENV_NAME"

## Hack should be no longer necessary
# ## Cython hack (for some reason insightface requires Cython==0.29 but doesn't list it as a requirement)
# pip install Cython==0.29

# Install other requirements
echo "Installing other requirements..."
conda activate "$ENV_NAME"
# mamba env update -n $ENV_NAME --file "$ENV_FILE"
FORCE_CUDA=1 pip install -r "$REQ_FILE"

# Insightface has problems with downloading some of their models
echo -e "\nDownloading insightface models..."
mkdir -p ~/.insightface/models/
if [ ! -d ~/.insightface/models/antelopev2 ]; then
  wget -O ~/.insightface/models/antelopev2.zip "https://keeper.mpdl.mpg.de/f/2d58b7fed5a74cb5be83/?dl=1"
  unzip ~/.insightface/models/antelopev2.zip -d ~/.insightface/models/antelopev2
fi
if [ ! -d ~/.insightface/models/buffalo_l ]; then
  wget -O ~/.insightface/models/buffalo_l.zip "https://keeper.mpdl.mpg.de/f/8faabd353cfc457fa5c5/?dl=1"
  unzip ~/.insightface/models/buffalo_l.zip -d ~/.insightface/models/buffalo_l
fi


# # Install the codebase in editable mode if it exists
# # if [ -d "~/workspace/repos/inferno" ]; then
# echo "Installing the Inferno codebase..."
# cd ~~/workspace/repos/inferno
# pip install -e . 
# cd ../..
# # fi
# # fi

echo "Installation finished"

