#!/bin/bash

# Source bash profile and conda/mamba initialization
source ~/.bashrc
source ~/miniforge3/etc/profile.d/conda.sh
source ~/miniforge3/etc/profile.d/mamba.sh

# Add miniforge to PATH
export PATH="$HOME/miniforge3/bin:$PATH"

# Get environment name from environment.yaml
ENV_NAME=$(head -1 /workspace/environment.yaml | cut -d' ' -f2)

# Initialize conda/mamba in shell
conda init bash
mamba init bash

# Ensure the environment exists and activate it
if ! $HOME/miniforge3/bin/mamba env list | grep -q "^${ENV_NAME} "
then
    echo "Environment ${ENV_NAME} not found. Creating..."
    FORCE_CUDA=1 $HOME/miniforge3/bin/mamba env create -f /workspace/environment.yaml
fi

# Activate the environment
echo "Activating environment ${ENV_NAME}..."
source $HOME/miniforge3/bin/activate ${ENV_NAME}

# Change to the inferno directory
cd ~/workspace/repos/inferno

# Check and install inferno if needed
if ! python -c "import inferno" &> /dev/null
then
    echo "Installing INFERNO..."
    $HOME/miniforge3/envs/${ENV_NAME}/bin/pip install -e .
else
    echo "INFERNO is already installed"
fi

# Print welcome message
echo "Welcome to the INFERNO docker container"
echo "Using Python environment: ${ENV_NAME}"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Execute the main command or fall back to bash
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec /bin/bash
fi
