#!/bin/bash
## source this script first from the entry point script you call the docker task with
source ~/.bashrc
mamba activate work38

cd ~/workspace/repos/inferno

# if inferno is not installed, install it
if ! python -c "import inferno" &> /dev/null
then
    echo "Installing INFERNO"
    pip install -e . 
else 
    echo "INFERNO is installed"
fi

# Execute the main command of the container
echo "Welcome to the INFERNO docker container"
exec "$@"
