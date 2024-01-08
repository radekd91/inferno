#!/bin/bash
echo "Pulling submodules"
bash pull_submodules.sh
echo "Installing mamba"
conda install mamba -n base -c conda-forge
if ! command -v mamba &> /dev/null
then
    echo "mamba could not be found. Please install mamba before running this script"
    exit
fi
echo "Creating conda environment"
mamba create -n work38 python=3.8 
eval "$(conda shell.bash hook)" # make sure conda works in the shell script
conda activate work38
if echo $CONDA_PREFIX | grep work38
then
    echo "Conda environment successfully activated"
else
    echo "Conda environment not activated. Probably it was not created successfully for some reason. Please activate the conda environment before running this script"
    exit
fi
echo "Installing conda packages"
# mamba env update -n work38 --file conda-environment_py38_cu11_ubuntu.yml 
mamba env update -n work38 --file conda-environment_py38_cu11.yaml 
echo "Installing other requirements"
pip install -r requirements38.txt
pip install Cython==0.29
echo "Making sure Pytorch3D installed correctly"
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
echo "Installing INFERNO"
pip install -e . 

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

echo "Installation finished"

