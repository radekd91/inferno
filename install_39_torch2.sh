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
mamba create -n work39_torch2 python=3.9 
eval "$(conda shell.bash hook)" # make sure conda works in the shell script
conda activate work39_torch2
if echo $CONDA_PREFIX | grep work39_torch2
then
    echo "Conda environment successfully activated"
else
    echo "Conda environment not activated. Probably it was not created successfully for some reason. Please activate the conda environment before running this script"
    exit
fi
echo "Installing conda packages"
# mamba env update -n work39_torch2 --file conda-environment_py39_cu12_torch2_ubuntu.yml 
FORCE_CUDA=1 mamba env update -n work39_torch2 --file conda-environment_py39_cu12_torch2.yaml 
echo "Installing other requirements"
FORCE_CUDA=1 pip install -r requirements39_torch2.txt
# pip install Cython==0.29
# echo "Making sure Pytorch3D installed correctly"
# pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
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

