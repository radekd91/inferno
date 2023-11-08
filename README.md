
<!-- # INFERNO -->
<h1 align="center">🔥🔥🔥 INFERNO: Set the world on fire with FLAME 🔥🔥🔥</h1>

  <p align="center">
    <a href="https://ps.is.tuebingen.mpg.de/person/rdanecek"><strong>Radek Daněček</strong></a>
  </p>


Welcome to INFERNO. INFERNO is a library of tools and applications for deep-learning-based in-the-wild face reconstruction, animation and accompanying tasks. 
It contains many tools, from processing face video datasets, training face reconstruction networks, applying those face reconstruction networks to get 3D faces and then using these 3D faces to do other things (such as speech-driven animation). 

INFERNO makes heavy use of [FLAME](https://flame.is.tue.mpg.de/), [PyTorch](https://pytorch.org/) and [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).

Current infernal projects: 
- [TalkingHead](inferno_apps/TalkingHead) 
  - official release of [EMOTE: Emotional Speech- Driven Animation with Content- Emotion Disentanglement](https://emote.is.tue.mpg.de/index.html)
  - tools to run, train or finetune speech-driven 3D avatars 
- [FaceReconstruction](inferno_apps/EMOCA)
  - contains EMICA - a combination of [DECA](https://deca.is.tue.mpg.de/), [EMOCA](https://emoca.is.tue.mpg.de/), [SPECTRE](https://filby89.github.io/spectre/) and [MICA](https://zielon.github.io/mica/) which produces excellent results
  - tools to run, train or fine-tune state-of-the-art in-the-wild face reconstruction 
- [MotionPrior](inferno_apps/MotionPrior) 
  - contains FLINT - facial motion prior used in [EMOTE](https://emote.is.tue.mpg.de/index.html)
- [EmotionRecognition](inferno_apps/EmotionRecognition)
  - tools to run and train single-image emotion recognition networks 
- [VideoEmotionRecognition](inferno_apps/VideoEmotionRecognition)
  - contains the vide emotion network used to supervise [EMOTE](https://emote.is.tue.mpg.de/index.html)
  - tools to run and train emotion recognition networks on videos
- [EMOCA](inferno_apps/EMOCA) (deprecated)
  - emotion-driven face reconstruction 
  - (deprecated, for a much better version of face reconstruction go to FaceReconstruction [FaceReconstruction](inferno_apps/FaceReconstruction))


## Installation 

### Dependencies

1) Install [conda](https://docs.conda.io/en/latest/miniconda.html) and update its base environment: 
```bash
conda update -n base -c defaults conda
```

<!-- 2) Install [mamba](https://github.com/mamba-org/mamba) -->

<!-- 0) Clone the repo with submodules:  -->
<!-- ``` -->
<!-- git clone --recurse-submodules ... -->
<!-- ``` -->
2) Clone this repo
<!-- 3) Clone this repo -->

### Short version 

1) Run the installation script: 

```bash
bash install_38.sh
```
If this ran without any errors, you now have a functioning conda environment with all the necessary packages to [run the demos](#usage). If you had issues with the installation script, go through the [long version](#long-version) of the installation and see what went wrong. Certain packages (especially for CUDA, PyTorch and PyTorch3D) may cause issues for some users.

### Long version

1) Pull the relevant submodules using: 
```bash
bash pull_submodules.sh
```


2) Set up a conda environment with one of the provided conda files. I recommend using `conda-environment_py38_cu11_ubuntu.yml`.  

You can use [mamba](https://github.com/mamba-org/mamba) to create a conda environment (strongly recommended):

<!-- ```bash
mamba env create python=3.8 --file conda-environment_py38_cu11_ubuntu.yml
``` -->
```bash
mamba env create python=3.8 --file conda-environment_py38_cu11.yml
```

but you can also use plain conda if you want (but it will be slower): 
<!-- ```bash
conda env create python=3.8 --file conda-environment_py38_cu11_ubuntu.yml
``` -->
```bash
conda env create python=3.8 --file conda-environment_py38_cu11.yml
```


In case the specified pytorch version somehow did not install, try again manually: 
```bash
mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Note: If you find the environment has a missing then just `conda/mamba`- or  `pip`- install it and please notify me.

2) Activate the environment: 
```bash 
conda activate work38_cu11
```

3) For some reason cython is glitching in the requirements file so install it separately: 
```bash 
pip install Cython==0.29.14
```

4) Install `inferno` using pip install. I recommend using the `-e` option and I have not tested otherwise. 

```bash
pip install -e .
```

5) Verify that previous step correctly installed Pytorch3D

For some people the compilation fails during requirements install and works after. Try running the following separately: 

```bash
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
```

Pytorch3D installation (which is part of the requirements file) can unfortunately be tricky and machine specific. EMOCA was developed with is Pytorch3D 0.6.2 and the previous command includes its installation from source (to ensure its compatibility with pytorch and CUDA). If it fails to compile, you can try to find another way to install Pytorch3D.

Notes: 
1) INFERNO was developed with Pytorch 1.12.1 and Pytorch3d 0.6.2 running on CUDA toolkit 11.3. If for some reason installation of these failed on your machine (which can happen), feel free to install these dependencies another way. The most important thing is that version of Pytorch and Pytorch3D match. The version of CUDA is probably less important.
2) Some people experience import issues with opencv-python from either pip or conda. If the OpenCV version installed by the automated script does not work for you (i.e. it does not import without errors), try updating with `pip install -U opencv-python` or installing it through other means. 
The install script installs `opencv-python~=4.5.1.48` installed via `pip`.



## Structure 
This repo has two subpackages. `inferno` and `inferno_apps` 

### INFERNO
`inferno` is a library full of research code. Some things are OK organized, some things less so. It includes but is not limited to the following: 

- `models` is a module with (larger) deep learning modules (pytorch based) 
- `layers` contains individual deep learning layers 
- `datasets` contains base classes and their implementations for various datasets I had to use at some points. It's mostly image-based datasets with various forms of GT if any
- `utils` - various tools

The repo is heavily based on PyTorch, Pytorch Lightning, makes us of Hydra for configuration and 

### INFERNO_APPS 
`inferno_apps` contains prototypes that use the INFERNO library. These can include scripts on how to train, evaluate, test and analyze models from `inferno` and/or data for various tasks. 

Look for individual READMEs in each sub-projects in [`inferno_apps`](./inferno_apps/)

## Usage 

0) Activate the environment: 
```bash
conda activate work38_cu11
```
1) Go the demo folder of one of the projects above and follow the instructions

## Contribute
Contributions to INFERNO are very welcome. Here are two ways to contribute.
#### Projects building on top of INFERNO: 
  - Create a submodule repo in apps and use INFERNO tools to build something cool. I will be happy to promote and/or merge your project if you do so. 

#### Improving INFERNO 
  - INFERNO can do many things, but there is many more it cannot do or it should do better. If you implement a new feature (such as a dataset, add an architecture etc.) or upgrade an existing feature, you are most welcome to create a PR. We will merge it.

  - If you want to build your own tools with INFERNO, refer to [this](./inferno_apps/sandbox_apps/README.md) and [this](./inferno/sandboxes/README.md)


## License
This code and model are **available for non-commercial scientific research purposes** as defined in the [LICENSE](https://emote.is.tue.mpg.de/license.html) file. By downloading and using the code and model you agree to the terms of this license. 

## Acknowledgements 
There are many people who deserve to get credited. These include but are not limited to: 
- [Timo Bolkart](https://sites.google.com/site/bolkartt/) and [Michael Black]() for valuable research guidance
- [Yao Feng](https://is.mpg.de/~yfeng) and [Haiwen Feng](https://is.mpg.de/~hfeng) and their original implementation of [DECA](https://github.com/YadiraF/DECA).
- [Wojciech Zielonka](https://zielon.github.io/) for the original implementation of [MICA](https://zielon.github.io/mica/) 
- [Evelyn Fan](https://github.com/EvelynFan) for the original implementation of [FaceFormer] (https://github.com/EvelynFan/FaceFormer)
- [Panagiotis P. Filntisis](https://filby89.github.io/) and [George Retsinas](https://georgeretsi.github.io/) for the original implementation of [SPECTRE](https://github.com/filby89/spectre)
- Antoine Toisoul and colleagues for [EmoNet](https://github.com/face-analysis/emonet).
