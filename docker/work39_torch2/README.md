# Docker 

## WARNING 
The new Python 3.9 environment with Pytorch 2 and CUDA 12 is experimental and may cause issues

**UPDATE (2 August 2026):** The known jittery EMOTE/FLINT inference result in
the PyTorch 2.1 environment has been resolved with a model-scoped legacy
`TransformerEncoderLayer` compatibility path. Archived EMOTE/FLINT configs
enable it automatically in memory; future models can explicitly set
`force_legacy_transformer_encoder_path: false` to retain standard PyTorch
behavior.

## Building 
0) Install docker from the [official website](https://www.docker.com/)

1) Clone the codebase with submodules as described in [root](../README.md#installation)  

2) Run 
```bash
./build_docker.sh
```

## Running 
Run the docker with:
```bash
./run_docker.sh
```
This opens the bash within the container with this codebase mounted. 
Then you can run inferno programs. For instance, running the EMOTE demo: 

```bash 
Xvfb :99 -screen 0 1024x768x16 &
cd /workspace/repos/inferno
mamba activate work38 
pip install -e .
cd inferno_apps/TalkingHead 
python demos/demo_eval_talking_head_on_audio.py
```
