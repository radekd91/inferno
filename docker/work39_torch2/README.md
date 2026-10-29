# Docker 

## WARNING 
The new Python 3.9 environment with Pytorch 2 and CUDA 12 is experimental and may cause issues

**UPDATE (2 August 2026):** The known jittery EMOTE/FLINT inference result in
the PyTorch 2.1 environment has been resolved with a model-scoped legacy
`TransformerEncoderLayer` compatibility path. EMOTE (`BertPriorDecoder`) and
FLINT (`L2lDecoder`) select it automatically when their own `inout.time` is
before `2023_12_16_09-39-43`. Models trained at or after that cutoff use standard
PyTorch. An optional decoder flag can override this choice; no flag is inserted
into configs and other decoders are unchanged.
See the [compatibility policy](../../inferno_apps/TalkingHead/demos/emote_backend_diagnostics/README.md).

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
