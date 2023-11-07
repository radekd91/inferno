#!/bin/bash
#/home/rdanecek/anaconda3/condabin/conda activate work36
#which python
source /home/rdanecek/.bashrc
source /home/rdanecek/anaconda3/etc/profile.d/conda.sh
# conda activate work36_cu11
# conda activate work37_cu11_v2
# conda activate work38
conda activate work38_clone
module load cuda/11.4
# /home/rdanecek/anaconda3/envs/work36_cu11/bin/python process_lrs3.py $@
/home/rdanecek/anaconda3/envs/work38_clone/bin/python process_affectnet_mediapipe.py $@

