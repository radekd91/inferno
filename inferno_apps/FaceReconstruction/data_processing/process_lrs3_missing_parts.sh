#!/bin/bash
#/home/rdanecek/anaconda3/condabin/conda activate work36
#which python
source /home/rdanecek/.bashrc
source /home/rdanecek/anaconda3/etc/profile.d/conda.sh
# conda activate work36_cu11
# conda activate work37_cu11_v2
conda activate work38
module load cuda/11.4
/home/rdanecek/anaconda3/envs/work37_cu11_v2/bin/python process_lrs3_missing_parts.py $@

