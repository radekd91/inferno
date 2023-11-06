#!/bin/bash
#/home/rdanecek/anaconda3/condabin/conda activate work36
#which python
source /home/rdanecek/.bashrc
source /home/rdanecek/anaconda3/etc/profile.d/conda.sh
conda activate work38_fast_clone
module load cuda/11.4
/is/cluster/fast/rdanecek/envs/work38_fast_clone/bin/python process_celebvtext.py $@

