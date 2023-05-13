#!/bin/bash
#/home/rdanecek/anaconda3/condabin/conda activate work36
#which python
source /home/rdanecek/.bashrc
source /home/rdanecek/anaconda3/etc/profile.d/conda.sh
# conda activate work36_cu11
# conda activate work37_cu11_v2
# conda activate work38
conda activate /is/cluster/fast/rdanecek/envs/work38_fast_clone
module load cuda/11.4
cd /home/rdanecek/workspace/repos/gdl/gdl_apps/TalkingHead/evaluation
/is/cluster/fast/rdanecek/envs/work38_fast_clone/bin/python eval_talking_head_on_audio.py $@

