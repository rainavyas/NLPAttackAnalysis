#!/bin/bash
#$ -S /bin/bash

source ~/rds/hpc-work/Projects/AdvAnalysis/venv-cpu/bin/activate

python ~/rds/hpc-work/Projects/AdvAnalysis/NLPAttackAnalysis/attack_batch.py $@ --batch_ind $SLURM_ARRAY_TASK_ID
