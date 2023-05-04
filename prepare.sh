#!/bin/sh -l
# FILENAME: submit_job.sh

#SBATCH -A standby
#SBATCH --nodes=1 --gpus-per-node=2 --cpus-per-task 8
#SBATCH --time=04:00:00
model="ensemble_resnet18"
search_P="${1:-0}"
merge_K="${2:-0}"


# -d singleton -J resnet18
source ~/.bashrc

python3 create_input_files.py

# python3 validate.py \
#                 --num-classes 1000 --batch-size 256 \
