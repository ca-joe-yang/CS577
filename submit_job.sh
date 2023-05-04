#!/bin/sh -l
# FILENAME: submit_job.sh

#SBATCH -A standby
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task 1
#SBATCH --time=04:00:00
model="ensemble_resnet18"
search_P="${1:-0}"
merge_K="${2:-0}"


# -d singleton -J resnet18
source ~/.bashrc

python3 train.py

# python caption.py \
#     --img='datasets/COCO_val2014_000000000208.jpg' \
#     --model='checkpoints/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' \
#     --word_map='datasets/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' \
#     --beam_size=5 \
#     $@
# python3 validate.py \
#                 --num-classes 1000 --batch-size 256 \
#                 --model resnet18 --split valid \
#                 --search-P 1
# ensemble_resnet18
# resnet18                  
# efficientnet_b0 512
# mobilenetv3_small_050
# mobilenetv2_050
# mobilevitv2_050 512
# vit_tiny_patch16_224
# swin_tiny_patch4_window7_224
# swinv2_tiny_window8_256
# mvitv2_tiny
# deit3_small_patch16_224_in21ft1k 256 time 5hrs