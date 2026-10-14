#!/bin/bash
# DART (reward-guided truncation, ERDAA scheduler) on DMControl-GB.
# Run from the repository root:  bash scripts/train_dart.sh
# Per-environment settings used in the paper (100K steps, SAC backbone, T_max=100):
#   walker/walk        ACTION_REPEAT=4  MIXING_RATIO=0.1
#   walker/stand       ACTION_REPEAT=4  MIXING_RATIO=0.5
#   ball_in_cup/catch  ACTION_REPEAT=4  MIXING_RATIO=0.5
#   cartpole/swingup   ACTION_REPEAT=8  MIXING_RATIO=0.2
#   finger/spin        ACTION_REPEAT=2  MIXING_RATIO=0.2
# Note: --mixing_ratio is the weight of the ORIGINAL observation in the blend.

export MUJOCO_GL=egl
export PYTHONPATH=src/env/dmc2gym:src/env/dm_control:$PYTHONPATH

DOMAIN="walker"
TASK="walk"
ACTION_REPEAT=4
K=1
TMAX=100
IMG_SIZE=32
ALGO="sac"          # sac (DART-SAC) or svea_c (DART-SVEA)
TRAIN_STEPS="100k"
CUDA_DEV=0

seeds=(1111 2222 7824)
ratios=(0.1)

for ratio in "${ratios[@]}"; do
  for seed in "${seeds[@]}"; do
    CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/train.py \
      --domain_name $DOMAIN \
      --task_name $TASK \
      --action_repeat $ACTION_REPEAT \
      --seed $seed \
      --k $K \
      --mixing_ratio $ratio \
      --tmax $TMAX \
      --img_size $IMG_SIZE \
      --algorithm $ALGO \
      --train_steps $TRAIN_STEPS
  done
done
