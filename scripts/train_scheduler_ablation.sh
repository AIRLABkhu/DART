#!/bin/bash
# Scheduler ablation: replaces the reward-guided ERDAA scheduler with fixed or
# heuristic T schedules (src/train_scheduler.py). Run from the repository root.
# Output dir gets the scheduler name appended:
#   logs/<domain>_<task>/<algo>/<seed>_<tmax>_<ratio>_<img>_<k><scheduler>/

export MUJOCO_GL=egl
export PYTHONPATH=src/env/dmc2gym:src/env/dm_control:$PYTHONPATH

DOMAIN="walker"
TASK="walk"
ACTION_REPEAT=4
K=1
TMAX=100
IMG_SIZE=32
ALGO="sac"
TRAIN_STEPS="100k"
CUDA_DEV=0

seeds=(1111)
ratios=(0.1)
er_scheduler=("constance_scheduler1" "constance_scheduler50" "constance_scheduler100" "linear_up_scheduler" "cosine_scheduler_start100" "randomized_t_scheduler")

for ratio in "${ratios[@]}"; do
  for seed in "${seeds[@]}"; do
    for scheduler in "${er_scheduler[@]}"; do
      CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/train_scheduler.py \
        --domain_name $DOMAIN \
        --task_name $TASK \
        --action_repeat $ACTION_REPEAT \
        --seed $seed \
        --k $K \
        --mixing_ratio $ratio \
        --tmax $TMAX \
        --img_size $IMG_SIZE \
        --algorithm $ALGO \
        --train_steps $TRAIN_STEPS \
        --er_scheduler $scheduler
    done
  done
done
