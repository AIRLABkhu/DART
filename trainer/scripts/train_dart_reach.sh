#!/usr/bin/env bash
# DART on Robot Reach (paper setting): --mixing_ratio 0.7, --action_space xy, batch 32 (+32 augmented), 250k steps.
# The generator server must already be listening on PORT (see ../generator, `python make_image.py --port 5000`).
# Usage, from trainer/:  bash scripts/train_dart_reach.sh [SEED] [PORT] [GPU]
# Paper seeds: 10081 10082 10083.  --tmax 100 and --k 1 are the defaults and are not passed.
set -e
SEED=${1:-10081}
PORT=${2:-5000}
GPU=${3:-0}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

CUDA_VISIBLE_DEVICES=$GPU python src/train.py --algorithm sac --domain_name robot --task_name reach --action_space xy \
    --train_steps 250k --batch_size 32 --aux_lr 1e-3 --seed $SEED --projection_dim 256 \
    --mixing_ratio 0.7 --port $PORT
