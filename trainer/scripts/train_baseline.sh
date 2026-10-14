#!/usr/bin/env bash
# Vanilla SAC baseline (no generator needed): batch 64, 250k steps.
# Uses the three baseline files in baseline/src (sac.py, train.py, arguments.py) layered over src/:
# PYTHONPATH puts baseline/src first, and algorithms/ (no __init__.py) is a namespace package, so
# algorithms.sac resolves to baseline/src/algorithms/sac.py while everything else comes from src/.
# Usage, from trainer/:  bash scripts/train_baseline.sh [reach|pegbox] [SEED] [GPU]
# Paper seeds: 10081 10082 10083.
set -e
TASK=${1:-reach}
SEED=${2:-10081}
GPU=${3:-0}
if [ "$TASK" = "reach" ]; then AS=xy; elif [ "$TASK" = "pegbox" ]; then AS=xyz; else echo "task must be reach or pegbox"; exit 1; fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=baseline/src:src${PYTHONPATH:+:$PYTHONPATH}

CUDA_VISIBLE_DEVICES=$GPU python baseline/src/train.py --algorithm sac --domain_name robot --task_name $TASK --action_space $AS \
    --train_steps 250k --batch_size 64 --aux_lr 1e-3 --seed $SEED --projection_dim 256
