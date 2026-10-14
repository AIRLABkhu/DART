#!/bin/bash
# Robust evaluation of one DART run on the 4 Adroit distraction settings
# (100 episodes per level, appended to <model_dir>/file_<seed>.txt).
# Usage: bash robust_eval_dart.sh <GPU> <task> <seed> <abs model_dir>
# The level is passed through wandb_group=test_pieg_<level> (exactly two underscores,
# hyphenated level names); eval_adroit_pieg.py parses it. No generator server is needed.
GPU=$1; TASK=$2; SEED=$3; MODEL_DIR=$4
for lvl in color-easy color-hard video-easy video-hard; do
  echo ">>> EVAL $TASK seed=$SEED $lvl"
  MUJOCO_GL=egl MKL_SERVICE_FORCE_INTEL=1 CUDA_VISIBLE_DEVICES=$GPU python eval_adroit_pieg.py \
    task=$TASK seed=$SEED device=cuda:0 stage1_use_pretrain=False save_snapshot=False \
    use_wandb=False stage2_n_update=0 num_train_frames=201000 save_models=False \
    model_dir=$MODEL_DIR wandb_group=test_pieg_${lvl} || echo "!!! EVAL FAILED $TASK $SEED $lvl"
done
