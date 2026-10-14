#!/bin/bash

save_snapshot=False
use_wandb=False
stage1_use_pretrain=False
num_train_frames=201000 
stage2_n_update=0
save_models=True
# must be an ABSOLUTE path (Hydra chdirs into the run dir)
local_data_dir=/ABSOLUTE/PATH/TO/SAM-G/src/vrl3data

port=5000          # port of the running generator server (make_image_tinysd.py)

for task in 'door' 'hammer'; do
  case $task in door) mixing=0.7 ;; hammer) mixing=0.8 ;; esac   # lambda per task (paper Table I)
  for mixing in $mixing; do
    for seed in 16051 16052 16053; do
      MUJOCO_GL=egl MKL_SERVICE_FORCE_INTEL=1 CUDA_VISIBLE_DEVICES=0 python train_adroit_dhm.py \
        task=${task} \
        seed=${seed} \
        stage1_use_pretrain=${stage1_use_pretrain} \
        save_snapshot=${save_snapshot} \
        device=cuda:0 \
        local_data_dir=${local_data_dir} \
        use_wandb=${use_wandb} \
        stage2_n_update=${stage2_n_update} \
        num_train_frames=${num_train_frames} \
        save_models=${save_models} \
        wandb_group=${1:-dart} \
        port=${port} \
        mixing=${mixing} \
        hydra.run.dir=logs/pieg_${task}_${seed}_${mixing}_10000
    done
  done
done