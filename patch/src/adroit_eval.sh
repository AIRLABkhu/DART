task_name='hammer'
save_snapshot=False
use_wandb=False
stage1_use_pretrain=False
num_train_frames=201000
stage2_n_update=0
save_models=False
# absolute path of a DART run dir containing snapshot200000.pt
model_dir=/ABSOLUTE/PATH/TO/SAM-G/src/logs/pieg_hammer_16052_0.6_10000
# $1 = test_pieg_<level>, level in color-easy color-hard video-easy video-hard (see robust_eval_dart.sh)
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python eval_adroit_pieg.py \
		task=${task_name} \
		seed=16052 \
		stage1_use_pretrain=${stage1_use_pretrain} \
		save_snapshot=${save_snapshot} \
		device=cuda:0 \
		use_wandb=${use_wandb} \
		stage2_n_update=${stage2_n_update} \
		num_train_frames=${num_train_frames} \
		save_models=${save_models} \
		model_dir=${model_dir} \
		wandb_group=$1