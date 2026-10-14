# DART, Robotic Manipulation branch (Reach / Peg-in-box)

Code for the Robotic Manipulation experiments of
*Diffusion-based Augmentation with Reward-guided Truncation for Visuomotor Policy Generalization* (DART).

The trainer is the SGQN codebase (SAC, xArm robot, MuJoCo) with DART added to the SAC update.
Two tasks are used: `reach` (action space `xy`) and `pegbox` (action space `xyz`).
Each task has a training scene plus five held-out test scenes (`Test 1..5`) that change textures and
colors of the scene; they are defined by the MuJoCo XML files under `trainer/src/env/robot/assets/robot/`
(`reach_test{1..5}.xml`, `peg_in_box_test{1..5}.xml`, `shared_test{1..5}.xml`).

## How DART is wired in

DART runs as two processes:

1. **Generator server** (`generator/make_image.py`): a Flask HTTP server that holds InstaFlow
   (1-step rectified flow, `XCLIU/instaflow_0_9B_from_sd_1_5`) with the TAESD tiny VAE
   (`sayakpaul/taesd-diffusers`), working at 32x32. For each request it encodes the input images,
   adds noise up to timestep `T` (`transform_strength`), and regenerates them in one step.
2. **Trainer** (`trainer/src/train.py`): SAC. In every update (`src/algorithms/sac.py`, `SAC.update`)
   the sampled batch of 32 observations (3 stacked 84x84 frames) is downsampled to 32x32, sent to the
   generator, upsampled back to 84x84 and blended with the clean observation:

   `aug_obs = mixing_ratio * obs + (1 - mixing_ratio) * generated`

   The clean and augmented batches are concatenated, so the critic/actor see 2 x 32 = 64 samples per
   update. The vanilla baseline uses a plain batch of 64 to match this.

   `T` is set once per training episode by the reward-guided scheduler `ERDAA()` in `train.py`:
   `T = 0` during the first `init_steps` (1000) steps, afterwards `T` is moved by
   `fluctuation * k * tmax/1000 * tanh(last_reward_delta / (fluctuation * k))` and clipped to
   `[1, tmax]`, with `fluctuation = std(diff(episode_rewards))`. Defaults: `--tmax 100`, `--k 1`.
   The current `T` is logged as `train/ES_Step`.

## Environments

Two separate Python environments are needed (the generator's torch/diffusers stack is not compatible
with gym 0.21 / mujoco-py).

**Trainer env** (`trainer/requirements.txt`): Python 3.8, `torch==1.12.1`, `gym==0.21.0`,
`mujoco-py==2.1.2.14` with MuJoCo 2.1.0 (`mujoco210`) unpacked in `~/.mujoco`, plus
`numpy==1.24.4 opencv-python==4.5.3.56 imageio imageio-ffmpeg termcolor requests matplotlib
kornia==0.5.11 captum==0.7.0 tensorboard`. `trainer/install.sh` shows the mujoco210 download.
gym must stay at 0.21: later gym versions drop `gym.envs.robotics`, which the robot environment
inherits from. `kornia` and `captum` are imported at module load (augmentations / saliency utils) even
though SAC does not use them. `trainer/setup/` is kept from the upstream SGQN release for reference
(its `conda.yaml` lists the original torch 1.7 stack; the versions above are the ones used here).

**Generator env** (`generator/requirements.txt`): Python 3.8, `torch==2.0.1`, `diffusers==0.24.0`,
`transformers==4.35.2`, `Flask==3.0.3`, `huggingface-hub==0.25.1`, `accelerate`, `safetensors`.
Model weights are downloaded from the Hugging Face hub on first start.

Before running the trainer in every shell:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

`train.py` sets `MUJOCO_GL=egl` and `MJKEY_PATH` itself (mujoco210 does not need a key, the variable
is just required by mujoco-py).

## Step 1: start the generator

From `generator/` (the pipeline is imported relatively), in the generator env:

```bash
cd generator
CUDA_VISIBLE_DEVICES=0 python make_image.py --port 5000
```

One server per GPU. The paper runs used port 5000 for reach and 5500 for pegbox; any port works as
long as the trainer's `--port` matches.

## Step 2: train DART

From `trainer/`, in the trainer env. Paper settings (250k steps, batch 32, seeds 10081/10082/10083):

```bash
cd trainer
# Reach: mixing_ratio 0.7, action space xy
bash scripts/train_dart_reach.sh 10081 5000 0        # [SEED] [PORT] [GPU]
# Peg-in-box: mixing_ratio 0.4, action space xyz
bash scripts/train_dart_pegbox.sh 10081 5500 0
```

which expand to

```bash
python src/train.py --algorithm sac --domain_name robot --task_name reach  --action_space xy  --train_steps 250k --batch_size 32 --aux_lr 1e-3 --seed 10081 --projection_dim 256 --mixing_ratio 0.7 --port 5000
python src/train.py --algorithm sac --domain_name robot --task_name pegbox --action_space xyz --train_steps 250k --batch_size 32 --aux_lr 1e-3 --seed 10081 --projection_dim 256 --mixing_ratio 0.4 --port 5500
```

## Step 3: train the vanilla SAC baseline

No generator needed, batch size **64**:

```bash
cd trainer
bash scripts/train_baseline.sh reach  10081 0        # [reach|pegbox] [SEED] [GPU]
bash scripts/train_baseline.sh pegbox 10081 0
```

The DART trainer cannot run the plain baseline by itself: `src/train.py` always calls the
reward-guided scheduler and `src/algorithms/sac.py` always queries the generator once `T > 0`, and
there is no switch to turn this off. The baseline is therefore the upstream SGQN SAC code, which
differs from the DART trainer in exactly three files, shipped under `trainer/baseline/src/`:

- `algorithms/sac.py`: plain `update()` (no generator call, no blending, no batch concatenation)
- `train.py`: no `ERDAA` scheduler, run tag without `k/mr/tmax`, output dir without the `esdda` suffix
- `arguments.py`: defaults `--train_steps 250k --batch_size 64`, no DART arguments
  (`--port --mixing_ratio --tmax --k ...`)

(`augmentations.py` of the DART tree only adds one unused helper function and `utils.py` differs in
whitespace, so they are shared.) `scripts/train_baseline.sh` runs `baseline/src/train.py` with
`PYTHONPATH=baseline/src:src`; because `algorithms/` has no `__init__.py` it is a namespace package,
so `algorithms.sac` is taken from `baseline/src` and everything else from `src/`.

## Where outputs go

Everything is written below `trainer/logs/`:

```
logs/<domain>_<task>/sac/<runtag>/esdda/        # DART
logs/<domain>_<task>/sac/<runtag>/              # baseline
    train.log     # JSONL, one record per episode (train/episode_reward, train/ES_Step = current T, ...)
    eval.log      # JSONL, one record per evaluation: episode_reward (train scene),
                  #        episode_reward_test_env_1 ... episode_reward_test_env_5
    info.log      # all arguments of the run
    model/        # actor_<step>.pt, critic_<step>.pt every 50k steps
    video/        # only with --save_video
```

`<runtag>` encodes the hyperparameters with `.` replaced by `p`, e.g.

```
seed10081_ts250000_bs32_proj256_auxlr0p001_asxy_fs3_k1_mr0p7_tmax100     # DART reach
seed10081_ts250000_bs32_proj256_auxlr0p001_asxyz_fs3_k1_mr0p4_tmax100    # DART pegbox
seed10081_ts250000_bs64_proj256_auxlr0p001_asxy_fs3                      # baseline reach
```

## Evaluation

There is no separate evaluation script. Every `--eval_freq` (50k) steps, including step 0 and the
final step 250000, the agent is evaluated for `--eval_episodes` (30) episodes on the training scene
and on each of the five test scenes (`RobotReach_test_{1..5}-v0`, `RobotPegbox_test_{1..5}-v0`), and
the mean returns are appended to `eval.log`. Episodes are 50 steps long.

## Building the table

`trainer/test_robot.py` reads the `eval.log` of the three paper seeds, keeps the step-250000 record,
and writes mean / std over seeds per test scene (`int(mean)` and `int(std)` in the `bound` column):

```bash
cd trainer
python test_robot.py --environment robot_reach                       # -> eval_robot_reach.csv
python test_robot.py --environment robot_pegbox                      # -> eval_robot_pegbox.csv
python test_robot.py --environment robot_reach  --method baseline    # -> eval_robot_reach_baseline.csv
python test_robot.py --environment robot_pegbox --method baseline
```

Seeds default to `10081,10082,10083` (`--seeds` to change). The script raises if a log has no
step-100000 record, i.e. if a run did not get far enough.

## Gotchas

- The generator port and the trainer's `--port` must match; one generator per GPU. If the generator
  is not up, the first update after `init_steps` fails with a connection error.
- DART uses `--batch_size 32` (plus 32 augmented copies), the vanilla baseline `--batch_size 64`.
  Do not swap them.
- `--mixing_ratio` is task specific: 0.7 for reach, 0.4 for pegbox.
- Re-running with the same arguments reuses the same run directory and stops immediately with
  `specified working directory already exists` (assert on an existing `train.log`). Delete or move the
  old run directory first.
- `test_robot.py` needs `pandas`, which the training code itself does not.
- Run the generator from `generator/` and the trainer and `test_robot.py` from `trainer/`; all paths are
  relative to those directories.

## License

The trainer keeps the MIT license of the SGQN codebase it is built on (`trainer/LICENSE`).
`generator/pipeline_rf.py` is the InstaFlow rectified-flow pipeline, derived from the HuggingFace
`diffusers` Stable Diffusion pipeline (Apache License 2.0, header kept in the file).
