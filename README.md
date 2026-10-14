# DART on DMControl-GB

This folder is the DMControl Generalization Benchmark branch of **DART** (Diffusion-based Augmentation with Reward-guided Truncation for Visuomotor Policy Generalization, RA-L). It trains a pixel-based SAC (or SVEA) agent whose replay observations are augmented in-process by a one-step rectified-flow model (InstaFlow) with a tiny VAE (TAESD), and where the diffusion truncation step `T` is driven by the training-reward signal (the ERDAA scheduler in `src/train.py`).

The code descends from the DMControl-GB / SVEA / DGA2 codebases; `dm_control` and `dmc2gym` are bundled under `src/env/` with their original licenses.

## What is here

```
src/train.py              DART: ERDAA reward-guided T scheduler (the paper method)
src/train_scheduler.py    ablation: fixed / heuristic T schedulers (--er_scheduler)
src/algorithms/sac.py     SAC + in-process InstaFlow/TAESD augmentation and blending
src/algorithms/svea_c.py  SVEA variant (random_conv consistency on the critic)
src/algorithms/drq.py     legacy DrQ variant (kept for the factory; not used in the paper)
src/env/                  dmc2gym, dm_control (MuJoCo 2.0), distracting_control, wrappers,
                          data/ (color_easy/hard .pt, video_easy/hard .mp4 used by evaluation)
setup/config.cfg          dataset root, only needed for places365 overlay / distracting_cs
scripts/                  representative training, ablation and aggregation scripts
main_table.py             3-seed aggregation of eval.log into eval_<name>.csv
```

## Environment setup

* Python 3.8 (tested with 3.8.8), CUDA GPU required.
* `pip install -r requirements.txt` (versions are the ones used for the paper runs).
* MuJoCo 2.0: the bundled `dm_control` is the MuJoCo-2.0 (`mujoco200`) version and expects the binaries at `~/.mujoco/mujoco200_linux/bin/libmujoco200.so` and the key at `~/.mujoco/mjkey.txt` (override with `MJLIB_PATH` / `MJKEY_PATH`).
* The bundled environment packages are picked up through `PYTHONPATH`; alternatively run `bash setup/install_envs.sh` to `pip install -e` them.
* InstaFlow (`XCLIU/instaflow_0_9B_from_sd_1_5`) and TAESD (`sayakpaul/taesd-diffusers`) are downloaded from Hugging Face on first run, so internet access is needed the first time.
* All commands are run from this directory (the code opens `src/env/data/...` and `setup/config.cfg` relative to the cwd).

```bash
export MUJOCO_GL=egl
export PYTHONPATH=src/env/dmc2gym:src/env/dm_control
```

`setup/config.cfg` holds a `datasets` list. It is only read by the places365 `random_overlay` augmentation and by the `distracting_cs` evaluation mode (DAVIS backgrounds). Standard DART / SVEA training and the four standard evaluation modes do not need it, so the placeholder `/your/data/path/here/` can stay.

## Training DART

```bash
MUJOCO_GL=egl PYTHONPATH=src/env/dmc2gym:src/env/dm_control CUDA_VISIBLE_DEVICES=0 \
python src/train.py --domain_name walker --task_name walk --action_repeat 4 \
  --seed 1111 --k 1 --mixing_ratio 0.1 --tmax 100 --img_size 32 \
  --algorithm sac --train_steps 100k
```

`scripts/train_dart.sh` loops this over the three seeds.

Per-environment settings used for the main table (SAC backbone, 100K steps, `--tmax 100`, `--k 1`, `--img_size 32`, seeds 1111 / 2222 / 7824):

| domain / task      | `--action_repeat` | `--mixing_ratio` |
|--------------------|-------------------|------------------|
| walker / walk      | 4                 | 0.1              |
| walker / stand     | 4                 | 0.5              |
| ball_in_cup / catch| 4                 | 0.5              |
| cartpole / swingup | 8                 | 0.2              |
| finger / spin      | 2                 | 0.2              |

Argument meanings:

* `--mixing_ratio` is the weight of the **original** observation in the blend: `aug_obs = obs * mixing_ratio + diffusion_out * (1 - mixing_ratio)`. So `0.1` means 90 percent synthesized content.
* `--tmax` is the upper bound `T_max` of the reward-guided truncation step; `--k` scales the ERDAA update.
* `--img_size 32` is the working resolution of the diffusion model; outputs are resized back to 84x84 bilinearly.
* The augmented half of the batch is concatenated with the clean half (64 + 64) before the critic / actor updates.

### SVEA variant

Same command with `--algorithm svea_c`, for example the 500K walker/walk run:

```bash
python src/train.py --domain_name walker --task_name walk --action_repeat 4 \
  --seed 1111 --k 1 --mixing_ratio 0.5 --tmax 100 --img_size 32 \
  --algorithm svea_c --train_steps 500k
```

### Scheduler ablation

`src/train_scheduler.py` replaces ERDAA with a fixed or heuristic schedule chosen by `--er_scheduler` (one of `constance_scheduler1`, `constance_scheduler50`, `constance_scheduler100`, `linear_up_scheduler`, `cosine_scheduler_start100`, `randomized_t_scheduler`):

```bash
python src/train_scheduler.py --domain_name walker --task_name walk --action_repeat 4 \
  --seed 1111 --k 1 --mixing_ratio 0.1 --tmax 100 --img_size 32 \
  --algorithm sac --train_steps 100k --er_scheduler randomized_t_scheduler
```

`scripts/train_scheduler_ablation.sh` loops over all six schedulers.

## Outputs

Each run writes to

```
logs/<domain>_<task>/<algorithm>/<seed>_<tmax>_<mixing_ratio>_<img_size>_<k>/
```

(`train_scheduler.py` appends the scheduler name to the last component). Inside:

* `train.log`: JSONL, one line per episode; `ES_STEP` is the current diffusion truncation `T`.
* `eval.log`: JSONL, one line per evaluation. Every `--eval_freq` steps (default `100k`) the agent is evaluated for `--eval_episodes` (default 30) episodes in five modes: train, color_easy, color_hard, video_easy, video_hard. For a 100K run this is exactly one record with `step == 100000`; that record is the number reported in the paper.
* `video/`: a GIF every 5th evaluation episode. `model/`: checkpoints (only `train_scheduler.py` saves them, every `--save_freq`).

There is no separate evaluation script; evaluation is inline.

## Aggregating results

`main_table.py` reads `logs/<env>/sac/{1111,2222,7824}_100_<ratio>_32_1.0/eval.log` for the five environments (with the ratios in the table above), keeps the `step == 100000` record, and writes `eval_<name>.csv` with integer mean and std for color_easy, color_hard, video_easy and video_hard:

```bash
python main_table.py --environment all      # or: bash scripts/aggregate.sh
```

It raises `ValueError` if any run has no `step == 100000` record.

## Gotchas

* `train.py` prompts interactively (`working dir already exists, delete it? y or n`) when the run directory exists, and deletes it on `y`. Do not launch it unattended on an existing directory.
* The ERDAA scheduler returns `T = 0` for the first `--init_steps` (1000) steps, so augmentation is off during the warm-up.
* `--mixing_ratio` is the weight of the original observation, not of the synthesized one.
* `PYTHONPATH` must point at the bundled `src/env/dmc2gym` and `src/env/dm_control`; a pip-installed, newer `dm_control` will not match.
* `MUJOCO_GL=egl` needs a working headless EGL (GPU driver). `train.py` also sets `PYOPENGL_PLATFORM=egl`.
* `distracting_cs` and the places365 overlay augmentation are present in the code but are not used by the paper's DART runs; they need the dataset root in `setup/config.cfg`.
