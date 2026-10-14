# DART on Adroit (patch set for SAM-G)

This directory contains the DART (Diffusion-based Augmentation with Reward-guided Truncation)
code for the Adroit benchmark. It is a **patch on top of the public SAM-G repository**; SAM-G itself
is not re-hosted here. The DART agent is the PIE-G agent of SAM-G whose strong augmentation is
replaced by a diffusion-generated image blended with the original frame, where the diffusion
truncation value T is driven by the reward-fluctuation scheduler `erdaa()`.

Contents

| Path | What it is |
|---|---|
| `patch/` | DART-modified and DART-added files at their SAM-G-relative paths (copy over a SAM-G checkout) |
| `dart_adroit.patch` | The same change as one unified diff against SAM-G commit `1df37a1` (`git apply`) |
| `generator/` | The diffusion image server the trainer talks to over HTTP (`make_image.py` + `pipeline_rf.py`) |

## 1. Set up SAM-G

```bash
git clone https://github.com/wadiuvatzy/SAM-G.git
cd SAM-G
git checkout 1df37a1        # "remove __pycache__ folders"; the patch is made against this commit
```

Follow SAM-G's own `README.md` for the environment (`bash install_adroit.sh`, MuJoCo 2.1, `mujoco-py`,
`DHM`, `mjrl`, `mj_envs`), for the `vrl3data` folder (Adroit demos and pretrained encoders, from the
Google Drive link in SAM-G's README) and for the Places365 dataset. DART itself does not need the
EfficientViT/PerSAM checkpoints (`use_SAM_g` is `false` in the patched `pieg_config.yaml`), but the SAM-G
environment package still imports them, so install SAM-G completely.

## 2. Apply the DART patch

Either

```bash
git apply /path/to/adroit/dart_adroit.patch
```

or copy the files:

```bash
cp -r /path/to/adroit/patch/. .
```

Files touched (all paths relative to the SAM-G root):

| File | Status | DART change |
|---|---|---|
| `src/train_adroit_dhm.py` | modified | adds `erdaa()` (reward-fluctuation scheduler for T), passes T, `mixing` and the server port into `agent.update()`, logs the schedule as `ES_Step`, saves `snapshot<frame>.pt` every 100K steps |
| `src/algos/pieg.py` | modified | DART augmentation: downsamples the batch to 32x32, POSTs it with T to the generator, blends the returned image with the original, then applies PIE-G's random overlay; dumps example images under `PIEG/<mixing>/<step>/` every 10K steps |
| `src/algos/vrl3_agent.py` | modified | `VRL3Agent.__init__` takes a `port` argument (stored as `self.port`) |
| `src/cfgs_adroit/pieg_config.yaml` | modified | new keys `port`, `tmax` (=100), `mixing`, `k`, `domain_ratio`, `saliency_ratio`, `diffusion_step`, `shift_ratio`; `agent.port`; `use_SAM_g: false` |
| `src/cfgs_adroit/config.yaml`, `src/cfgs_adroit/sgqn_config.yaml` | modified | `agent.port: 5000` (required by the new `VRL3Agent` signature) plus unused keys from the same experiments |
| `src/eval_adroit_pieg.py` | added | copy of SAM-G's `eval_adroit.py` that loads `snapshot200000.pt` and uses `pieg_config.yaml`; used for the robust evaluation |
| `src/robust_eval_dart.sh` | added | runs the 4-setting robust evaluation for one run directory |
| `src/train.sh`, `src/adroit_eval.sh` | modified | launcher scripts (paths replaced with placeholders, see below) |
| `DHM/dhm/make_env_RRL_sam.py` | modified | video-background frames are resized to 84x84 (was 160x160) so the `video-*` test settings work at the 84x84 input resolution; DINO patch-token width 768 -> 384 (only used when `use_SAM_g` is true) |

`src/algos/sgqn.py`, `src/algos/svea.py`, `src/rl_utils.py` and `src/eval_adroit.py` are left as in SAM-G.

## 3. Places365 path

The PIE-G backbone (and therefore DART) uses Places365 for the random-overlay augmentation. Edit
`src/cfgs_adroit/aug_config.cfg` so that the `datasets` list holds the directory that contains
`places365_standard/{train,val}` (the loader appends those two sub-folders itself):

```json
{ "datasets": [ "/ABSOLUTE/PATH/TO/places365_root" ] }
```

## 4. Start the generator server

The generator is a small Flask server around InstaFlow-0.9B (`XCLIU/instaflow_0_9B_from_sd_1_5`,
one-step rectified flow, `pipeline_rf.py`) with the TAESD VAE (`sayakpaul/taesd-diffusers`). It is the
same server used for the Robotic Manipulation experiments. Run it in its own Python environment (see
`generator/requirements.txt`; the paper used Python 3.8, torch 2.0.1, diffusers 0.24.0). Both models are
downloaded from the Hugging Face hub on first start.

```bash
cd /path/to/adroit/generator
CUDA_VISIBLE_DEVICES=0 python make_image.py --port 5000
```

Protocol: the trainer POSTs a `(B*3, 3, 32, 32)` float tensor in `[0, 1]` (form file field
`initial_latent`, a `torch.save` buffer) and the scalar form field `transform_strength` (= T from
`erdaa()`). The server encodes the batch with the VAE, noises the latents to timestep T
(`scheduler.add_noise(..., T)`), runs one InstaFlow step starting at timestep T (`timesteps=T`, empty
prompt, `guidance_scale=0`) and returns the decoded images as a `torch.save` buffer of shape
`(B*3, 3, 32, 32)` in `[0, 1]`; the trainer upsamples them to 84x84 before blending. T therefore
controls how much of the original frame survives (small T = mild change, large T = closer to a fresh
sample). Keep the server on the same machine (the trainer calls `http://localhost:<port>/generate`).
Training without a running server fails with a `requests` connection error at the first update.

## 5. Train DART

Environment variables and the exact arguments of the paper runs (from the saved Hydra overrides).
`local_data_dir` **must be an absolute path**: Hydra changes the working directory into the run
directory, so the relative default in `pieg_config.yaml` does not resolve.

```bash
cd SAM-G/src
export MUJOCO_GL=egl MKL_SERVICE_FORCE_INTEL=1
CUDA_VISIBLE_DEVICES=0 python train_adroit_dhm.py \
    task=door seed=16051 \
    stage1_use_pretrain=False save_snapshot=False device=cuda:0 \
    local_data_dir=/ABSOLUTE/PATH/TO/SAM-G/src/vrl3data \
    use_wandb=False stage2_n_update=0 num_train_frames=201000 save_models=True \
    port=5000 mixing=0.7 \
    hydra.run.dir=logs/pieg_door_16051_0.7_10000
```

* `mixing` is lambda. **In this code base `mixing` is the weight of the ORIGINAL frame**:
  `blend = original * mixing + diffused * (1 - mixing)` (`src/algos/pieg.py`, `PIEGAgent.update`).
  `mixing=0.7` therefore keeps 70% of the original frame and injects 30% of the diffused frame; the
  random overlay is applied afterwards.
* Per-task lambda used for the paper (Table I): `door` = 0.7, `hammer` = 0.8, seeds 16051/16052/16053,
  `T_max` = 100. The shared single value reported for Adroit is 0.8.
* `T_max` is `tmax` in `pieg_config.yaml` (default 100, not overridden in the paper runs); `T` starts
  at 1, is updated by `erdaa()` after every training episode, clipped to `[1, tmax]`, and is applied by
  the generator server as the noising timestep (step 4).
* `num_train_frames=201000` (200K frames, 100K agent steps with `action_repeat=2`); snapshots are
  written every 100K steps, so the value must exceed 200000 for `snapshot200000.pt` to exist.
* Seeds used in the paper: 16051, 16052, 16053. `src/train.sh` loops over tasks, lambdas and seeds
  with these arguments (edit the placeholders at its top).
* Hydra writes the run into `hydra.run.dir`; the paper runs were named
  `logs/pieg_<task>_<seed>_<mixing>_<fluctmult>`.

### The fluctuation multiplier (`*10000`)

`erdaa()` in `src/train_adroit_dhm.py` (lines 51-72; the multiplier is at line 64) scales the reward-fluctuation term with a
constant that is **not a command-line argument**; it is selected by editing the source:

```python
first = fluctuation_strength*10000      # paper runs (run-dir suffix `_10000`)   <- active in this patch
# first = fluctuation_strength*1000     # ablation (no suffix)
# first = fluctuation_strength*100      # ablation (`_100`)
```

The patch ships with the `*10000` line active, which is the variant used for the paper's Adroit
results. If you reproduce the multiplier ablation, comment/uncomment these lines and name the run
directory accordingly.

## 6. Robust evaluation (4 settings, 100 episodes each)

No generator server is needed for evaluation. The distraction level is passed through
`wandb_group=test_pieg_<level>` (exactly two underscores; levels are `color-easy`, `color-hard`,
`video-easy`, `video-hard`, hyphenated). `src/robust_eval_dart.sh` runs all four for one run:

```bash
cd SAM-G/src
bash robust_eval_dart.sh 0 door 16051 /ABSOLUTE/PATH/TO/SAM-G/src/logs/pieg_door_16051_0.7_10000
```

which expands to, per level,

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python eval_adroit_pieg.py task=door seed=16051 device=cuda:0 \
    stage1_use_pretrain=False save_snapshot=False use_wandb=False stage2_n_update=0 \
    num_train_frames=201000 save_models=False \
    model_dir=/ABSOLUTE/PATH/TO/SAM-G/src/logs/pieg_door_16051_0.7_10000 wandb_group=test_pieg_color-easy
```

`eval_adroit_pieg.py` loads `<model_dir>/snapshot200000.pt`. An episode counts as a success when
`n_goal_achieved_total > 25`; `success_rate = successes / 100`.

## 7. Where results are written

* Training run directory (`hydra.run.dir`): `train.csv`, `eval.csv` (periodic in-domain evaluation,
  15 episodes), `snapshot<frame>.pt` (every 100K steps), `buffer/`, `.hydra/` (config and overrides),
  `train_adroit_dhm.log`, and `PIEG/<mixing>/<step>/` example augmented frames.
* Robust evaluation: one line `success_rate: <x>` is **appended** to `<model_dir>/file_<seed>.txt`
  per level, without a level label. Always run the four levels in the order
  color-easy, color-hard, video-easy, video-hard and read the last four lines in that order; a partial
  re-run silently shifts the mapping.

## Gotchas

* `mixing` weights the original frame here (`original*mixing + diffused*(1-mixing)`), see above.
* `local_data_dir` must be absolute; `aug_config.cfg` must point at the Places365 root.
* The `*10000` fluctuation line must be the active one for the paper configuration.
* `export MUJOCO_GL=egl MKL_SERVICE_FORCE_INTEL=1` (the trainer also sets them internally) and the
  usual MuJoCo 2.1 `LD_LIBRARY_PATH` (`~/.mujoco/mujoco210/bin`, `/usr/lib/nvidia`).
* `port` is both a top-level and an `agent.port` key; the trainer uses the top-level `port=` override.
* The environment writes a temporary MJCF `DAPG_<task>_test_<pid>.xml` into
  `DHM/mj_envs/.../assets/adroit/` per process and deletes it on exit; stale files there are harmless.
* `eval_adroit_pieg.py` asserts on `wandb_group`; running `adroit_eval.sh` without the
  `test_pieg_<level>` argument fails that assert.
