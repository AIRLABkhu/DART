# DART: Diffusion-based Augmentation with Reward-guided Truncation

Official code for **"Diffusion-based Augmentation with Reward-guided Truncation for Visuomotor Policy Generalization"** (IEEE Robotics and Automation Letters, accepted 2026).

DART is a plug-in data augmentation for visual reinforcement learning. Each observation is encoded into the latent space of a pretrained one-step generator (InstaFlow with the TAESD autoencoder), perturbed up to a truncation step *T*, decoded with a single generator step, and blended with the original image. The truncation step *T* is adjusted online from the variability of the episode reward: when returns improve, *T* rises and the augmentation gets stronger; when they stall, *T* is held back or reduced.

## Where is the code?

The paper evaluates DART on three benchmark suites that live on three different codebases, so each one is kept on its own branch. Pick the branch you need and follow its README; every branch is self-contained.

| Branch | Benchmark | Base codebase | What you get |
|---|---|---|---|
| [`dmcontrol-gb`](../../tree/dmcontrol-gb) | DMControl-GB (Walker walk/stand, Cartpole swingup, Ball-in-cup catch, Finger spin) | SAC / SVEA trainer with the generator loaded in-process | full training and evaluation code, scripts, aggregation |
| [`robotic_manipulation`](../../tree/robotic_manipulation) | Robotic Manipulation (Reach, Peg in box) | SGQN-based trainer + a small HTTP generator server | trainer, generator, MuJoCo task assets, scripts |
| [`adroit`](../../tree/adroit) | Adroit (Door, Hammer) | PIE-G / SAM-G | a patch set to apply on top of the public SAM-G repository, plus the generator server |

```bash
git clone https://github.com/AIRLABkhu/DART.git
cd DART
git checkout dmcontrol-gb        # or robotic_manipulation, adroit
```

## What is shared across branches

- **Generator.** InstaFlow-0.9B (one step) with the TAESD autoencoder, used as released and downloaded from Hugging Face on first run. Images are processed at 32 x 32 in latent space and resized back to 84 x 84.
- **Two knobs.** `T_max` caps the truncation step (100 everywhere in the paper); the mixing ratio `lambda` sets the blend between the original and the generated image and is the one value chosen per environment. Each branch README lists the values used in the paper.
- **Reward-guided schedule.** The signed update of *T* uses the episode-reward difference normalized by its running standard deviation, with step size `eta = 0.1` on all simulated benchmarks.

In every branch the mixing ratio is the weight of the *original* image (`aug = lambda * obs + (1 - lambda) * generated`); each README repeats this next to its training command.

## Citation

```bibtex
@article{lee2026dart,
  title   = {Diffusion-based Augmentation with Reward-guided Truncation for Visuomotor Policy Generalization},
  author  = {Lee, Jeong Woon and Kwak, Kyoleen and Kim, Daeho and Hwang, Hyoseok},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}
```

## Acknowledgements

The three branches build on the public codebases of [DGA2 / SVEA (dmcgb)](https://github.com/nicklashansen/dmcontrol-generalization-benchmark), [SGQN](https://github.com/SuReLI/SGQN), and [SAM-G / PIE-G](https://github.com/wadiuvatzy/SAM-G), and on [InstaFlow](https://github.com/gnobitab/InstaFlow) and [TAESD](https://github.com/madebyollin/taesd). Thank you to their authors.

Questions and issues are welcome through the GitHub issue tracker.
