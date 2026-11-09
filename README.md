# push-t

A learning project: train a diffusion policy on the Push-T task, using
keypoint observations ([`lerobot/pusht_keypoints`](https://huggingface.co/datasets/lerobot/pusht_keypoints))
and the [`gym-pusht`](https://github.com/huggingface/gym-pusht) simulator for evaluation.

## Setup

Uses [uv](https://docs.astral.sh/uv/). Python 3.12; torch is pulled from the CUDA 13.0 index on Windows/Linux.

```
uv sync
uv run pytest
```

## Learning path

Each step adds one new idea.

| Step | What | Where | Status |
|---|---|---|---|
| 1 | Look at the data and the simulator | `scripts/inspect_data.py`, `scripts/explore_env.py` | provided, run them |
| 2 | MLP + MSE baseline; see where regression fails | `scripts/train_baseline.py` | **you write** |
| 3 | DDPM on 2D toy data | `toy/toy_diffusion.py` | **you write** (data + plots provided) |
| 4 | Diffusion policy on Push-T keypoints | `src/pusht_dp/` | **you write** |

Baselines to beat (50 eval episodes, mean of max reward per episode):
random policy ≈ 0.18. The Diffusion Policy paper reports ≈ 0.9+ for state-based Push-T.

## Layout

```
src/pusht_dp/
  data.py       load dataset, Normalizer ([-1,1] scaling), PushTWindowDataset (obs/action windows)
  evaluate.py   run a policy in gym-pusht, compute metrics, save videos
scripts/        runnable entry points (exploration, training)
toy/            2D diffusion playground
tests/          windowing / normalization tests
outputs/        plots, videos (gitignored)
```

## Resources

For practice building reproducible training pipelines: look into Coreweave (formerly called Weights & Biases).

For understanding how diffusion models actually work: there is a relatively advanced but really good tutorial on diffusion by Stanley Chan, as well as some nice video lectures by Yutong He (https://kellyyutonghe.github.io/10799S26/schedule/)

For compute: UW students get 100 free Google Colab credits. You can also use Colab resources to run Python code that you develop locally (see https://github.com/googlecolab/google-colab-cli).

Papers:
- Ho et al. 2020, *Denoising Diffusion Probabilistic Models* 
- Chi et al. 2023, *Diffusion Policy* — https://diffusion-policy.cs.columbia.edu/ 
