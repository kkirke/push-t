# push-t

For practice building reproducible training pipelines: look into Coreweave (formerly called Weights & Biases).

For understanding how diffusion models actually work: there is a relatively advanced but really good tutorial on diffusion by Stanley Chan, as well as some nice video lectures by Yutong He (https://kellyyutonghe.github.io/10799S26/schedule/)

For compute: UW students get 100 free Google Colab credits. You can also use Colab resources to run Python code that you develop locally (see https://github.com/googlecolab/google-colab-cli).

## Setup

```bash
conda create -n pusht python=3.10 -y
conda activate pusht
pip install torch datasets gym-pusht matplotlib "imageio[ffmpeg]" wandb tqdm pytest
# opencv-python and pygame both bundle SDL2 on macOS; keep only the headless OpenCV
pip uninstall -y opencv-python && pip install --force-reinstall --no-deps opencv-python-headless==4.12.0.88
python scripts/smoke_test.py
```

`requirements.txt` pins the exact versions that were tested (macOS arm64, Python 3.10).

Image-free dataset: `lerobot/pusht_keypoints` (206 episodes, 25,650 frames, 10 Hz).
Matching env: `gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos")`.

## Experiments (`pusht/`)

Run from the repo root with the `pusht` conda env active. Outputs (plots, videos, `results.json`) go to `outputs/`.

| Command | What it does | Time on a laptop |
| --- | --- | --- |
| `python -m pusht.mlp_baseline` | MLP behavior cloning (obs → action, MSE), eval on 50 seeds vs. a random policy, failure-case plots and videos | ~1 min |
| `python -m pusht.toy_diffusion` | DDPM on two 2D clusters, compared with MSE regression | ~30 s |

`pusht/common.py` holds the shared data loading, normalization and evaluation (mean max reward over seeds 100000–100049, same metric as the Diffusion Policy paper).
