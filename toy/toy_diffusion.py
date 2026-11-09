"""Step 3 (you write the diffusion parts): DDPM on 2D points, no robots.

    uv run python toy/toy_diffusion.py

The data and plotting are provided. Learning a distribution with two separate
clusters is the same problem as Push-T's "go left or go right": a model trained
with MSE would put everything in the middle, but a diffusion model should
recover both clusters.

What you implement (Ho et al. 2020, "Denoising Diffusion Probabilistic Models"):

  1. Noise schedule. Choose T (e.g. 100) and betas (start with linear,
     1e-4 -> 0.02). Precompute alphas = 1 - betas and
     alpha_bar = cumprod(alphas).

  2. Forward process (adding noise). For clean x0, a random step t, and
     eps ~ N(0, I):
         x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * eps

  3. Model. An MLP that takes (x_t, t) and predicts eps. It needs to know t:
     try a sinusoidal embedding of t concatenated to x_t.

  4. Training. loss = MSE(model(x_t, t), eps). That's the whole objective.

  5. Sampling (reverse process). Start from x_T ~ N(0, I). For t = T-1 ... 0:
         eps_hat = model(x_t, t)
         mean = (x_t - betas[t] / sqrt(1 - alpha_bar[t]) * eps_hat) / sqrt(alphas[t])
         x_{t-1} = mean + sqrt(betas[t]) * z,   z ~ N(0, I)  (no noise at t=0)

  6. Plot samples at several t during sampling with plot_snapshots, and compare
     the final samples to the data.

Experiments once it works: change T; change the schedule; add a third cluster.
What happens if you train the same MLP with plain MSE to output points directly?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = Path("outputs/toy")


def sample_data(n: int, seed: int | None = None) -> torch.Tensor:
    """Two Gaussian blobs at (-1, 0) and (1, 0). Returns (n, 2)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-1.0, 0.0], [1.0, 0.0]])
    which = rng.integers(0, 2, size=n)
    x = centers[which] + 0.1 * rng.standard_normal((n, 2))
    return torch.tensor(x, dtype=torch.float32)


def plot_snapshots(snapshots: dict[int, torch.Tensor], path: Path = OUT / "sampling.png") -> None:
    """snapshots: {timestep: (n, 2) samples at that step}, shown left to right
    in descending t (noise -> data)."""
    items = sorted(snapshots.items(), reverse=True)
    fig, axes = plt.subplots(1, len(items), figsize=(3 * len(items), 3), squeeze=False)
    for ax, (t, x) in zip(axes[0], items):
        x = x.detach().cpu().numpy()
        ax.scatter(x[:, 0], x[:, 1], s=2, alpha=0.5)
        ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", title=f"t = {t}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


if __name__ == "__main__":
    data = sample_data(2000, seed=0)
    plot_snapshots({0: data}, OUT / "data.png")  # what you're trying to generate

    # TODO: noise schedule, model, training loop, sampling loop (see docstring).
