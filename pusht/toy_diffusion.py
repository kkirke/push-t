"""Toy DDPM on two clusters of 2D points, compared with plain MSE regression.

    python -m pusht.toy_diffusion

Why: Push-T demos often split into two valid moves for the same state (go
around the block on the left or on the right). This toy has the same
structure: the data is two clusters at (-1, 0) and (1, 0).
  - An MSE model asked to "output a data point" learns the MEAN, (0, 0),
    which is in neither cluster. That is the MLP baseline's failure in 2D.
  - A diffusion model learns to remove noise step by step, and its samples
    land in BOTH clusters.

DDPM (Ho et al. 2020), T steps, linear betas:
  forward:  x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,   eps ~ N(0, I)
  training: loss = || eps - eps_theta(x_t, t) ||^2
  sampling: x_{t-1} = (x_t - beta_t / sqrt(1 - abar_t) * eps_theta) / sqrt(alpha_t) + sqrt(beta_t) z

Outputs go to outputs/toy_diffusion/.
"""

from __future__ import annotations

import argparse
import json
import math

import matplotlib
import numpy as np
import torch
from torch import nn

from pusht.common import OUTPUTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = OUTPUTS / "toy_diffusion"
CENTERS = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])


def sample_data(n: int, std: float = 0.1, gen: torch.Generator | None = None) -> torch.Tensor:
    which = torch.randint(0, len(CENTERS), (n,), generator=gen)
    return CENTERS[which] + std * torch.randn(n, 2, generator=gen)


class Schedule:
    """Linear beta schedule and the quantities DDPM needs, precomputed."""

    def __init__(self, T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        ab = self.abar[t][:, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps


def timestep_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """Sinusoidal embedding so the network knows how noisy its input is."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class NoisePredictor(nn.Module):
    """eps_theta(x_t, t): MLP on [x_t, embed(t)]."""

    def __init__(self, hidden: int = 128, emb: int = 32):
        super().__init__()
        self.emb = emb
        self.net = nn.Sequential(
            nn.Linear(2 + emb, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, timestep_embedding(t, self.emb)], dim=1))


@torch.no_grad()
def sample(model: nn.Module, sched: Schedule, n: int, snapshot_at: tuple[int, ...] = ()):
    """Reverse process from pure noise. Returns final samples and {t: x_t} snapshots."""
    x = torch.randn(n, 2)
    snaps = {sched.T: x.clone()}
    for t in reversed(range(sched.T)):
        tt = torch.full((n,), t)
        eps_hat = model(x, tt)
        beta, alpha, abar = sched.betas[t], sched.alphas[t], sched.abar[t]
        mean = (x - beta / (1 - abar).sqrt() * eps_hat) / alpha.sqrt()
        x = mean + (beta.sqrt() * torch.randn_like(x) if t > 0 else 0)
        if t in snapshot_at:
            snaps[t] = x.clone()
    return x, snaps


def cluster_stats(x: torch.Tensor, radius: float = 0.3) -> dict:
    """Fraction of samples within `radius` of each center, and in neither."""
    d = torch.cdist(x, CENTERS)
    near = d.min(1).values < radius
    nearest = d.argmin(1)
    return {"left": float((near & (nearest == 0)).float().mean()),
            "right": float((near & (nearest == 1)).float().mean()),
            "neither": float((~near).float().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--steps", type=int, default=5_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    sched = Schedule(T=args.T)

    # ---- diffusion: predict the noise that was added
    model = NoisePredictor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(args.steps):
        x0 = sample_data(args.batch)
        t = torch.randint(0, sched.T, (args.batch,))
        eps = torch.randn_like(x0)
        loss = nn.functional.mse_loss(model(sched.add_noise(x0, t, eps), t), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"diffusion step {step + 1}: loss {np.mean(losses[-1000:]):.4f}")

    snap_ts = tuple(t for t in (75, 50, 25, 10, 0) if t < sched.T)
    diff_x, snaps = sample(model, sched, args.n_samples, snapshot_at=snap_ts)

    # ---- MSE baseline: same kind of MLP, maps random noise z -> a data point.
    # Inputs and targets are unpaired, so the loss-minimizing output is E[x] = (0, 0).
    mse = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(mse.parameters(), lr=1e-3)
    for _ in range(args.steps):
        loss = nn.functional.mse_loss(mse(torch.randn(args.batch, 2)), sample_data(args.batch))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse_x = mse(torch.randn(args.n_samples, 2))

    data = sample_data(args.n_samples, gen=torch.Generator().manual_seed(123))
    results = {
        "T": sched.T,
        "final_diffusion_loss": float(np.mean(losses[-500:])),
        "data": cluster_stats(data),
        "diffusion": cluster_stats(diff_x),
        "mse": cluster_stats(mse_x),
        "mse_mean_output": mse_x.mean(0).tolist(),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    # ---- plots
    def scatter(ax, x, title, color="tab:blue"):
        x = x.numpy()
        ax.scatter(x[:, 0], x[:, 1], s=2, alpha=0.4, c=color)
        ax.scatter(*CENTERS.T.numpy(), marker="x", c="k", s=40)
        ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", title=title)
        ax.set_xticks([-2, 0, 2]), ax.set_yticks([-2, 0, 2])

    items = sorted(snaps.items(), reverse=True)
    fig, axes = plt.subplots(1, len(items), figsize=(2.8 * len(items), 3.1))
    for ax, (t, x) in zip(axes, items):
        scatter(ax, x, f"t = {t}" + (" (noise)" if t == sched.T else " (samples)" if t == 0 else ""))
    fig.suptitle("Reverse diffusion: pure noise is denoised into the two clusters")
    fig.tight_layout()
    fig.savefig(OUT / "denoising.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    d, f, m = results["data"], results["diffusion"], results["mse"]
    scatter(axes[0], data, "Training data", "0.4")
    scatter(axes[1], diff_x, f"Diffusion samples\nleft {f['left']:.0%} / right {f['right']:.0%} / neither {f['neither']:.0%}")
    scatter(axes[2], mse_x, f"MSE regression outputs\nneither cluster: {m['neither']:.0%}", "tab:red")
    fig.suptitle("Two valid answers: diffusion keeps both, MSE averages them")
    fig.tight_layout()
    fig.savefig(OUT / "diffusion_vs_mse.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    k = 100
    ax.plot(np.convolve(losses, np.ones(k) / k, mode="valid"))
    ax.set(xlabel="training step", ylabel="noise-prediction MSE", title="Toy diffusion training loss (100-step average)")
    fig.tight_layout()
    fig.savefig(OUT / "loss.png", dpi=150)

    print(json.dumps(results, indent=2))
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
