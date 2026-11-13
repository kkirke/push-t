"""DDPM on 2D points, no robots.

    uv run python toy/toy_diffusion.py

The data and plotting are provided. Learning a distribution with two separate
clusters is the same problem as Push-T's "go left or go right": a model trained
with MSE would put everything in the middle, but a diffusion model should
recover both clusters.
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

T = 100
betas = torch.linspace(1e-4, 0.1, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def add_noise(x0, t):
    a = alpha_bar[t].unsqueeze(1)
    eps = torch.randn_like(x0)
    x_t = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps
    return x_t, eps

class NoiseModel(torch.nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden), 
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, x_t, t):
        t = t.float() / T
        t = t.unsqueeze(1)
        h = torch.cat([x_t, t], dim=1)
        return self.net(h)

if __name__ == "__main__":
    data = sample_data(2000, seed=0)
    plot_snapshots({0: data}, OUT / "data.png")  # what you're trying to generate

    model = NoiseModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(5000):
        x0 = sample_data(256)                     
        t = torch.randint(0, T, (256,))
        x_t, eps = add_noise(x0, t)
        pred = model(x_t, t)
        loss = torch.nn.functional.mse_loss(pred, eps)

        optimizer.zero_grad() 
        loss.backward()       
        optimizer.step()   

        if step % 500 == 0:
            print(step, loss.item())

with torch.no_grad():
    x = torch.randn(2000, 2)
    snapshots = {99: x}
    for step in reversed(range(T)):
        t = torch.full((2000,), step)
        eps_hat = model(x, t)

        mean = (x - betas[step] / torch.sqrt(1 - alpha_bar[step]) * eps_hat) / torch.sqrt(alphas[step])

        if step > 0:
            x = mean + torch.sqrt(betas[step]) * torch.randn_like(x)
        else:
            x = mean

        if step in (75, 50, 25, 10, 0):
            snapshots[step] = x

    plot_snapshots(snapshots, OUT / "sampling.png")