"""Baseline 2: an MLP trained with MSE, features -> action.

    python -m pusht.mlp_baseline

Every 20 epochs we also run the model in the simulator, to check whether a
lower validation loss actually gives a better policy.
"""

import json

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from pusht.common import (EVAL_SEEDS, OUTPUTS, evaluate, load_data, random_policy, run_episode,
                          save_video, train_val_split)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = OUTPUTS / "mlp_baseline"
OUT.mkdir(parents=True, exist_ok=True)
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------- 1. data
X, y, episode = load_data()
is_train, is_val = train_val_split(episode)

# Scale features and actions using training data only
x_scaler = StandardScaler().fit(X[is_train])
y_scaler = StandardScaler().fit(y[is_train])
X_train = torch.tensor(x_scaler.transform(X[is_train]), dtype=torch.float)
y_train = torch.tensor(y_scaler.transform(y[is_train]), dtype=torch.float)
X_val = torch.tensor(x_scaler.transform(X[is_val]), dtype=torch.float)
y_val = torch.tensor(y_scaler.transform(y[is_val]), dtype=torch.float)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
print(f"train frames: {len(X_train)}, val frames: {len(X_val)}")


# ---------------------------------------------------------------- 2. model
class MLP(nn.Module):
    def __init__(self, input_size=18, hidden_size=256, output_size=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.output(x)


model = MLP()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def policy(x):
    """Raw 18 features -> raw action in pixels, for the simulator."""
    with torch.no_grad():
        x = torch.tensor(x_scaler.transform(x[None]), dtype=torch.float)
        action = model(x).numpy()
    return y_scaler.inverse_transform(action)[0]


# ---------------------------------------------------------------- 3. training
train_history, val_history, checkpoints = [], [], []
for epoch in range(1, EPOCHS + 1):
    model.train()
    batch_losses = []
    for features, actions in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(features), actions)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val).item()
    train_history.append(np.mean(batch_losses))
    val_history.append(val_loss)

    if epoch % 20 == 0:
        quick = evaluate(policy, seeds=EVAL_SEEDS[:20])
        checkpoints.append((epoch, val_loss, quick["mean_reward"]))
        print(f"epoch {epoch:3d} | train MSE {train_history[-1]:.4f} | val MSE {val_loss:.4f} | "
              f"simulator reward (20 seeds) {quick['mean_reward']:.3f}")

# ---------------------------------------------------------------- 4. evaluation
mlp = evaluate(policy)
rnd = evaluate(random_policy)


def stall_fraction(paths):
    """How often the agent barely moves (< 1 pixel in a step)."""
    steps = np.concatenate([np.linalg.norm(np.diff(p, axis=0), axis=1) for p in paths])
    return float((steps < 1.0).mean())


demo_paths = [X[episode == e, :2] for e in np.unique(episode)]
results = {
    "random": {"mean_reward": rnd["mean_reward"], "success_rate": rnd["success_rate"]},
    "mlp": {"mean_reward": mlp["mean_reward"], "std_reward": mlp["std_reward"],
            "success_rate": mlp["success_rate"], "final_val_mse": val_history[-1]},
    "stall_fraction": {"mlp": stall_fraction(mlp["paths"]), "demos": stall_fraction(demo_paths)},
    "checkpoints": [{"epoch": e, "val_mse": v, "reward_20_seeds": r} for e, v, r in checkpoints],
    "rewards": mlp["rewards"].tolist(),
}
(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))

print(f"\nrandom: mean max reward {rnd['mean_reward']:.3f}, success {rnd['success_rate']:.0%}")
print(f"MLP:    mean max reward {mlp['mean_reward']:.3f} ± {mlp['std_reward']:.3f}, "
      f"success {mlp['success_rate']:.0%}")
print(f"barely moving (<1 px/step): MLP {results['stall_fraction']['mlp']:.1%}, "
      f"demos {results['stall_fraction']['demos']:.1%}")

# ---------------------------------------------------------------- 5. plots and videos
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, EPOCHS + 1), train_history, label="train MSE")
ax.plot(range(1, EPOCHS + 1), val_history, label="validation MSE")
ax.set(xlabel="epoch", ylabel="MSE (scaled actions)", yscale="log")
ax2 = ax.twinx()
ax2.plot([c[0] for c in checkpoints], [c[2] for c in checkpoints], "o-", c="tab:red",
         label="simulator reward (20 seeds)")
ax2.set(ylabel="mean max reward", ylim=(0, 1))
fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88), fontsize=8)
ax.set_title("MLP: lower loss does not mean a better policy")
fig.tight_layout()
fig.savefig(OUT / "loss_vs_reward.png", dpi=150)

# 3 worst and 3 best episodes: replay them and draw the path on the last frame
order = np.argsort(mlp["rewards"])
fig, axes = plt.subplots(2, 3, figsize=(12, 9))
for ax, i in zip(axes.flat, list(order[:3]) + list(order[-3:])):
    seed = EVAL_SEEDS[i]
    reward, _, path, frames = run_episode(policy, seed, render=True)
    save_video(frames, OUT / "videos" / f"seed{seed}_reward{reward:.2f}.mp4")
    ax.imshow(frames[-1], extent=(0, 512, 512, 0))
    ax.plot(path[:, 0], path[:, 1], c="tab:orange", lw=1)
    ax.scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), s=4, cmap="viridis", zorder=3)
    ax.plot(*path[0], "o", c="tab:red", ms=7, zorder=4)
    ax.set(title=f"seed {seed}: reward {reward:.2f}, {len(path) - 1} steps")
    ax.set_xticks([]), ax.set_yticks([])
fig.suptitle("MLP rollouts: 3 worst (top), 3 best (bottom).\n"
             "Path colored by time (dark = early), red dot = start, green T = goal")
fig.tight_layout()
fig.savefig(OUT / "rollouts.png", dpi=150)
print(f"saved plots and videos to {OUT}")
