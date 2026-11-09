"""Step 1a: look at the demonstration data before training anything.

    uv run python scripts/inspect_data.py

Prints dataset stats and saves plots to outputs/inspect_data/. Things to look
for while you read the plots:
  - How long are episodes? How varied are they?
  - The action is a *target* the agent is pulled toward, not a velocity.
    How far ahead of the agent's position does the action tend to be?
  - Find two episodes where the T starts in a similar place but the human
    took different routes. That is the "multimodality" diffusion handles.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pusht_dp.data import load_episodes

OUT = Path("outputs/inspect_data")


def draw_keypoints(ax, env_state, **kw):
    kp = env_state.reshape(8, 2)
    ax.scatter(kp[:, 0], kp[:, 1], s=12, **kw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ep = load_episodes()
    lengths = np.diff(np.concatenate([[0], ep.episode_ends]))

    print(f"episodes: {ep.num_episodes}, frames: {len(ep.obs)}")
    print(f"episode length: min {lengths.min()}, median {np.median(lengths):.0f}, max {lengths.max()}"
          f"  (at 10 Hz -> {np.median(lengths) / 10:.0f} s median)")
    print(f"obs dim: {ep.obs.shape[1]} (2 agent xy + 8 keypoints x 2)")
    print("obs   min:", ep.obs.min(0).round(0))
    print("obs   max:", ep.obs.max(0).round(0))
    print("action min/max:", ep.action.min(0), ep.action.max(0))
    gap = np.linalg.norm(ep.action - ep.obs[:, :2], axis=1)
    print(f"distance from agent to its action target: median {np.median(gap):.1f} px, 95th pct {np.percentile(gap, 95):.1f} px")

    # 1. Episode length histogram.
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(lengths, bins=30)
    ax.set(xlabel="episode length (frames)", ylabel="count", title="Episode lengths")
    fig.tight_layout()
    fig.savefig(OUT / "episode_lengths.png", dpi=120)

    # 2. A grid of episodes: agent path + T keypoints at start (grey) and end (color).
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for ax, i in zip(axes.flat, np.linspace(0, ep.num_episodes - 1, axes.size).astype(int)):
        obs, act = ep.episode(i)
        draw_keypoints(ax, obs[0, 2:], c="0.6", label="T start")
        draw_keypoints(ax, obs[-1, 2:], c="tab:green", label="T end")
        ax.plot(obs[:, 0], obs[:, 1], "-", lw=1, c="tab:blue", label="agent")
        ax.plot(act[:, 0], act[:, 1], ":", lw=1, c="tab:orange", label="action")
        ax.plot(*obs[0, :2], "o", c="tab:blue")
        ax.set(xlim=(0, 512), ylim=(512, 0), aspect="equal", title=f"episode {i} ({len(obs)} steps)")
        ax.set_xticks([]), ax.set_yticks([])
    axes.flat[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Agent path (blue), action targets (orange), T keypoints start (grey) / end (green)")
    fig.tight_layout()
    fig.savefig(OUT / "episodes.png", dpi=120)

    # 3. Actions vs agent position over time for one episode.
    obs, act = ep.episode(0)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for d, name in enumerate("xy"):
        axes[d].plot(obs[:, d], label=f"agent {name}")
        axes[d].plot(act[:, d], label=f"action {name}", alpha=0.8)
        axes[d].legend(fontsize=8)
    axes[1].set_xlabel("frame (10 Hz)")
    fig.suptitle("Episode 0: the action leads the agent's position")
    fig.tight_layout()
    fig.savefig(OUT / "action_vs_state.png", dpi=120)

    print(f"\nsaved plots to {OUT.resolve()}")


if __name__ == "__main__":
    main()
