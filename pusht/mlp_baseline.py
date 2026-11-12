"""MLP behavior-cloning baseline on image-free Push-T.

    python -m pusht.mlp_baseline            # from the repo root
    python -m pusht.mlp_baseline --steps 2000 --eval-episodes 10   # quick run

What it does
  1. Splits the 206 demos by episode (90% train / 10% val) and normalizes to [-1, 1].
  2. Trains an MLP  obs_t (18) -> action_t (2)  with MSE. This regresses to the
     MEAN action for each state, which is the weakness we want to expose.
  3. Every --eval-every steps, runs a short eval (20 seeds) so we can see
     whether lower loss actually means a better policy.
  4. Final eval on 50 seeds, compared to a random policy, with videos.
  5. Failure-case statistics: how often the agent stalls and how jittery its
     motion is, compared with the human demos.

Outputs go to outputs/mlp_baseline/.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib
import numpy as np
import torch
from torch import nn

from pusht.common import (ACTION_DIM, EVAL_SEED_OFFSET, OBS_DIM, OUTPUTS, Normalizer, evaluate, load_episodes,
                          random_policy, rollout, train_val_split)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = OUTPUTS / "mlp_baseline"


def make_mlp(hidden: int = 256, layers: int = 3) -> nn.Module:
    dims = [OBS_DIM] + [hidden] * layers
    mods = []
    for a, b in zip(dims[:-1], dims[1:]):
        mods += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*mods, nn.Linear(hidden, ACTION_DIM))


class MLPPolicy:
    """Wraps the trained network so it takes and returns raw pixel coordinates."""

    def __init__(self, model: nn.Module, obs_norm: Normalizer, act_norm: Normalizer):
        self.model, self.obs_norm, self.act_norm = model.eval(), obs_norm, act_norm

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(self.obs_norm.normalize(obs)).float()[None]
        return self.act_norm.unnormalize(self.model(x)[0].numpy())


def motion_stats(paths: list[np.ndarray]) -> dict:
    """Stall = agent moves < 1 px in a step. Jitter = how often the direction of
    motion reverses (angle between consecutive moves > 90 degrees)."""
    stalls, reversals, n = 0, 0, 0
    for p in paths:
        d = np.diff(p, axis=0)
        speed = np.linalg.norm(d, axis=1)
        stalls += int((speed < 1.0).sum())
        n += len(d)
        moving = speed >= 1.0
        a, b = d[:-1][moving[:-1] & moving[1:]], d[1:][moving[:-1] & moving[1:]]
        reversals += int(((a * b).sum(1) < 0).sum())
    return {"stall_frac": stalls / max(n, 1), "reversal_frac": reversals / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=2_000)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- data
    ep = load_episodes()
    train, val = train_val_split(ep, seed=args.seed)
    obs_norm, act_norm = Normalizer.fit(train.obs), Normalizer.fit(train.action)
    Xtr = torch.from_numpy(obs_norm.normalize(train.obs)).float()
    Ytr = torch.from_numpy(act_norm.normalize(train.action)).float()
    Xva = torch.from_numpy(obs_norm.normalize(val.obs)).float()
    Yva = torch.from_numpy(act_norm.normalize(val.action)).float()
    print(f"train: {train.num_episodes} episodes / {len(Xtr)} frames | val: {val.num_episodes} / {len(Xva)}")

    # ---- train
    model = make_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log, checkpoints = [], []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(Xtr), (args.batch,))
        loss = nn.functional.mse_loss(model(Xtr[idx]), Ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(model(Xva), Yva).item()
            model.train()
            log.append({"step": step, "train": loss.item(), "val": val_loss})
        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate(MLPPolicy(model, obs_norm, act_norm), num_episodes=20)
            model.train()
            checkpoints.append({"step": step, "val_loss": log[-1]["val"], "mean_max_reward": r["mean_max_reward"]})
            print(f"step {step:>6} | train {loss.item():.4f} | val {log[-1]['val']:.4f} | "
                  f"eval(20) reward {r['mean_max_reward']:.3f} | {time.time() - t0:.0f}s")

    torch.save({"model": model.state_dict(), "obs_norm": obs_norm.state_dict(),
                "act_norm": act_norm.state_dict(), "args": vars(args)}, OUT / "model.pt")

    # ---- final eval: MLP vs random, same seeds
    policy = MLPPolicy(model, obs_norm, act_norm)
    mlp = evaluate(policy, num_episodes=args.eval_episodes, video_dir=OUT / "videos", num_videos=6)
    rnd = evaluate(random_policy(), num_episodes=args.eval_episodes)
    demo_paths = [ep.episode(i)[0][:, :2] for i in np.unique(ep.episode_index)]
    stats = {
        "mlp": motion_stats([e["agent_path"] for e in mlp["episodes"]]),
        "demos": motion_stats(demo_paths),
    }

    results = {
        "eval_episodes": args.eval_episodes,
        "mlp": {k: mlp[k] for k in ("mean_max_reward", "std_max_reward", "success_rate", "max_rewards")},
        "random": {k: rnd[k] for k in ("mean_max_reward", "std_max_reward", "success_rate")},
        "motion": stats,
        "checkpoints": checkpoints,
        "final_val_loss": log[-1]["val"],
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    # ---- plots
    s = np.array([[r["step"], r["train"], r["val"]] for r in log])
    c = np.array([[r["step"], r["mean_max_reward"]] for r in checkpoints])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s[:, 0], s[:, 1], label="train MSE", alpha=0.6)
    ax.plot(s[:, 0], s[:, 2], label="val MSE")
    ax.set(xlabel="training step", ylabel="MSE (normalized actions)", yscale="log")
    ax2 = ax.twinx()
    ax2.plot(c[:, 0], c[:, 1], "o-", c="tab:red", label="eval reward (20 seeds)")
    ax2.set(ylabel="mean max reward", ylim=(0, 1))
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88), fontsize=8)
    ax.set_title("MLP baseline: val loss flat after 2k steps, eval reward swings")
    fig.tight_layout()
    fig.savefig(OUT / "loss_vs_reward.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(rnd["max_rewards"], bins=bins, alpha=0.6, label=f"random ({rnd['mean_max_reward']:.2f})")
    ax.hist(mlp["max_rewards"], bins=bins, alpha=0.6, label=f"MLP ({mlp['mean_max_reward']:.2f})")
    ax.set(xlabel="max reward in episode", ylabel="episodes", title=f"Per-episode score, {args.eval_episodes} seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "reward_hist.png", dpi=150)

    # Worst and best rollouts: replay the seed (policy and env are deterministic),
    # then draw the agent's path on top of the final simulator frame.
    order = np.argsort(mlp["max_rewards"])
    picks = list(order[:3]) + list(order[-3:])
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    for ax, i in zip(axes.flat, picks):
        e = rollout(policy, EVAL_SEED_OFFSET + int(i), record=True)
        p = e["agent_path"]
        ax.imshow(e["frames"][-1], extent=(0, 512, 512, 0))
        ax.plot(p[:, 0], p[:, 1], "-", lw=1, c="tab:orange", alpha=0.8)
        ax.scatter(p[:, 0], p[:, 1], c=np.arange(len(p)), s=4, cmap="viridis", zorder=3)
        ax.plot(*p[0], "o", c="tab:red", ms=7, zorder=4)
        ax.set(xlim=(0, 512), ylim=(512, 0), aspect="equal",
               title=f"seed {EVAL_SEED_OFFSET + int(i)}: reward {e['max_reward']:.2f}, {e['steps']} steps")
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("MLP rollouts, final frame: 3 worst (top), 3 best (bottom).\n"
                 "Path colored by time (dark = early), red dot = start, green T = goal")
    fig.tight_layout()
    fig.savefig(OUT / "rollouts.png", dpi=150)

    print("\n=== results ===")
    print(f"random: mean max reward {rnd['mean_max_reward']:.3f}, success {rnd['success_rate']:.0%}")
    print(f"MLP:    mean max reward {mlp['mean_max_reward']:.3f} ± {mlp['std_max_reward']:.3f}, "
          f"success {mlp['success_rate']:.0%}")
    print(f"stall fraction  MLP {stats['mlp']['stall_frac']:.1%} vs demos {stats['demos']['stall_frac']:.1%}")
    print(f"reversal frac.  MLP {stats['mlp']['reversal_frac']:.1%} vs demos {stats['demos']['reversal_frac']:.1%}")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
