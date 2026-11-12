"""Shared pieces: load the image-free Push-T dataset, normalize it, and evaluate a policy in gym-pusht.

Dataset: lerobot/pusht_keypoints (206 human demos, 25,650 frames, 10 Hz)
    observation.state             (2,)  agent x, y in pixels [0, 512]
    observation.environment_state (16,) 8 (x, y) keypoints on the T-block
    action                        (2,)  target x, y the agent is pulled toward
We concatenate state + keypoints into one 18-dim observation, the same way for
the dataset and for the simulator (obs_type="environment_state_agent_pos").

Evaluation metric (as in the Diffusion Policy paper): for each episode, the
MAX reward reached at any step (reward = coverage of the goal T, ~1.0 at 95%
overlap). We report the mean over 50 seeds and the success rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  registers gym_pusht/PushT-v0
import numpy as np
import torch

DATASET_ID = "lerobot/pusht_keypoints"
OBS_DIM, ACTION_DIM = 18, 2
EVAL_SEED_OFFSET = 100_000  # eval seeds never overlap with seeds used while developing
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


# ---------------------------------------------------------------- data

@dataclass
class Episodes:
    obs: np.ndarray  # (N, 18)
    action: np.ndarray  # (N, 2)
    episode_index: np.ndarray  # (N,)

    @property
    def num_episodes(self) -> int:
        return len(np.unique(self.episode_index))

    def select(self, episodes: np.ndarray) -> "Episodes":
        m = np.isin(self.episode_index, episodes)
        return Episodes(self.obs[m], self.action[m], self.episode_index[m])

    def episode(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        m = self.episode_index == i
        return self.obs[m], self.action[m]


def make_obs(agent_pos: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    return np.concatenate([agent_pos, keypoints], axis=-1).astype(np.float32)


def load_episodes() -> Episodes:
    """Download once (cached by Hugging Face) and return all frames as numpy."""
    from datasets import load_dataset

    d = load_dataset(DATASET_ID, split="train").with_format("numpy")[:]
    return Episodes(
        obs=make_obs(d["observation.state"], d["observation.environment_state"]),
        action=d["action"].astype(np.float32),
        episode_index=d["episode_index"].astype(np.int64),
    )


def train_val_split(ep: Episodes, val_frac: float = 0.1, seed: int = 0) -> tuple[Episodes, Episodes]:
    """Split by whole episodes, so validation frames come from unseen demos."""
    ids = np.unique(ep.episode_index)
    rng = np.random.default_rng(seed)
    val = rng.choice(ids, size=max(1, int(len(ids) * val_frac)), replace=False)
    return ep.select(np.setdiff1d(ids, val)), ep.select(val)


class Normalizer:
    """Per-dimension min/max scaling to [-1, 1]. Fit on training data only."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.lo = np.asarray(lo, np.float32)
        self.hi = np.asarray(hi, np.float32)
        self.span = np.maximum(self.hi - self.lo, 1e-6)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        return cls(x.min(0), x.max(0))

    def normalize(self, x):
        lo, span = self._as(x, self.lo), self._as(x, self.span)
        return (x - lo) / span * 2 - 1

    def unnormalize(self, x):
        lo, span = self._as(x, self.lo), self._as(x, self.span)
        return (x + 1) / 2 * span + lo

    @staticmethod
    def _as(x, a):
        return torch.as_tensor(a, device=x.device) if isinstance(x, torch.Tensor) else a

    def state_dict(self) -> dict:
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        return cls(np.array(d["lo"]), np.array(d["hi"]))


# ---------------------------------------------------------------- evaluation

def make_env(render: bool = False) -> gym.Env:
    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="rgb_array" if render else None,
        max_episode_steps=300,
    )


def rollout(policy, seed: int, record: bool = False) -> dict:
    """Run one episode. `policy(obs)` takes a raw (18,) observation and returns
    a raw (2,) action, or a (k, 2) chunk of actions to execute in order."""
    env = make_env(render=record)
    raw, _ = env.reset(seed=seed)
    obs = make_obs(raw["agent_pos"], raw["environment_state"])
    rewards, agent_path, frames, success, done = [], [obs[:2]], [], False, False
    if record:
        frames.append(env.render())
    while not done:
        actions = np.asarray(policy(obs), np.float32).reshape(-1, ACTION_DIM)
        for a in actions:
            raw, r, term, trunc, info = env.step(a)
            obs = make_obs(raw["agent_pos"], raw["environment_state"])
            rewards.append(float(r))
            agent_path.append(obs[:2])
            success |= bool(info["is_success"])
            if record:
                frames.append(env.render())
            done = term or trunc
            if done:
                break
    env.close()
    return {"max_reward": max(rewards), "success": success, "steps": len(rewards),
            "rewards": rewards, "agent_path": np.array(agent_path), "frames": frames}


def evaluate(policy, num_episodes: int = 50, video_dir: Path | None = None, num_videos: int = 4) -> dict:
    """Mean max reward and success rate over seeds EVAL_SEED_OFFSET + 0..num_episodes-1."""
    from tqdm import tqdm

    episodes = []
    for i in tqdm(range(num_episodes), desc="eval", leave=False):
        record = video_dir is not None and i < num_videos
        ep = rollout(policy, EVAL_SEED_OFFSET + i, record=record)
        if record:
            save_video(ep["frames"], Path(video_dir) / f"seed{EVAL_SEED_OFFSET + i}_r{ep['max_reward']:.2f}.mp4")
        ep.pop("frames")
        episodes.append(ep)
    max_rewards = np.array([e["max_reward"] for e in episodes])
    return {
        "mean_max_reward": float(max_rewards.mean()),
        "std_max_reward": float(max_rewards.std()),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "max_rewards": max_rewards.tolist(),
        "episodes": episodes,
    }


def save_video(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def random_policy(seed: int = 0):
    rng = np.random.default_rng(seed)
    return lambda obs: rng.uniform(0, 512, size=2)
