"""Run a policy in the gym-pusht simulator and measure how well it does.

Metric (same as the Diffusion Policy paper): for each episode, take the MAX
reward reached at any step. Reward is how much of the T-shaped goal region the
block covers (normalized so ~0.95 overlap = 1.0). An episode counts as a
success if the env reports `is_success` at any step. Report the mean over
episodes; use many seeds (50+) because results are noisy.

A policy is anything with a `predict` method:

    predict(obs_history: np.ndarray) -> np.ndarray
        obs_history: (obs_horizon, 18) raw (un-normalized) observations,
                     oldest first, most recent last.
        returns:     (k, 2) raw actions (pixel coordinates) to execute now,
                     in order. k >= 1. After executing them, predict is called
                     again with the new observations ("receding horizon").

The policy is responsible for normalizing inputs and un-normalizing outputs,
and (for diffusion) for picking which slice of its predicted chunk to execute.
"""

from collections import deque
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers the gym_pusht/PushT-v0 env)
import numpy as np

from pusht_dp.data import make_obs


class Policy(Protocol):
    obs_horizon: int

    def predict(self, obs_history: np.ndarray) -> np.ndarray: ...


def make_env(render: bool = False) -> gym.Env:
    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="rgb_array" if render else None,
        max_episode_steps=300,
    )


def rollout(policy: Policy, seed: int, env: gym.Env | None = None, record: bool = False):
    """Run one episode. Returns (max_reward, success, frames)."""
    env = env or make_env(render=record)
    raw, _ = env.reset(seed=seed)
    obs = make_obs(raw["agent_pos"], raw["environment_state"])
    # Pad history at the start by repeating the first observation, which
    # matches how the training windows are padded.
    history = deque([obs] * policy.obs_horizon, maxlen=policy.obs_horizon)

    max_reward, success, frames = 0.0, False, []
    if record:
        frames.append(env.render())
    done = False
    while not done:
        actions = np.asarray(policy.predict(np.stack(history)), dtype=np.float32)
        assert actions.ndim == 2 and actions.shape[1] == 2, actions.shape
        for a in actions:
            raw, reward, terminated, truncated, info = env.step(a)
            history.append(make_obs(raw["agent_pos"], raw["environment_state"]))
            max_reward = max(max_reward, float(reward))
            success = success or bool(info["is_success"])
            if record:
                frames.append(env.render())
            done = terminated or truncated
            if done:
                break
    return max_reward, success, frames


def evaluate(
    policy: Policy,
    num_episodes: int = 50,
    seed_offset: int = 100_000,
    video_dir: str | Path | None = None,
    num_videos: int = 4,
) -> dict:
    """Evaluate over many seeds. Seeds start at seed_offset so they don't
    overlap with anything you might have used during development.

    If video_dir is set, saves mp4s of the first `num_videos` episodes.
    """
    from tqdm import tqdm

    rewards, successes = [], []
    env, render_env = make_env(), make_env(render=True) if video_dir else None
    for i in tqdm(range(num_episodes), desc="eval", leave=False):
        record = video_dir is not None and i < num_videos
        r, s, frames = rollout(policy, seed_offset + i, env=render_env if record else env, record=record)
        rewards.append(r)
        successes.append(s)
        if record:
            save_video(frames, Path(video_dir) / f"episode_{i:03d}.mp4")
    return {
        "mean_max_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "max_rewards": rewards,
    }


def save_video(frames: list[np.ndarray], path: str | Path, fps: int = 10) -> None:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


class RandomPolicy:
    """Sanity-check baseline: uniform random targets. Expect very low reward."""

    obs_horizon = 1

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs_history: np.ndarray) -> np.ndarray:
        return self.rng.uniform(0, 512, size=(1, 2))
