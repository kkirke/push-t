"""Loading the Push-T keypoints dataset and turning it into training windows.

Dataset: https://huggingface.co/datasets/lerobot/pusht_keypoints
  - 206 human demonstrations, 25,650 frames total, recorded at 10 Hz
  - observation.state             (2,)  agent (pusher) xy position, pixels in [0, 512]
  - observation.environment_state (16,) 8 keypoints on the T block, (x, y) each
  - action                        (2,)  target xy the agent is driven toward

We concatenate state + environment_state into one 18-dim observation vector,
which matches what the gym-pusht env returns (agent_pos + environment_state).
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_ID = "lerobot/pusht_keypoints"
OBS_DIM = 18
ACTION_DIM = 2


@dataclass
class Episodes:
    """All frames from all episodes, stacked, plus where each episode ends."""

    obs: np.ndarray  # (N, 18) float32
    action: np.ndarray  # (N, 2) float32
    episode_ends: np.ndarray  # (num_episodes,) exclusive end index of each episode

    @property
    def episode_starts(self) -> np.ndarray:
        return np.concatenate([[0], self.episode_ends[:-1]])

    @property
    def num_episodes(self) -> int:
        return len(self.episode_ends)

    def episode(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.episode_starts[i], self.episode_ends[i]
        return self.obs[s:e], self.action[s:e]


def make_obs(agent_pos: np.ndarray, environment_state: np.ndarray) -> np.ndarray:
    """Build the 18-dim observation the same way for the dataset and the env."""
    return np.concatenate([agent_pos, environment_state], axis=-1).astype(np.float32)


def load_episodes(dataset_id: str = DATASET_ID) -> Episodes:
    """Download (cached after the first time) and load the dataset into numpy."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train").with_format("numpy")[:]  # dict of arrays
    episode_index = np.asarray(ds["episode_index"])
    assert np.all(np.diff(episode_index) >= 0), "expected frames sorted by episode"

    obs = make_obs(np.asarray(ds["observation.state"]), np.asarray(ds["observation.environment_state"]))
    action = np.asarray(ds["action"], dtype=np.float32)
    # Index where episode_index changes = start of the next episode.
    change = np.flatnonzero(np.diff(episode_index)) + 1
    episode_ends = np.concatenate([change, [len(episode_index)]])
    return Episodes(obs=obs, action=action, episode_ends=episode_ends)


class Normalizer:
    """Per-dimension min/max scaling to [-1, 1].

    Diffusion models add Gaussian noise with std ~1, so the data needs to live
    on a similar scale. Raw Push-T coordinates are in [0, 512].
    Save the stats with your checkpoint: the policy must un-normalize its
    outputs at eval time using the *training* stats.
    """

    def __init__(self, min_: np.ndarray, max_: np.ndarray):
        self.min = np.asarray(min_, dtype=np.float32)
        self.max = np.asarray(max_, dtype=np.float32)
        # Avoid divide-by-zero for any dimension that never changes.
        self.range = np.maximum(self.max - self.min, 1e-6)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        return cls(x.min(axis=0), x.max(axis=0))

    def normalize(self, x):
        if isinstance(x, torch.Tensor):
            mn, rg = (torch.as_tensor(a, device=x.device) for a in (self.min, self.range))
            return (x - mn) / rg * 2 - 1
        return (x - self.min) / self.range * 2 - 1

    def unnormalize(self, x):
        if isinstance(x, torch.Tensor):
            mn, rg = (torch.as_tensor(a, device=x.device) for a in (self.min, self.range))
            return (x + 1) / 2 * rg + mn
        return (x + 1) / 2 * self.range + self.min

    def state_dict(self) -> dict:
        return {"min": self.min, "max": self.max}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        return cls(d["min"], d["max"])


class PushTWindowDataset(Dataset):
    """Each sample is a short window of observations and a longer window of actions.

    For a frame t, with obs_horizon=To and pred_horizon=Tp:
        obs    = frames [t-To+1, ..., t]           -> (To, 18)
        action = frames [t-To+1, ..., t-To+Tp]     -> (Tp, 2)

    So the action window starts at the same frame as the oldest observation and
    extends into the future. At eval time you execute the actions starting at
    index To-1 (the "current" step), not index 0.

    Windows never cross episode boundaries: indices that fall before the start
    or after the end of an episode are clamped, i.e. the first/last frame is
    repeated. With To=1, Tp=1 this reduces to plain (obs_t, action_t) pairs,
    which is what the MLP baseline needs.
    """

    def __init__(
        self,
        episodes: Episodes,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        obs_normalizer: Normalizer | None = None,
        action_normalizer: Normalizer | None = None,
    ):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_normalizer = obs_normalizer or Normalizer.fit(episodes.obs)
        self.action_normalizer = action_normalizer or Normalizer.fit(episodes.action)
        self.obs = self.obs_normalizer.normalize(episodes.obs).astype(np.float32)
        self.action = self.action_normalizer.normalize(episodes.action).astype(np.float32)

        # Precompute every window's frame indices once.
        obs_idx, act_idx = [], []
        for s, e in zip(episodes.episode_starts, episodes.episode_ends):
            t = np.arange(s, e)[:, None]  # (episode_len, 1)
            first = t - obs_horizon + 1
            obs_idx.append(np.clip(first + np.arange(obs_horizon), s, e - 1))
            act_idx.append(np.clip(first + np.arange(pred_horizon), s, e - 1))
        self.obs_idx = np.concatenate(obs_idx)
        self.act_idx = np.concatenate(act_idx)

    def __len__(self) -> int:
        return len(self.obs_idx)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.from_numpy(self.obs[self.obs_idx[i]]),  # (To, 18)
            "action": torch.from_numpy(self.action[self.act_idx[i]]),  # (Tp, 2)
        }
