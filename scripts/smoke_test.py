"""Smoke test: load the image-free Push-T dataset and run a random episode in gym-pusht.

Dataset lerobot/pusht_keypoints (no images):
  observation.state             (2,)  agent x, y
  observation.environment_state (16,) 8 (x, y) keypoints on the T-block
  action                        (2,)  target agent x, y, in [0, 512]
The env with obs_type="environment_state_agent_pos" returns the same format.
"""
import gymnasium as gym
import gym_pusht  # noqa: F401  registers gym_pusht/PushT-v0
import torch
from datasets import load_dataset

ds = load_dataset("lerobot/pusht_keypoints", split="train").with_format("torch")
print(f"episodes={len(ds.unique('episode_index'))} frames={len(ds)}")
for k in ["observation.state", "observation.environment_state", "action"]:
    x = ds[:][k]
    print(f"{k} {tuple(x.shape[1:])}: min={x.min().item():.1f} max={x.max().item():.1f}")

env = gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos", max_episode_steps=300)
obs, _ = env.reset(seed=0)
print("env obs:", {k: v.shape for k, v in obs.items()})
done, success, total = False, False, 0.0
while not done:
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    total += r
    success |= bool(info.get("is_success", False))
    done = term or trunc
print(f"random policy: total_reward={total:.3f} success={success}")
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
