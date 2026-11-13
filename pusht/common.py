"""Shared code: load the Push-T dataset and evaluate a policy in the simulator.

Dataset: lerobot/pusht_keypoints on Hugging Face (206 human demos, no images)
    observation.state             (2,)  agent x, y (pixels, 0-512)
    observation.environment_state (16,) x, y of 8 keypoints on the T-block
    action                        (2,)  target x, y the agent moves toward
We join state + keypoints into one 18-number feature vector.

Evaluation: run the policy in the gym-pusht simulator. Each episode's score is
the MAX reward it reaches (how well the T covers the goal, 1.0 = success).
We report the mean over 50 fixed seeds and the success rate.
"""

from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers the PushT environment)
import numpy as np

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
EVAL_SEEDS = range(100_000, 100_050)  # 50 seeds, the same for every model


def load_data():
    """Returns features X (N, 18), actions y (N, 2), and the episode id of each frame (N,)."""
    from datasets import load_dataset

    d = load_dataset("lerobot/pusht_keypoints", split="train").with_format("numpy")[:]
    X = np.concatenate([d["observation.state"], d["observation.environment_state"]], axis=1)
    y = d["action"]
    return X.astype(np.float32), y.astype(np.float32), d["episode_index"]


def train_val_split(episode, val_frac=0.1, seed=0):
    """Holdout split by whole episodes, so validation demos are unseen.
    Returns boolean masks (is_train, is_val) over frames."""
    ids = np.unique(episode)
    rng = np.random.default_rng(seed)
    val_ids = rng.choice(ids, size=int(len(ids) * val_frac), replace=False)
    is_val = np.isin(episode, val_ids)
    return ~is_val, is_val


def make_env(render=False):
    return gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos",
                    render_mode="rgb_array" if render else None, max_episode_steps=300)


def run_episode(policy, seed, render=False):
    """Run one episode. `policy(x)` takes an (18,) feature vector and returns an (2,) action.
    Returns the max reward, whether it succeeded, the agent's path, and the frames (if render)."""
    env = make_env(render)
    obs, _ = env.reset(seed=seed)
    max_reward, success, frames = 0.0, False, []
    path = [obs["agent_pos"]]
    done = False
    while not done:
        if render:
            frames.append(env.render())
        x = np.concatenate([obs["agent_pos"], obs["environment_state"]]).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(policy(x))
        max_reward = max(max_reward, reward)
        success = success or info["is_success"]
        path.append(obs["agent_pos"])
        done = terminated or truncated
    if render:
        frames.append(env.render())
    env.close()
    return max_reward, success, np.array(path), frames


def evaluate(policy, seeds=EVAL_SEEDS):
    """Mean max reward and success rate over the evaluation seeds."""
    results = [run_episode(policy, s) for s in seeds]
    rewards = np.array([r[0] for r in results])
    successes = np.array([r[1] for r in results])
    paths = [r[2] for r in results]
    return {"mean_reward": rewards.mean(), "std_reward": rewards.std(),
            "success_rate": successes.mean(), "rewards": rewards, "paths": paths}


def save_video(frames, path, fps=10):
    import imageio.v2 as imageio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def random_policy(x):
    return np.random.uniform(0, 512, size=2)
