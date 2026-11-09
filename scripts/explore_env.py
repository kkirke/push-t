"""Step 1b: get a feel for the simulator, and sanity-check the eval harness.

    uv run python scripts/explore_env.py

Runs a random policy, prints its score, and saves videos to
outputs/explore_env/. Random should score near zero; anything you train
later has to beat this. Open the videos to see what the task looks like:
the blue circle is the agent, the grey T is the block, and the green T is the goal.
"""

from pusht_dp.evaluate import RandomPolicy, evaluate, make_env


def main():
    env = make_env()
    obs, info = env.reset(seed=0)
    print("observation keys:", {k: v.shape for k, v in obs.items()})
    print("action space:", env.action_space)
    print("info at reset:", {k: info[k] for k in ("pos_agent", "block_pose", "goal_pose")})

    results = evaluate(RandomPolicy(), num_episodes=10, video_dir="outputs/explore_env", num_videos=2)
    print(f"random policy: mean max reward {results['mean_max_reward']:.3f}, "
          f"success rate {results['success_rate']:.2f}")
    print("videos saved to outputs/explore_env/")


if __name__ == "__main__":
    main()
