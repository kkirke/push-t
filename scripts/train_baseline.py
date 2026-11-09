"""Step 2 (you write this): an MLP behavior-cloning baseline trained with MSE.

Goal: build the train -> evaluate loop end to end with the simplest possible
model, and see firsthand how a regression policy fails on Push-T.

Outline:
  1. Data
       episodes = load_episodes()
       ds = PushTWindowDataset(episodes, obs_horizon=1, pred_horizon=1)
       DataLoader(ds, batch_size=256, shuffle=True)
     Each batch: obs (B, 1, 18), action (B, 1, 2), already in [-1, 1].

  2. Model
     A small MLP: 18 -> 256 -> 256 -> 2 with ReLU (or Mish). Flatten obs first.

  3. Training loop
     loss = MSE(model(obs), action). Adam, lr 1e-3. A few thousand steps
     is enough to see it learn. Log the loss (print, or wandb.init + wandb.log).

  4. Wrap the model as a Policy (see pusht_dp/evaluate.py for the interface):
       class MLPPolicy:
           obs_horizon = 1
           def predict(self, obs_history):  # (1, 18) raw
               normalize -> model -> unnormalize -> return shape (1, 2)
     Remember model.eval() and torch.no_grad().

  5. evaluate(policy, num_episodes=50, video_dir="outputs/baseline")
     Compare against the random policy from scripts/explore_env.py.

Questions to answer when it's done:
  - Does lower training loss give you a higher eval reward?
  - Watch the videos: where does the policy get stuck or hesitate? Can you
    connect that to the averaging problem (two valid moves -> their mean)?
"""

raise NotImplementedError("This one's yours. See the docstring above.")
