"""Baseline 1: linear regression, features -> action.

    python -m pusht.linear_baseline

Fits action = W x + b with scikit-learn, reports regression metrics
(MSE, MAE, R^2) on held-out episodes, then runs the model in the simulator.
"""

import json

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pusht.common import OUTPUTS, evaluate, load_data, train_val_split

OUT = OUTPUTS / "linear_baseline"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Data: holdout split by episode
X, y, episode = load_data()
is_train, is_val = train_val_split(episode)
print(f"train frames: {is_train.sum()}, val frames: {is_val.sum()}")

# 2. Fit
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X[is_train], y[is_train])

# 3. Regression metrics on validation episodes (actions are in pixels)
y_pred = model.predict(X[is_val])
metrics = {
    "val_mse": mean_squared_error(y[is_val], y_pred),
    "val_mae": mean_absolute_error(y[is_val], y_pred),
    "val_r2": r2_score(y[is_val], y_pred),
}
print(f"validation MSE {metrics['val_mse']:.1f} px^2 | MAE {metrics['val_mae']:.1f} px | R^2 {metrics['val_r2']:.3f}")


# 4. Run it in the simulator
def policy(x):
    return model.predict(x[None])[0]


result = evaluate(policy)
metrics.update(mean_reward=result["mean_reward"], std_reward=result["std_reward"],
               success_rate=result["success_rate"])
print(f"simulator: mean max reward {result['mean_reward']:.3f} ± {result['std_reward']:.3f}, "
      f"success rate {result['success_rate']:.0%}")

(OUT / "results.json").write_text(json.dumps({k: float(v) for k, v in metrics.items()}, indent=2))
