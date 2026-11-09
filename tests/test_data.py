"""Tests for windowing and normalization, using tiny fake episodes (no download)."""

import numpy as np
import torch

from pusht_dp.data import Episodes, Normalizer, PushTWindowDataset


def fake_episodes():
    # Two episodes of lengths 5 and 3. Every value encodes its global frame
    # index so we can check exactly which frames ended up in each window.
    n = 8
    frame = np.arange(n, dtype=np.float32)
    obs = np.repeat(frame[:, None], 18, axis=1)
    action = np.repeat(frame[:, None], 2, axis=1)
    return Episodes(obs=obs, action=action, episode_ends=np.array([5, 8]))


def raw_frames(ds, x, normalizer):
    return normalizer.unnormalize(x)[:, 0].round().astype(int).tolist()


def test_windows_pad_and_stay_in_episode():
    ep = fake_episodes()
    ds = PushTWindowDataset(ep, obs_horizon=2, pred_horizon=4)
    assert len(ds) == 8  # one window per frame

    def frames(i):
        s = ds[i]
        return (raw_frames(ds, s["obs"].numpy(), ds.obs_normalizer),
                raw_frames(ds, s["action"].numpy(), ds.action_normalizer))

    # First frame of episode 0: oldest obs is padded with frame 0.
    assert frames(0) == ([0, 0], [0, 0, 1, 2])
    # Last frame of episode 0: future actions are padded with frame 4,
    # and never leak into episode 1 (frame 5).
    assert frames(4) == ([3, 4], [3, 4, 4, 4])
    # First frame of episode 1 must not include frame 4 from episode 0.
    assert frames(5) == ([5, 5], [5, 5, 6, 7])


def test_baseline_shapes():
    ds = PushTWindowDataset(fake_episodes(), obs_horizon=1, pred_horizon=1)
    s = ds[3]
    assert s["obs"].shape == (1, 18) and s["action"].shape == (1, 2)


def test_normalizer_roundtrip():
    x = np.random.default_rng(0).uniform(0, 512, size=(100, 3)).astype(np.float32)
    n = Normalizer.fit(x)
    z = n.normalize(x)
    assert z.min() >= -1 - 1e-5 and z.max() <= 1 + 1e-5
    np.testing.assert_allclose(n.unnormalize(z), x, rtol=1e-5, atol=1e-3)
    zt = n.normalize(torch.from_numpy(x))
    np.testing.assert_allclose(zt.numpy(), z, atol=1e-5)
    n2 = Normalizer.from_state_dict(n.state_dict())
    np.testing.assert_allclose(n2.normalize(x), z)
