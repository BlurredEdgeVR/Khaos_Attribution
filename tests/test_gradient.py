"""Gradient features: the projection is reproducible from its seed, a pair's
gradient is the trainer's forward with the randomness removed, and an
output's feature is the mean over its windows. Needs torch."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from khaos_attribution import gradient as G  # noqa: E402

FRAMES, CHANNELS, TOKENS, DIM = 4, 3, 2, 5


class FakeDecoder(nn.Module):
    def __init__(self, *, dead=False):
        super().__init__()
        gen = torch.Generator().manual_seed(11)
        self.dead = dead
        self.scale = nn.Parameter(torch.randn(CHANNELS, generator=gen))
        self.bias = nn.Parameter(torch.randn(CHANNELS, generator=gen))
        self.frozen = nn.Parameter(torch.randn(CHANNELS, generator=gen), requires_grad=False)
        self.calls = []

    def forward(self, *, hidden_states, timestep, timestep_r, attention_mask, encoder_hidden_states,
                encoder_attention_mask, context_latents):
        assert torch.equal(timestep, timestep_r)
        self.calls.append({"t": float(timestep[0]), "training": self.training})
        if self.dead:
            return (torch.zeros_like(hidden_states) + 0.0 * self.scale.sum(),)
        conditioning = encoder_hidden_states.mean() + context_latents.mean()
        return (hidden_states * self.scale + self.bias * conditioning + hidden_states * self.frozen,)


class FakeModel(nn.Module):
    def __init__(self, *, dead=False):
        super().__init__()
        self.decoder = FakeDecoder(dead=dead)


class FakeModule:
    def __init__(self, *, dead=False):
        self.model = FakeModel(dead=dead)
        self.device = torch.device("cpu")
        self.device_type = "cpu"
        self.dtype = torch.float32


def _sample(seed):
    g = torch.Generator().manual_seed(seed)
    return {"target_latents": torch.randn(FRAMES, CHANNELS, generator=g),
            "attention_mask": torch.ones(FRAMES),
            "encoder_hidden_states": torch.randn(TOKENS, DIM, generator=g),
            "encoder_attention_mask": torch.ones(TOKENS),
            "context_latents": torch.randn(FRAMES, CHANNELS, generator=g)}


def test_projection_is_reproducible_from_its_seed_and_fingerprinted():
    a, b, c = G.make_projection(1000, 32, 7), G.make_projection(1000, 32, 7), G.make_projection(1000, 32, 8)
    assert np.array_equal(a.buckets, b.buckets) and a.fingerprint() == b.fingerprint() != c.fingerprint()
    v = np.random.default_rng(0).standard_normal(1000)
    assert np.allclose(G.project(v, a), G.project(v, b)) and not np.allclose(G.project(v, a), G.project(v, c))
    with pytest.raises(ValueError, match="not from the same adapter"):
        G.project(v[:10], a)
    meta = {"projection": {"n_params": 1000, "dim": 32, "seed": 7, "fingerprint": a.fingerprint()}}
    assert G.projection_from_meta(meta).fingerprint() == a.fingerprint()
    with pytest.raises(G.GradientUnavailable):
        G.projection_from_meta({"projection": {"n_params": 1000, "dim": 32, "seed": 7, "fingerprint": "nope"}})


def test_helpers_match_the_trainer_rules():
    assert G.bucket_timesteps(10)[0] == 0.05 and G.bucket_timesteps(10)[-1] == 0.95
    assert G.pair_noise_seed(42, "a.pt") != G.pair_noise_seed(42, "b.pt")
    assert G.pair_noise_seed(42, "a.pt") == G.pair_noise_seed(42, "a.pt")
    assert G.track_of_pair("trk_seg_001_w1") == "trk" and G.track_of_pair("bare") == "bare"
    assert np.all(G.l2_normalise([0.0, 0.0]) == 0.0) and np.linalg.norm(G.l2_normalise([3.0, 4.0])) == pytest.approx(1.0)


def test_pair_gradient_is_deterministic_over_the_adapter_only_and_leaves_train_mode():
    m = FakeModule()
    params = G.adapter_parameters(m)
    assert [tuple(p.shape) for p in params] == [(CHANNELS,), (CHANNELS,)]
    g1 = G.pair_gradient(m, _sample(1), timesteps=[0.3, 0.7], noise_seed=5, params=params)
    g2 = G.pair_gradient(m, _sample(1), timesteps=[0.3, 0.7], noise_seed=5, params=params)
    g3 = G.pair_gradient(m, _sample(1), timesteps=[0.3, 0.7], noise_seed=6, params=params)
    assert g1.shape == (2 * CHANNELS,) and np.array_equal(g1, g2) and not np.array_equal(g1, g3)
    assert [round(c["t"], 5) for c in m.model.decoder.calls] == [0.3, 0.7] * 3
    assert all(c["training"] is False for c in m.model.decoder.calls) and m.model.decoder.training
    assert np.all(G.pair_gradient(FakeModule(dead=True), _sample(1), timesteps=[0.5], noise_seed=1) == 0.0)
    frozen = type("M", (), {"model": nn.Linear(2, 2).requires_grad_(False)})()
    with pytest.raises(G.GradientUnavailable):
        G.adapter_parameters(frozen)


def test_output_feature_is_the_mean_over_its_windows(tmp_path):
    m = FakeModule()
    files = []
    for i in range(3):
        p = tmp_path / f"out_seg_{i:03d}.pt"
        torch.save(_sample(10 + i), p)
        files.append(p)
    proj = G.make_projection(2 * CHANNELS, 4, 1)
    meta = {"projection": {"n_params": 2 * CHANNELS, "dim": 4, "seed": 1, "fingerprint": proj.fingerprint()},
            "timesteps": [0.25, 0.75], "noise_seed": 9, "target": "vnorm"}
    feat = G.output_feature(m, meta, files)
    singles = [G.project(G.pair_gradient(m, G.load_sample(p), timesteps=[0.25, 0.75],
                                         noise_seed=G.pair_noise_seed(9, p.name), target="vnorm"), proj)
               for p in files]
    assert np.allclose(feat, np.mean(np.stack(singles), axis=0))
    with pytest.raises(G.GradientUnavailable):
        G.output_feature(m, meta, [])
