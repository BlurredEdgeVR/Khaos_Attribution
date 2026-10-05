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


class FakeNet:
    """An adapter whose strength scales the decoder's learned parameters."""
    def __init__(self, decoder):
        self.decoder, self.set_to = decoder, []
        self.scale, self.bias = decoder.scale.data.clone(), decoder.bias.data.clone()

    def set_multiplier(self, value):
        self.set_to.append(value)
        self.decoder.scale.data = self.scale * value
        self.decoder.bias.data = self.bias * value


def test_a_pairs_loss_is_the_fixed_forward_mean_over_timesteps_and_takes_no_gradient():
    module, sample = FakeModule(), _sample(3)
    loss = G.pair_loss(module, sample, timesteps=[0.25, 0.75], noise_seed=5)
    noise = torch.randn(sample["target_latents"].shape, generator=torch.Generator().manual_seed(5), dtype=torch.float32)
    x0 = sample["target_latents"]
    expected = []
    for t in (0.25, 0.75):
        out = module.model.decoder(hidden_states=(t * noise + (1 - t) * x0).unsqueeze(0), timestep=torch.full((1,), t),
                                   timestep_r=torch.full((1,), t), attention_mask=None,
                                   encoder_hidden_states=sample["encoder_hidden_states"].unsqueeze(0),
                                   encoder_attention_mask=None, context_latents=sample["context_latents"].unsqueeze(0))[0]
        expected.append(float(torch.nn.functional.mse_loss(out.squeeze(0), noise - x0)))
    assert loss == pytest.approx(sum(expected) / 2, rel=1e-6)
    assert G.pair_loss(module, sample, timesteps=[0.25, 0.75], noise_seed=5) == loss
    assert G.pair_loss(module, sample, timesteps=[0.25, 0.75], noise_seed=6) != loss
    assert all(p.grad is None for p in module.model.decoder.parameters())
    assert not any(c["training"] for c in module.model.decoder.calls[:2]) and module.model.decoder.training
    with pytest.raises(ValueError):
        G.pair_loss(module, sample, timesteps=[], noise_seed=5)


def test_an_outputs_loss_is_read_with_the_adapter_on_then_off_and_left_on(tmp_path):
    module = FakeModule()
    module.lycoris_net = FakeNet(module.model.decoder)
    files = []
    for i in (1, 2):
        path = tmp_path / f"out_seg_{i:03d}.pt"
        torch.save(_sample(i), path)
        files.append(path)
    meta = {"timesteps": [0.25, 0.75], "noise_seed": 9}
    both = G.paired_loss(module, meta, files)
    per = [G.pair_loss(module, G.load_sample(f), timesteps=[0.25, 0.75], noise_seed=G.pair_noise_seed(9, f.name)) for f in files]
    assert both["with"] == pytest.approx(sum(per) / 2) and both["with"] == G.output_loss(module, meta, files)
    assert both["without"] != both["with"] and module.lycoris_net.set_to[:3] == [1.0, 0.0, 1.0]
    module.lycoris_net.set_multiplier(0.0)
    assert G.output_loss(module, meta, files) == pytest.approx(both["without"])
    module.lycoris_net.set_multiplier(1.0)
    boom = FakeModule()
    boom.lycoris_net = FakeNet(boom.model.decoder)
    real, seen = G.output_loss, []

    def fails_with_the_adapter_off(module, meta, files):
        seen.append(module.lycoris_net.set_to[-1])
        if seen[-1] == 0.0:
            raise G.GradientUnavailable("lost mid-measurement")
        return real(module, meta, files)
    G.output_loss = fails_with_the_adapter_off
    try:
        with pytest.raises(G.GradientUnavailable, match="lost mid-measurement"):
            G.paired_loss(boom, meta, files)
    finally:
        G.output_loss = real
    assert seen == [1.0, 0.0] and boom.lycoris_net.set_to[-1] == 1.0, "a measurement that fails with the adapter off still leaves it on"
    decomposed = FakeModule()
    decomposed.lycoris_net = FakeNet(decomposed.model.decoder)
    decomposed.lycoris_net.loras = [type("Layer", (), {"wd": True})()]
    with pytest.raises(G.GradientUnavailable, match="weight-decomposed"):
        G.paired_loss(decomposed, meta, files)
    assert decomposed.lycoris_net.set_to == []
    with pytest.raises(G.GradientUnavailable, match="cannot be switched off"):
        G.paired_loss(FakeModule(), meta, files)


def test_the_rooms_one_call_reads_the_rule_measures_only_when_it_can_and_refuses_by_raising(tmp_path):
    from khaos_attribution import influence as I
    module = FakeModule()
    module.lycoris_net = FakeNet(module.model.decoder)
    path = tmp_path / "out_seg_000.pt"
    torch.save(_sample(1), path)
    meta = {"timesteps": [0.25, 0.75], "noise_seed": 9}
    both = G.paired_loss(module, meta, [path])
    gain = both["without"] - both["with"]
    module.lycoris_net.set_to.clear()
    unread = I.read_abstention(None, module, meta, [path])
    assert unread["checked"] is False and module.lycoris_net.set_to == [], "nothing is measured for a rule that cannot be read"
    numbers = {"gain_min": gain - 1.0, "loss_max": both["with"] + 1.0, "device": "cpu", "compute_dtype": "fp32", "render_seconds": 30}
    ok = I.read_abstention(numbers, module, meta, [path], duration_sec=30)
    assert ok["accepted"] is True and ok["gain"] == pytest.approx(gain, abs=1e-5) and module.lycoris_net.set_to[-1] == 1.0
    with pytest.raises(I.NotAttributable) as refused:
        I.read_abstention({**numbers, "gain_min": gain + 1.0}, module, meta, [path], duration_sec=30)
    assert refused.value.reading["accepted"] is False
    module.lycoris_net.set_to.clear()
    long = I.read_abstention({**numbers, "gain_min": gain + 1.0}, module, meta, [path], duration_sec=200)
    assert long["checked"] is False and module.lycoris_net.set_to == []
    assert G.dtype_tag("torch.bfloat16") == "bf16" and G.dtype_tag(torch.float32) == "fp32" and G.dtype_tag("odd") == "odd"
    # An adapter that cannot be switched off is not read, and is never refused for it.
    fixed = FakeModule()
    cannot = I.read_abstention({**numbers, "gain_min": gain + 1.0}, fixed, meta, [path], duration_sec=30)
    assert cannot["checked"] is False and "cannot be switched off" in cannot["reason"]
    other = I.read_abstention({**numbers, "gain_min": gain + 1.0, "index_sha256": "abc"}, module, meta, [path], duration_sec=30, index_sha256="rebuilt")
    assert other["checked"] is False and "another influence index" in other["reason"]
