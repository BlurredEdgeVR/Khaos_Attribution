"""Stable Audio 3's torch model is built from Stability's config with the text encoder read from the open base
repository: the config names the gated distilled repository, whose encoder files are the same bytes."""
import glob
import json
import os

import pytest

from khaos_attribution.engines.sa3 import loading as L
from khaos_attribution.engines.sa3 import DESCRIPTOR


def test_every_conditioner_naming_the_gated_repository_is_pointed_at_the_open_one():
    cfg = {"model": {"conditioning": {"configs": [
        {"id": "prompt", "type": "t5gemma", "config": {"repo_id": L.GATED_ENCODER_REPO, "subfolder": "t5gemma-b-b-ul2"}},
        {"id": "seconds_total", "type": "number", "config": {"min_val": 0}}]}}, "svd_bases_path": "svd_bases.pt"}
    out = L.with_open_encoder(cfg)
    assert out["model"]["conditioning"]["configs"][0]["config"] == {"repo_id": L.OPEN_ENCODER_REPO, "subfolder": "t5gemma-b-b-ul2"}
    assert out["model"]["conditioning"]["configs"][1] == cfg["model"]["conditioning"]["configs"][1]
    assert cfg["model"]["conditioning"]["configs"][0]["config"]["repo_id"] == L.GATED_ENCODER_REPO, "the input is not changed"


def test_the_real_base_config_in_the_cache_names_the_gated_copy_and_is_rewritten():
    found = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--stabilityai--stable-audio-3-medium-base/snapshots/*/model_config.json"))
    if not found:
        pytest.skip("the base model's config is not in this machine's cache")
    cfg = json.load(open(found[0]))
    gated = f'"repo_id": "{L.GATED_ENCODER_REPO}"'
    assert json.dumps(cfg).count(gated) == 1, "Stability's config names the gated repository once"
    assert gated not in json.dumps(L.with_open_encoder(cfg))


def test_the_descriptor_asks_for_no_gated_set():
    assert not [w for w in DESCRIPTOR.weights if w.get("gated")]
    torch_set = [w for w in DESCRIPTOR.weights if w.get("runtime") == "torch"]
    assert [w["repo"] for w in torch_set] == [L.OPEN_ENCODER_REPO]
    assert "t5gemma-b-b-ul2/*" in torch_set[0]["patterns"] and "svd_bases.pt" not in " ".join(torch_set[0]["patterns"])
