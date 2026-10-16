"""The checklist: the mock engine passes it live, ACE-Step passes its surface without its weights, and an engine
that lacks a slot or a name fails with the finding named."""
import types

import pytest

from khaos_attribution import engines as E
from khaos_attribution.engines import checklist as C


def test_the_mock_engine_passes_the_whole_checklist_live(tmp_path):
    result = C.check("mock", tmp_path, live=True)
    assert result.passed, result.report()
    names = {f.name for f in result.findings}
    assert {"live.deterministic", "live.seed_matters", "live.sample_rate", "live.duration", "live.cancel",
            "render.words", "measure.memory_plan"} <= names


def test_acestep_passes_the_surface_checks_without_its_weights(tmp_path):
    result = C.check("acestep-1.5", tmp_path, live=False)
    assert result.passed, result.report()
    assert not any(f.name.startswith("live.") for f in result.findings)


def test_an_unknown_or_broken_engine_fails_with_the_finding_named(tmp_path, monkeypatch):
    assert not C.check("nope", tmp_path).passed

    class Engine:
        descriptor = E.descriptor("mock")

        @property
        def render(self):
            return E.engine("mock").render

        @property
        def train(self):
            raise ImportError("no trainer here")

        @property
        def measure(self):
            return types.SimpleNamespace()
    monkeypatch.setitem(E.KNOWN_ENGINES, "broken", "khaos_attribution.engines.broken")
    monkeypatch.setattr(C, "descriptor", lambda engine_id: E.descriptor("mock"))
    monkeypatch.setattr(C, "engine", lambda engine_id: Engine())
    result = C.check("broken", tmp_path)
    failed = {f.name for f in result.failed}
    assert "slot.train" in failed and "descriptor.id" in failed
    assert "no trainer here" in result.report()
    assert not any(f.name.startswith(("render.", "measure.")) for f in result.findings), "a missing slot stops the walk"


def test_the_command_line_reports_and_exits_by_the_verdict(tmp_path, capsys):
    assert C.main(["mock", "--live", "--workdir", str(tmp_path)]) == 0
    assert "engine mock: PASS" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        C.main(["nope"])
