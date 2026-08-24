"""reset_history's unreadable-package-data fallback — the exact 0.16.0
regression its comment cites (package data not shipped → every freed ID
quietly read as "unknown"). Pinned so it cannot silently recur."""

import khaos_attribution.watermark as wm


def test_reset_history_survives_unreadable_package_data(monkeypatch, caplog):
    monkeypatch.setattr(wm, "_RESETS", None)   # force a re-read
    import importlib.resources

    class _Boom:
        def joinpath(self, *_a):
            return self

        def read_text(self, *a, **k):
            raise OSError("package data missing")

    monkeypatch.setattr(importlib.resources, "files", lambda *_a: _Boom())
    out = wm.reset_history()
    assert out == []                          # degraded, never raising
    assert any("reset history unavailable" in r.message for r in caplog.records)
    monkeypatch.setattr(wm, "_RESETS", None)  # leave no poisoned cache


def test_reset_history_deepcopies(monkeypatch):
    monkeypatch.setattr(wm, "_RESETS", None)
    first = wm.reset_history()
    if first:                                 # real package data present
        first[0]["mutated"] = True
        assert "mutated" not in wm.reset_history()[0]
