"""The register end to end: enrolment by invite, a numbered countersigned record, no doubles, withdrawal by
the owning account only, where it is served, the three public views, and an unbroken ledger."""
import json
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution.watermark import derive_payload  # noqa: E402
from register_service.app import create_app, public_view  # noqa: E402
from register_service.store import Store  # noqa: E402
from tests.test_registry import _record  # noqa: E402


class _Transport:
    """The client's one HTTP call, routed into the test app."""

    def __init__(self, client: TestClient, key: R.SigningKey):
        self.client, self.key = client, key

    def __call__(self, path, body):
        data = R.canonical_bytes(body)
        r = self.client.post(path, content=data, headers={"Content-Type": "application/json", "X-Machine-Key-Id": self.key.key_id,
                                                          "X-Machine-Signature": self.key.sign(data)})
        if r.status_code >= 400:
            raise R.RegisterError(r.status_code, r.text)
        return r.json()


@pytest.fixture
def register(tmp_path):
    cfg = {"data": tmp_path / "data", "key_file": tmp_path / "data" / "register-key.pem", "dev_key": True, "public_url": "https://register.example"}
    app = create_app(cfg)
    store: Store = app.state.store
    store.create_account("artist-one")
    store.create_account("artist-two")
    return app, store, TestClient(app)


def _machine(register, account):
    app, store, http = register
    key = R.SigningKey.generate()
    client = R.RegisterClient("https://register.example", key)
    client._post = _Transport(http, key)
    client.enrol(store.create_invite(account), "a machine")
    return key, client


def _signed(key, **over):
    rec = _record(**over)
    rec["model"]["machine_key_id"] = key.key_id
    rec["model"]["watermark_payload"] = derive_payload(key.key_id, rec["model"]["run_id"])
    return R.sign_record(rec, key)


def test_the_register_needs_a_dev_or_real_key_and_never_makes_one_in_production(tmp_path):
    with pytest.raises(SystemExit, match="ceremony"):
        create_app({"data": tmp_path, "key_file": tmp_path / "none.pem", "dev_key": False, "public_url": "x"})


def test_enrolment_takes_an_invite_once_and_an_unenrolled_machine_cannot_publish(register):
    app, store, http = register
    key = R.SigningKey.generate()
    client = R.RegisterClient("x", key); client._post = _Transport(http, key)
    with pytest.raises(R.RegisterError, match="not enrolled"):
        client.publish(_signed(key))
    code = store.create_invite("artist-one")
    assert client.enrol(code, "laptop")["account"] == "artist-one"
    other = R.SigningKey.generate(); c2 = R.RegisterClient("x", other); c2._post = _Transport(http, other)
    with pytest.raises(R.RegisterError, match="already used"):
        c2.enrol(code, "another")


def test_a_record_gets_a_serial_from_65536_a_valid_countersignature_and_the_same_serial_twice(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    signed = _signed(key)
    first = client.publish(signed)
    assert first["serial"] == R.FIRST_SERIAL and first["already"] is False
    pem = http.get("/api/public-key").json()["public_key_pem"]
    assert R.countersignature_valid(first["countersignature"], signed["record_sha256"], pem)
    again = client.publish(signed)
    assert again["serial"] == R.FIRST_SERIAL and again["already"] is True
    second = client.publish(_signed(key, published_at_utc="2026-10-11T00:00:00Z"))
    assert second["serial"] == R.FIRST_SERIAL + 1
    page = http.get(f"/models/{R.FIRST_SERIAL}")
    assert page.status_code == 200 and "Register No. 65536" in page.text and "One" in page.text
    doc = http.get(f"/api/models/{R.FIRST_SERIAL}").json()
    assert doc["serial"] == R.FIRST_SERIAL and doc["state"] == "registered" and doc["record"]["training_set"]["tracks"][0].get("writers") is None


def test_a_record_naming_another_machine_or_signed_by_another_key_is_refused(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    other, _ = _machine(register, "artist-two")
    signed = _signed(other)
    with pytest.raises(R.RegisterError, match="signature|key"):
        client.publish(signed)
    rec = _record(); rec["model"]["machine_key_id"] = other.key_id
    with pytest.raises(R.RegisterError, match="another machine"):
        client.publish(R.sign_record(rec, key))


def test_withdrawal_is_the_owning_accounts_alone_and_served_is_recorded(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    _, stranger = _machine(register, "artist-two")
    serial = client.publish(_signed(key))["serial"]
    with pytest.raises(R.RegisterError, match="account"):
        stranger.withdraw(serial, "not mine")
    assert stranger.served(serial, "https://space.example", "serving")["state"] == "serving"
    out = client.withdraw(serial, "the artist withdrew consent")
    assert out["state"] == "withdrawn" and http.get("/api/withdrawn").json()["withdrawn"] == [serial]
    doc = http.get(f"/api/models/{serial}").json()
    assert doc["state"] == "withdrawn" and doc["served"][0]["space_url"] == "https://space.example"
    assert "withdrawn" in http.get(f"/models/{serial}").text


def test_the_three_public_views_show_what_the_artist_chose():
    rec = _record(public_view="model")
    v = public_view(rec)["training_set"]
    assert v["tracks"] == [] and v["track_count"] == 1
    v = public_view(_record(public_view="tracks"))["training_set"]
    assert v["tracks"][0]["title"] == "One" and "writers" not in v["tracks"][0]
    full = _record(public_view="writers"); full["training_set"]["writers_agreed"] = True
    assert public_view(full)["training_set"]["tracks"][0]["writers"][0]["name"] == "W"


def test_the_ledger_chains_every_entry_and_a_changed_entry_breaks_it(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    serial = client.publish(_signed(key))["serial"]
    client.withdraw(serial, "x")
    kinds = [e["kind"] for e in store.ledger()]
    assert kinds == ["account", "account", "enrol", "publish", "withdraw"] and store.ledger_ok()
    assert http.get("/health").json()["ledger_ok"] is True
    lines = http.get("/api/ledger").text.strip().splitlines()
    assert len(lines) == 5 and json.loads(lines[-1])["kind"] == "withdraw"
    store.db.execute("UPDATE ledger SET detail_json = '{\"reason\": \"y\"}' WHERE kind = 'withdraw'")
    assert not store.ledger_ok()
    assert (app.state.cfg["data"] / "mirror" / "models" / f"{serial}.json").is_file()
    assert (app.state.cfg["data"] / "mirror" / "ledger.jsonl").read_text(encoding="utf-8").count("\n") == 5
