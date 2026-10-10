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


def _machine(register, account, role="workshop"):
    app, store, http = register
    key = R.SigningKey.generate()
    client = R.RegisterClient("https://register.example", key)
    client._post = _Transport(http, key)
    client.enrol(store.create_invite(account, role), "a machine")
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


def test_withdrawal_is_the_owning_accounts_alone_and_only_a_space_reports_serving(register):
    app, store, http = register
    key, client = _machine(register, "artist-one")
    _, stranger = _machine(register, "artist-two")
    _, space = _machine(register, "artist-two", role="space")
    serial = client.publish(_signed(key))["serial"]
    with pytest.raises(R.RegisterError, match="account"):
        stranger.withdraw(serial, "not mine")
    with pytest.raises(R.RegisterError, match="only an enrolled Space"):
        stranger.served(serial, "https://evil.example", "serving")
    with pytest.raises(R.RegisterError, match="only an enrolled Space"):
        client.served(serial, "https://evil.example", "serving")
    assert space.served(serial, "https://space.example", "serving")["state"] == "serving"
    out = client.withdraw(serial, "the artist withdrew consent")
    assert out["state"] == "withdrawn" and http.get("/api/withdrawn").json()["withdrawn"] == [serial]
    doc = http.get(f"/api/models/{serial}").json()
    assert doc["state"] == "withdrawn" and doc["served"][0]["space_url"] == "https://space.example"
    assert "withdrawn" in http.get(f"/models/{serial}").text
    assert doc["machine_public_pem"].startswith("-----BEGIN PUBLIC KEY-----"), "a reader of the mirror can check the machine's signature"


def test_a_captured_body_cannot_be_replayed(register):
    """A served or withdraw body carries a nonce the register takes once: the same signed bytes again have no second effect."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    _, space = _machine(register, "artist-one", role="space")
    serial = client.publish(_signed(key))["serial"]
    captured = {}

    real = space._post

    def capture(path, body):
        captured[path] = body
        return real(path, body)
    space._post = capture
    assert space.served(serial, "https://space.example", "serving")["state"] == "serving"
    assert space.served(serial, "https://space.example", "removed")["state"] == "removed"
    replayed = real("/api/served", captured["/api/served"])
    assert replayed["state"] == "removed", "the captured bytes get their own first answer back and change nothing"
    assert http.get(f"/api/models/{serial}").json()["served"][0]["state"] == "removed"


def test_a_record_that_passes_the_check_never_breaks_its_page(register):
    """Every shape the page and the views read is checked before a serial is spent."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    for breaker in (lambda r: r["training_set"].__setitem__("tracks", ["t1"]),
                    lambda r: r.__setitem__("provenance", {}),
                    lambda r: r["training_set"]["tracks"][0]["writers"].__setitem__(0, {"name": "W"}),
                    lambda r: r.__setitem__("model", "x")):
        rec = _record(); rec["model"]["machine_key_id"] = key.key_id
        rec["model"]["watermark_payload"] = derive_payload(key.key_id, "run_1")
        breaker(rec)
        with pytest.raises(ValueError):
            R.sign_record(rec, key)
    # malformed but signed bodies are refusals in words, never 500s
    for path, body, status in (("/api/withdraw", {"serial": "abc", "reason": "x", "nonce": "0123456789ab"}, 422),
                               ("/api/served", {"serial": 65536, "space_url": "x", "state": "sideways", "nonce": "0123456789ac"}, 422)):
        data = R.canonical_bytes(body)
        r = http.post(path, content=data, headers={"Content-Type": "application/json", "X-Machine-Key-Id": key.key_id, "X-Machine-Signature": key.sign(data)})
        assert r.status_code == status, (path, r.status_code, r.text)
    data = b"[1]"
    r = http.post("/api/records", content=data, headers={"Content-Type": "application/json", "X-Machine-Key-Id": key.key_id, "X-Machine-Signature": key.sign(data)})
    assert r.status_code == 400


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
    assert "machine" not in store.ledger()[2]["detail"], "the public ledger names the key and the account, never the machine's name"
    assert http.get("/api/public-key").json()["development"] is True and "development key" in http.get("/").text
    assert http.get("/health").json()["ledger_ok"] is True
    lines = http.get("/api/ledger").text.strip().splitlines()
    assert len(lines) == 5 and json.loads(lines[-1])["kind"] == "withdraw"
    store.db.execute("UPDATE ledger SET detail_json = '{\"reason\": \"y\"}' WHERE kind = 'withdraw'")
    assert not store.ledger_ok()
    assert (app.state.cfg["data"] / "mirror" / "models" / f"{serial}.json").is_file()
    assert (app.state.cfg["data"] / "mirror" / "ledger.jsonl").read_text(encoding="utf-8").count("\n") == 5


def test_the_same_signed_body_again_gets_its_first_answer_and_a_refused_one_does_not_burn_its_nonce(register):
    """The nonce is an idempotency key: a lost answer is recovered by sending the same bytes; a 404 can be retried."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    _, space = _machine(register, "artist-one", role="space")
    serial = client.publish(_signed(key))["serial"]
    body = {"serial": serial, "space_url": "https://space.example", "state": "serving", "nonce": "nonce-0001-abcd"}
    first = space._post("/api/served", body)
    again = space._post("/api/served", body)
    assert again == first, "the same body is answered the same, with no second effect"
    assert len(http.get(f"/api/models/{serial}").json()["served"]) == 1
    with pytest.raises(R.RegisterError, match="replay"):
        space._post("/api/served", {**body, "state": "removed"})
    missing = {"serial": 70000, "reason": "x", "nonce": "nonce-0002-abcd"}
    for _ in range(2):
        with pytest.raises(R.RegisterError) as e:
            client._post("/api/withdraw", missing)
        assert e.value.status == 404, "a refusal is the same refusal twice, never a replay"


def test_sync_sets_a_refused_entry_aside_and_goes_on(register, tmp_path):
    """A withdrawal the register refuses for what it is does not hold the publishes behind it."""
    app, store, http = register
    key, client = _machine(register, "artist-one")
    box = R.Outbox(tmp_path)
    box.append("withdraw", {"serial": 70000, "reason": "never issued", "nonce": "0123456789ab"})
    box.append("publish", _signed(key))
    done = client.sync(box)
    assert [a["serial"] for _, a in done] == [R.FIRST_SERIAL] and not box.pending()
    refused = box.refused()
    assert len(refused) == 1 and refused[0]["answer"]["status"] == 404
    assert box.drop("nothing") is False
    eid = box.append("served", {"serial": R.FIRST_SERIAL, "space_url": "x", "state": "serving", "nonce": "0123456789ac"})
    assert box.drop(eid) is True and not box.pending()
