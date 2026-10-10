"""The register's commands: serve, and the operator's account and invite management.
    python -m register_service serve [--host 127.0.0.1] [--port 8900]
    python -m register_service account <name>
    python -m register_service invite <account>
    python -m register_service revoke <machine_key_id> <reason>
    python -m register_service check
"""

import argparse
import sys

from register_service.app import create_app, settings
from register_service.store import Store


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8900)
    sub.add_parser("account").add_argument("name")
    sub.add_parser("invite").add_argument("account")
    r = sub.add_parser("revoke"); r.add_argument("key_id"); r.add_argument("reason")
    sub.add_parser("check")
    a = p.parse_args(argv)
    cfg = settings()
    if a.cmd == "serve":
        import uvicorn  # noqa: PLC0415
        uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_level="info")
        return 0
    store = Store(cfg["data"] / "register.sqlite3")
    if a.cmd == "account":
        print(f"account {a.name}: id {store.create_account(a.name)}")
    elif a.cmd == "invite":
        print(store.create_invite(a.account))
    elif a.cmd == "revoke":
        store.revoke_machine(a.key_id, a.reason); print("revoked")
    elif a.cmd == "check":
        ok = store.ledger_ok()
        print(f"ledger {'unbroken' if ok else 'BROKEN'}: {len(store.ledger())} entries, {len(store.serials())} models")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
