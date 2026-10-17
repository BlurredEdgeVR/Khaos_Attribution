# The Guild Register on its own server

One small Ubuntu 24.04 server, SQLite, a few files. `register-server.sh` does the whole setup as root and can be run again after a change of tag.

```
ssh root@<server>
git clone https://github.com/khaos-foundation/Khaos_Attribution.git /root/Khaos_Attribution
REGISTER_DOMAIN=register.khaos.dev REGISTER_TAG=v0.27.0 bash /root/Khaos_Attribution/deploy/register-server.sh
```

What it leaves behind:

- `/srv/register/`: the checkout at the tag, the venv (the `registry` and `service` extras only), `data/` (the two databases, the mirror files, the key), `backups/`.
- `/etc/register.env`: the service's settings. `REGISTER_DEV_KEY=1` until the ceremony, with `REGISTER_DEV_KEY_PUBLIC=1` beside it: a development key serves behind the proxy only when the deploy says so (binding loopback keeps nothing private once Caddy fronts it).
- `register.service` on loopback port 8912; Caddy on 80 and 443 with the certificate for the domain; the firewall open on 22, 80 and 443.
- `register-backup.timer`: nightly at 03:30, a consistent copy of both databases and the mirror into `backups/`, kept 30 days, sent to `BACKUP_TARGET` over ssh when one is set in `/etc/register.env` (a Hetzner Storage Box takes rsync over ssh on port 23). The key is never in a backup.
- `register-restore.sh <archive>`: the way back, ending with the ledger check.

Accounts and invites are made on the server with `register-cli account <name>` and `register-cli invite <name> --role workshop|space` (the register's commands as the service user, with its settings). An invite is a one-time string handed to one machine.

## Updating

Update the register by re-running `register-server.sh` with the new `REGISTER_TAG`, never by checking out a tag and restarting by hand. The script also brings `/etc/register.env` up to what the new release needs: from 0.37.0 a development key serves behind Caddy only with `REGISTER_DEV_KEY_PUBLIC=1` beside `REGISTER_DEV_KEY=1`, and a settings file without that line makes the service refuse to start.

## The key

The service countersigns with the Ed25519 key at `REGISTER_KEY_FILE`, a PEM file it reads at start. With `REGISTER_DEV_KEY=1` it makes one if none is there and says on every page that nothing it signs is a Guild record. The ceremony replaces that file with the Foundation's key and removes the line: the short version is `register-key make`, then `register-key switch "Name, Name"` (see `key-ceremony.md`). The code signs from a file, not from a hardware token: a key that must never touch a disk needs a signing step the service does not have yet.
