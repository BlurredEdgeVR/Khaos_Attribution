#!/usr/bin/env bash
# Put a backup back: stop the register, replace the two databases and the mirror, start it, check the ledger.
#   register-restore.sh /srv/register/backups/register-<stamp>.tar.gz
set -euo pipefail
ARCHIVE="${1:?the backup archive}"
DATA="${REGISTER_DATA_DIR:-/srv/register/data}"
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
sha256sum -c "$ARCHIVE.sha256"
WORK="$(mktemp -d)"
tar -C "$WORK" -xzf "$ARCHIVE"
systemctl stop register.service
for db in register index; do
  if [ -f "$WORK/$db.sqlite3" ]; then
    # The copy is a whole database; a write-ahead log left from the old one must not be replayed over it.
    rm -f "$DATA/$db.sqlite3-wal" "$DATA/$db.sqlite3-shm"
    install -o register -g register -m 640 "$WORK/$db.sqlite3" "$DATA/$db.sqlite3"
  fi
done
if [ -d "$WORK/mirror" ]; then rm -rf "$DATA/mirror"; cp -a "$WORK/mirror" "$DATA/mirror"; chown -R register:register "$DATA/mirror"; fi
rm -rf "$WORK"
systemctl start register.service
sudo -u register bash -c 'set -a; . /etc/register.env; set +a; cd /srv/register && exec venv/bin/python -m register_service check'
