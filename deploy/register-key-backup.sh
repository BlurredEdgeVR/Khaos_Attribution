#!/usr/bin/env bash
# The key's only backup, locked to the directors' YubiKeys; run on the laptop between register-key make and switch.
#   register-key-backup.sh lock <out.age> <recipient> <recipient>...   the staged key, encrypted on the way in
#   register-key-backup.sh check <out.age> <identity-file>             one YubiKey opens it to the server's key
# Recipients and identity files come from `age-plugin-yubikey` with each key plugged in; neither is secret.
set -euo pipefail
HOST="${REGISTER_HOST:-register}"
STAGED="${REGISTER_STAGED:-/srv/register/data/staged/register-key.pem}"

server_hash() { ssh "$HOST" "sudo sha256sum $STAGED" | cut -d' ' -f1; }

case "${1:-}" in
lock)
  OUT="${2:?the encrypted file to write}"; shift 2
  [ "$#" -ge 2 ] || { echo "name every director's recipient: two at least"; exit 1; }
  [ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
  args=(); for r in "$@"; do args+=(-r "$r"); done
  # The key passes through memory only: what lands on this disk is already encrypted.
  ssh "$HOST" "sudo cat $STAGED" | age "${args[@]}" -o "$OUT"
  echo "locked to $# keys: $OUT"
  echo "server's key sha256: $(server_hash)"
  echo "Now check it opens with each YubiKey: register-key-backup.sh check $OUT <identity-file>"
  ;;
check)
  OUT="${2:?the encrypted file}"; ID="${3:?the identity file of the YubiKey plugged in}"
  GOT="$(age -d -i "$ID" "$OUT" | shasum -a 256 | cut -d' ' -f1)"
  WANT="$(server_hash)"
  [ "$GOT" = "$WANT" ] && echo "opens, and matches the server's key ($WANT)" || { echo "DOES NOT MATCH: $GOT vs $WANT"; exit 1; }
  ;;
*)
  sed -n '2,5p' "$0"; exit 1 ;;
esac
