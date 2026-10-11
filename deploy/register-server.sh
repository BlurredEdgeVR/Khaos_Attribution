#!/usr/bin/env bash
# The Guild Register on one small Ubuntu 24.04 server, run once as root and safe to run again:
# updates on a timer, a firewall open on 22, 80 and 443, a service user with no shell, the register under
# systemd on loopback port 8912, Caddy in front of it with HTTPS, a nightly backup to a storage box.
#   REGISTER_DOMAIN=register.khaos.dev REGISTER_TAG=v0.27.0 bash register-server.sh
# Until the Guardians' key exists the register signs with a development key and says so on every page;
# the ceremony replaces /srv/register/data/register-key.pem and clears REGISTER_DEV_KEY in /etc/register.env.
set -euo pipefail

DOMAIN="${REGISTER_DOMAIN:?set REGISTER_DOMAIN to the name people will open}"
TAG="${REGISTER_TAG:-main}"
REPO="${REGISTER_REPO:-https://github.com/BlurredEdgeVR/Khaos_Attribution.git}"
USER_NAME=register
HOME_DIR=/srv/register
CHECKOUT="$HOME_DIR/Khaos_Attribution"
VENV="$HOME_DIR/venv"
DATA="$HOME_DIR/data"
ENV_FILE=/etc/register.env
PORT=8912

[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
export DEBIAN_FRONTEND=noninteractive

echo "== packages"
apt-get update -q
apt-get install -y -q unattended-upgrades ufw fail2ban git curl sqlite3 rsync debian-keyring debian-archive-keyring apt-transport-https ca-certificates
dpkg-reconfigure -f noninteractive unattended-upgrades

echo "== firewall"
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
systemctl enable --now fail2ban

echo "== caddy"
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -q && apt-get install -y -q caddy
fi

echo "== service user"
id -u "$USER_NAME" >/dev/null 2>&1 || useradd --system --home-dir "$HOME_DIR" --create-home --shell /usr/sbin/nologin "$USER_NAME"
mkdir -p "$DATA" "$HOME_DIR/backups"
chown -R "$USER_NAME:$USER_NAME" "$HOME_DIR"
chmod 750 "$HOME_DIR"

echo "== uv and the code at $TAG"
sudo -u "$USER_NAME" -H bash -c '
  set -euo pipefail
  cd "$HOME"
  [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
  if [ -d "'"$CHECKOUT"'/.git" ]; then git -C "'"$CHECKOUT"'" fetch -q --tags origin; else git clone -q "'"$REPO"'" "'"$CHECKOUT"'"; fi
  git -C "'"$CHECKOUT"'" checkout -q --detach "'"$TAG"'"
  [ -d "'"$VENV"'" ] || uv venv -q --python 3.12 "'"$VENV"'"
  uv pip install -q --python "'"$VENV"'/bin/python" "'"$CHECKOUT"'[registry,service]"
'

echo "== settings"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<ENV
REGISTER_DATA_DIR=$DATA
REGISTER_KEY_FILE=$DATA/register-key.pem
REGISTER_PUBLIC_URL=https://$DOMAIN
# 1 until the Guardians' key is placed at REGISTER_KEY_FILE by the ceremony; then delete this line and restart.
REGISTER_DEV_KEY=1
# The storage box for the nightly backup, as user@host:path over ssh (Hetzner: port 23); empty keeps backups local only.
BACKUP_TARGET=
BACKUP_PORT=23
ENV
  chmod 640 "$ENV_FILE"; chown root:"$USER_NAME" "$ENV_FILE"
fi

echo "== systemd"
cat > /etc/systemd/system/register.service <<UNIT
[Unit]
Description=The Guild Register
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_NAME
Group=$USER_NAME
EnvironmentFile=$ENV_FILE
WorkingDirectory=$HOME_DIR
ExecStart=$VENV/bin/python -m register_service serve --host 127.0.0.1 --port $PORT
Restart=on-failure
RestartSec=3
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DATA
PrivateTmp=true
PrivateDevices=true
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
LockPersonality=true
MemoryDenyWriteExecute=true

[Install]
WantedBy=multi-user.target
UNIT

install -m 755 "$(dirname "$0")/register-backup.sh" /usr/local/bin/register-backup
cat > /etc/systemd/system/register-backup.service <<UNIT
[Unit]
Description=Back up the Guild Register's data

[Service]
Type=oneshot
User=$USER_NAME
EnvironmentFile=$ENV_FILE
ExecStart=/usr/local/bin/register-backup
UNIT
cat > /etc/systemd/system/register-backup.timer <<UNIT
[Unit]
Description=Nightly backup of the Guild Register

[Timer]
OnCalendar=*-*-* 03:30:00
RandomizedDelaySec=15m
Persistent=true

[Install]
WantedBy=timers.target
UNIT

echo "== caddy site"
cat > /etc/caddy/Caddyfile <<CADDY
$DOMAIN {
	encode zstd gzip
	request_body {
		max_size 10MB
	}
	reverse_proxy 127.0.0.1:$PORT
}
CADDY

systemctl daemon-reload
systemctl enable --now register.service register-backup.timer
systemctl restart caddy
sleep 2
systemctl --no-pager --quiet is-active register.service && echo "register: running" || { journalctl -u register.service --no-pager -n 30; exit 1; }
curl -fsS "http://127.0.0.1:$PORT/health" && echo
echo
echo "Done. Next: the first account and invites, as the service user:"
echo "  sudo -u $USER_NAME env \$(grep -v '^#' $ENV_FILE | xargs) $VENV/bin/python -m register_service account <name>"
echo "  sudo -u $USER_NAME env \$(grep -v '^#' $ENV_FILE | xargs) $VENV/bin/python -m register_service invite <name> --role workshop"
