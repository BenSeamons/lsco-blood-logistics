#!/bin/bash
# ============================================================
# LSCO Blood Logistics Sim — GCP VPS Setup Script
# Run as root on fresh Ubuntu 24.04
# Usage: bash setup_vps.sh
# ============================================================

set -e  # exit on any error

echo "============================================"
echo "  LSCO Blood Logistics — VPS Setup"
echo "============================================"

# ── 1. System packages ───────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y python3 python3-pip python3-venv git nginx ufw

# ── 2. Firewall ──────────────────────────────────────────────
echo "[2/6] Configuring firewall..."
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw allow 8000
ufw --force enable

# ── 3. Clone repo ────────────────────────────────────────────
echo "[3/6] Cloning repo..."
cd /opt
if [ -d "lsco-blood-logistics" ]; then
    echo "  Repo already exists — pulling latest..."
    cd lsco-blood-logistics && git pull
else
    git clone https://github.com/BenSeamons/lsco-blood-logistics.git
    cd lsco-blood-logistics
fi

# ── 4. Python environment ────────────────────────────────────
echo "[4/6] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
deactivate

# ── 5. Generate access token if not set ─────────────────────
echo "[5/6] Configuring access token..."
ENV_FILE="/opt/lsco-blood-logistics/.env"
if [ ! -f "$ENV_FILE" ]; then
    TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    echo "ACCESS_TOKEN=$TOKEN" > "$ENV_FILE"
    echo ""
    echo "  ╔══════════════════════════════════════════════════╗"
    echo "  ║  YOUR ACCESS TOKEN (save this somewhere safe):  ║"
    echo "  ║  $TOKEN  ║"
    echo "  ╚══════════════════════════════════════════════════╝"
    echo ""
else
    echo "  .env already exists — keeping existing token."
fi

# ── 6. Create systemd service ────────────────────────────────
echo "[6/6] Creating systemd service..."
cat > /etc/systemd/system/blood-sim.service << 'EOF'
[Unit]
Description=LSCO Blood Logistics Simulation Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/lsco-blood-logistics
EnvironmentFile=/opt/lsco-blood-logistics/.env
ExecStart=/opt/lsco-blood-logistics/venv/bin/gunicorn server:app \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --timeout 120 \
    --access-logfile /var/log/blood-sim-access.log \
    --error-logfile /var/log/blood-sim-error.log
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable blood-sim
systemctl restart blood-sim

# ── Done ─────────────────────────────────────────────────────
EXTERNAL_IP=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip \
    2>/dev/null || echo "YOUR_IP")

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  App running at:  http://$EXTERNAL_IP:8000"
echo ""
echo "  Useful commands:"
echo "    systemctl status blood-sim     # check status"
echo "    systemctl restart blood-sim    # restart"
echo "    journalctl -u blood-sim -f     # live logs"
echo "    cat /opt/lsco-blood-logistics/.env  # view token"
echo "============================================"
