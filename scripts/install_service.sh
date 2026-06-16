#!/bin/bash
# Installs and enables systemd timers to run the counter 6 AM – 6 PM daily.
# Run once after cloning the repo on a new Jetson.
# Usage: sudo ./scripts/install_service.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY_DIR="${REPO_DIR}/deploy"

echo "Installing from: ${REPO_DIR}"

# Copy repo to /opt so systemd has a stable path
sudo cp -r "${REPO_DIR}" /opt/jetson-yolo-cpp

# Install systemd units
sudo cp "${DEPLOY_DIR}/vehicle-counter.service" /etc/systemd/system/
sudo cp "${DEPLOY_DIR}/start.timer"             /etc/systemd/system/
sudo cp "${DEPLOY_DIR}/stop.timer"              /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable vehicle-counter.service
sudo systemctl enable start.timer
sudo systemctl enable stop.timer
sudo systemctl start start.timer
sudo systemctl start stop.timer

echo ""
echo "Installed. The counter will start at 06:00 and stop at 18:00 daily."
echo "To start immediately: sudo systemctl start vehicle-counter.service"
echo "To check status:      sudo systemctl status vehicle-counter.service"
