#!/bin/bash
SERVICE_FILE="bag_counter.service"
TARGET_DIR="/etc/systemd/system"

if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: $SERVICE_FILE not found in current directory."
    exit 1
fi

sudo cp "$SERVICE_FILE" "$TARGET_DIR/$SERVICE_FILE"
sudo systemctl daemon-reload
sudo systemctl enable bag_counter.service
sudo systemctl start bag_counter.service
sudo systemctl status bag_counter.service
