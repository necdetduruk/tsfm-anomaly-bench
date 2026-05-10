#!/usr/bin/env bash
# Download SMD (Server Machine Dataset) from the OmniAnomaly repo.
# Idempotent: skips download if data already exists.

set -euo pipefail

DATA_DIR="data/smd"
TMP_DIR=".smd_tmp"

if [ -d "$DATA_DIR/train" ] && [ -n "$(ls -A "$DATA_DIR/train" 2>/dev/null)" ]; then
  echo "SMD already exists at $DATA_DIR, skipping."
  exit 0
fi

echo "Cloning OmniAnomaly repo for SMD..."
rm -rf "$TMP_DIR"
git clone --depth 1 https://github.com/NetManAIOps/OmniAnomaly.git "$TMP_DIR"

echo "Copying ServerMachineDataset to $DATA_DIR..."
mkdir -p "$DATA_DIR"
cp -r "$TMP_DIR/ServerMachineDataset/"* "$DATA_DIR/"

echo "Cleaning up..."
rm -rf "$TMP_DIR"

echo "Done. SMD available at $DATA_DIR/"
echo "Train files:  $(ls "$DATA_DIR/train" | wc -l)"
echo "Test files:   $(ls "$DATA_DIR/test" | wc -l)"
echo "Label files:  $(ls "$DATA_DIR/test_label" | wc -l)"
