#!/usr/bin/env bash
# Download the competition data with the Kaggle CLI.
# Requires: `pip install kaggle`, an API token in ~/.kaggle/kaggle.json,
# and access to the (private) competition.
set -euo pipefail

COMPETITION="heavy-equipment-selling-price-prediction-challenge"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"

mkdir -p "$DEST"
kaggle competitions download -c "$COMPETITION" -p "$DEST"
unzip -o "$DEST/$COMPETITION.zip" -d "$DEST"
rm -f "$DEST/$COMPETITION.zip"
echo "Data saved to $DEST"
ls -lh "$DEST"
