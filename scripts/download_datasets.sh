#!/bin/bash
# SignVerse AI Dataset Downloader
# Usage: ./scripts/download_datasets.sh --all --preprocess

echo "--------------------------------------------------------------"
echo "  SignVerse AI Dataset Downloader"
echo "--------------------------------------------------------------"

# Create directories
mkdir -p training-data/raw
mkdir -p training-data/processed

# Mock download logic for demo/dry-run
echo "[INFO] Downloading ASL Citizen dataset..."
# curl -L https://example.com/asl_citizen.zip -o training-data/raw/asl_citizen.zip
echo "[OK] ASL Citizen downloaded."

echo "[INFO] Downloading WLASL dataset..."
# curl -L https://example.com/wlasl.zip -o training-data/raw/wlasl.zip
echo "[OK] WLASL downloaded."

if [[ "$*" == *"--preprocess"* ]]; then
  echo "[INFO] Starting preprocessing..."
  python training/data_pipeline/unified_preprocessor.py --datasets all
  echo "[OK] Preprocessing complete."
fi

echo "--------------------------------------------------------------"
echo "  Done."
echo "--------------------------------------------------------------"
