#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-solar-ar-attention-unet}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

GPU_ARGS=()
if [[ "${USE_GPU:-1}" == "1" ]]; then
  GPU_ARGS+=(--gpus all)
fi

echo "Building Docker image: ${IMAGE_NAME}"
docker build --pull -t "${IMAGE_NAME}" .

echo "Running dataset download and extraction inside the container"
docker run --rm -it \
  -v "${PROJECT_ROOT}:/workspace" \
  "${IMAGE_NAME}" \
  bash scripts/download_uad_dataset.sh data/raw

echo "Creating manifest"
docker run --rm -it \
  -v "${PROJECT_ROOT}:/workspace" \
  "${IMAGE_NAME}" \
  python scripts/prepare_uad_manifest.py --raw-root data/raw/Solar_data_UAD --output data/processed/uad_manifest.csv

echo "Starting training"
docker run --rm -it "${GPU_ARGS[@]}" \
  -v "${PROJECT_ROOT}:/workspace" \
  "${IMAGE_NAME}" \
  python train.py \
    --manifest data/processed/uad_manifest.csv \
    --channels 171 195 284 304 \
    --image-size 256 \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-3 \
    --amp \
    --output-dir runs/attention_unet_uad
