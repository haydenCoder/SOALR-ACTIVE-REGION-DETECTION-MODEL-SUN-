#!/usr/bin/env bash
set -euo pipefail

# Assumption: user meant AIA 193 Å when they wrote 191.
# 3 channels = AIA 171, AIA 193, HMI LOS magnetogram.
# Designed for a ~4 GB RAM environment.

MANIFEST="${1:-data/processed/arpil_3ch_lowram/manifest.csv}"
OUTPUT_DIR="${2:-runs/attention_unet_arpil_171_193_hmi_lowram}"

python train.py \
  --manifest "${MANIFEST}" \
  --channels aia171 aia193 hmi_m \
  --image-size 256 \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --epochs 1000 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --num-workers 0 \
  --base-channels 24 \
  --dropout 0.05 \
  --normalize-mode solar_physics \
  --loss bce_dice \
  --patience 0 \
  --output-dir "${OUTPUT_DIR}"
