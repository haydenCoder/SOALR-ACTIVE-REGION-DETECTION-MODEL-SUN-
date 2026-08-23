#!/usr/bin/env bash
set -euo pipefail

# 15 GB RAM / 4 CPU preset for solar active region segmentation.
# Physically meaningful 3-channel stack:
#   - AIA 171 Å : cooler coronal loops / quiet active-region plasma
#   - AIA 193 Å : hotter coronal emission and active coronal structure
#   - HMI LOS magnetogram : magnetic-field driver of active regions
#
# User note: AIA "191" is usually a typo for the standard AIA 193 Å channel,
# so this preset uses 171 + 193 + HMI.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python3 -m pip install -r requirements.txt

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export PYTHONUNBUFFERED=1

SPLIT="${SPLIT:-train}"
MAX_FRAMES="${MAX_FRAMES:-48}"
PATCH_SIZE="${PATCH_SIZE:-512}"
STRIDE="${STRIDE:-512}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-1000}"
DATASET_DIR="${DATASET_DIR:-data/processed/arpil_3ch_512}"
RUN_DIR="${RUN_DIR:-runs/attention_unet_arpil_171_193_hmi_15gb}"

python3 scripts/build_arpil_3ch_tiles.py \
  --output-dir "${DATASET_DIR}" \
  --split "${SPLIT}" \
  --channels aia171 aia193 hmi_m \
  --patch-size "${PATCH_SIZE}" \
  --stride "${STRIDE}" \
  --max-frames "${MAX_FRAMES}" \
  --min-mask-fraction 0.0025 \
  --keep-empty-every 32 \
  --val-ratio 0.15

python3 train.py \
  --manifest "${DATASET_DIR}/manifest.csv" \
  --channels aia171 aia193 hmi_m \
  --image-size 512 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accumulation-steps "${GRAD_ACCUM}" \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 0 \
  --torch-num-threads 4 \
  --torch-num-interop-threads 1 \
  --output-dir "${RUN_DIR}" \
  --base-channels "${BASE_CHANNELS}" \
  --dropout 0.05 \
  --normalize-mode solar_physics \
  --loss bce_dice \
  --patience 0
