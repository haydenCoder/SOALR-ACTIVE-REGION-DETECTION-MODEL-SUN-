#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-data/raw/smarp_sample}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Cloning SMARPs sample repository..."
git clone --depth 1 https://github.com/mbobra/SMARPs.git "${TMP_DIR}/SMARPs" >/dev/null

mkdir -p "${OUTPUT_DIR}"
cp -f "${TMP_DIR}/SMARPs/example_gallery/files/"*.fits "${OUTPUT_DIR}/"

echo "Copied sample FITS files to ${OUTPUT_DIR}:"
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.fits' -print | sort
